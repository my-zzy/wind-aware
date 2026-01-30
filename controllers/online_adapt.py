# controllers/metapinn/online_adapt.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .wrapper import MetaPINNWrapper


# -----------------------------
# Utilities
# -----------------------------
def _wrap_angle_rad(e: float) -> float:
    return (e + math.pi) % (2.0 * math.pi) - math.pi


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class OnlineAdaptConfig:
    # --- data filtering (same spirit as your original script) ---
    min_speed_mps: float = 0.3
    max_roll_rad: float = math.radians(30)
    max_pitch_rad: float = math.radians(30)
    thr_min: float = 0.03
    thr_max: float = 0.97

    # --- update control ---
    enable_online_learn: bool = True
    update_every: int = 10  # call wrapper.online_update() every N push-eligible steps

    # --- warm-start export dhat (dx,dy,dz) estimation ---
    dhat_window_s: float = 5.0  # use last N seconds to compute median
    dhat_stat: str = "median"   # "median" or "mean"

    # --- warm-start file naming ---
    warm_dir: str = "warm_start"


# -----------------------------
# Online adapter
# -----------------------------
class OnlineAdapter:
    """
    Online adaptation manager:
      - owns a MetaPINNWrapper (predict + online_update + warm-start emb/beta)
      - handles push filtering + maybe_update scheduling
      - computes dhat_steady from a stream of dhat(t) (dx,dy,dz) for exporting
      - saves/loads warm-start npz: {task_emb, task_beta (optional), dhat (optional)}
    """

    def __init__(
        self,
        wrapper: MetaPINNWrapper,
        cfg: OnlineAdaptConfig,
        warm_condition: Optional[str] = None,
    ):
        self.wrapper = wrapper
        self.cfg = cfg
        self.warm_condition = warm_condition  # e.g. "10wind" / "ou15" / ...

        # internal store for dhat(t)
        self._dhat_t: list[float] = []
        self._dhat_xyz: list[np.ndarray] = []  # each shape (3,)

        # online update counter
        self._push_steps = 0

    # -----------------------------
    # Warm-start I/O
    # -----------------------------
    def warm_path(self, condition: Optional[str] = None) -> Path:
        cond = condition or self.warm_condition or "default"
        return Path(self.cfg.warm_dir) / f"{cond}.npz"

    def load_warm_start(self, condition: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Loads task embedding/beta into wrapper (inject on first model creation if needed),
        and returns dhat_init if present in npz.
        """
        p = self.warm_path(condition)
        if not p.exists():
            return False, None

        ok = self.wrapper.import_adapt_state(p)
        dhat_init = None
        try:
            npz = np.load(p, allow_pickle=True)
            if "dhat" in npz.files:
                dhat_init = np.asarray(npz["dhat"], dtype=np.float32).reshape(-1)
                # keep only dx,dy,dz if longer
                if dhat_init.size >= 3:
                    dhat_init = dhat_init[:3]
        except Exception:
            dhat_init = None

        return ok, dhat_init

    def save_warm_start(
        self,
        condition: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        dhat_steady: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Save wrapper's current task params + (optional) dhat_steady + extra into a npz.
        """
        p = self.warm_path(condition)
        payload: Dict[str, Any] = {}
        if extra:
            payload.update(extra)

        if dhat_steady is not None:
            payload["dhat"] = np.asarray(dhat_steady, dtype=np.float32).reshape(-1)

        self.wrapper.save_adapt_state(p, extra=payload)
        return p

    # -----------------------------
    # dhat steady estimation
    # -----------------------------
    def record_dhat(self, t: float, dhat_xyz: Sequence[float]) -> None:
        """
        Record dx,dy,dz (the adaptive controller's estimated disturbance accel/force proxy).
        You can call this once per control step.
        """
        v = np.asarray(dhat_xyz, dtype=np.float32).reshape(-1)
        if v.size < 3:
            return
        v = v[:3]
        self._dhat_t.append(float(t))
        self._dhat_xyz.append(v)

        # keep memory bounded: ~ last 2*dhat_window_s seconds
        # (we don't know dt; prune by time)
        t_min = float(t) - 2.0 * float(self.cfg.dhat_window_s)
        # prune from front
        while self._dhat_t and self._dhat_t[0] < t_min:
            self._dhat_t.pop(0)
            self._dhat_xyz.pop(0)

    def estimate_dhat_steady(self, now_t: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Use last dhat_window_s seconds (ending at now_t) to estimate a steady dhat.
        Default: median of samples in that window.
        """
        if not self._dhat_t:
            return None

        if now_t is None:
            now_t = self._dhat_t[-1]
        now_t = float(now_t)

        t0 = now_t - float(self.cfg.dhat_window_s)
        idx = [i for i, tt in enumerate(self._dhat_t) if tt >= t0]
        if not idx:
            arr = np.stack(self._dhat_xyz, axis=0)
        else:
            arr = np.stack([self._dhat_xyz[i] for i in idx], axis=0)

        if arr.shape[0] == 0:
            return None

        if str(self.cfg.dhat_stat).lower() == "mean":
            out = np.mean(arr, axis=0)
        else:
            out = np.median(arr, axis=0)

        return out.astype(np.float32)

    # -----------------------------
    # Online push / update orchestration
    # -----------------------------
    def _should_push(
        self,
        cur_vel: Optional[Sequence[float]] = None,
        att: Optional[Sequence[float]] = None,
        thr: Optional[float] = None,
    ) -> bool:
        """
        Filtering policy: similar to your original.
        - speed too low => skip
        - roll/pitch too large => skip
        - throttle saturated => skip
        """
        if cur_vel is not None:
            v = float(np.linalg.norm(np.asarray(cur_vel, dtype=np.float32)))
            if v < float(self.cfg.min_speed_mps):
                return False

        if att is not None and len(att) >= 2:
            r = float(att[0])
            p = float(att[1])
            if abs(r) > float(self.cfg.max_roll_rad) or abs(p) > float(self.cfg.max_pitch_rad):
                return False

        if thr is not None:
            if thr < float(self.cfg.thr_min) or thr > float(self.cfg.thr_max):
                return False

        return True

    def push_sample(
        self,
        feat_np: np.ndarray,
        target_fa_np: np.ndarray,
        *,
        cur_vel: Optional[Sequence[float]] = None,
        att: Optional[Sequence[float]] = None,
        thr: Optional[float] = None,
    ) -> bool:
        """
        Push one sample into MetaPINNWrapper buffer, if it passes filters.
        Returns True if pushed, False if filtered out or disabled.
        """
        if not self.cfg.enable_online_learn:
            return False

        if not self._should_push(cur_vel=cur_vel, att=att, thr=thr):
            return False

        self.wrapper.push(feat_np, target_fa_np, cur_vel=cur_vel, att=att, thr=thr)
        self._push_steps += 1
        return True

    def maybe_update(self) -> Optional[float]:
        """
        Call online update every cfg.update_every "pushed" steps.
        Returns loss if an update was performed, else None.
        """
        if not self.cfg.enable_online_learn:
            return None
        if self.cfg.update_every <= 0:
            return None

        if self._push_steps > 0 and (self._push_steps % int(self.cfg.update_every) == 0):
            return self.wrapper.online_update()
        return None

    # -----------------------------
    # Convenience: one-step hook
    # -----------------------------
    def step(
        self,
        t: float,
        feat_np: Optional[np.ndarray] = None,
        target_fa_np: Optional[np.ndarray] = None,
        *,
        cur_vel: Optional[Sequence[float]] = None,
        att: Optional[Sequence[float]] = None,
        thr: Optional[float] = None,
        dhat_xyz: Optional[Sequence[float]] = None,
    ) -> Optional[float]:
        """
        A convenience function for your loop:
          - optionally record dhat(t)
          - optionally push (feat, target) if provided
          - maybe_update
        """
        if dhat_xyz is not None:
            self.record_dhat(t, dhat_xyz)

        pushed = False
        if (feat_np is not None) and (target_fa_np is not None):
            pushed = self.push_sample(
                feat_np, target_fa_np,
                cur_vel=cur_vel, att=att, thr=thr,
            )

        if pushed:
            return self.maybe_update()
        return None
