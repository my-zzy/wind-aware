# controllers/safety_shield.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from controllers.interfaces import Obs, Cmd


@dataclass
class SafetyShieldConfig:
    # master switch
    enable: bool = True

    # limits / nominal commands
    max_vel_xy: float = 4.0
    target_z_ned: float = -3.0
    depth_max_m: float = 20.0  # used as a safe fallback when depth is invalid

    # basic distances
    safety_slow_depth_m: float = 2.8
    safety_stop_depth_m: float = 1.2
    safety_climb_m: float = 0.8

    # robust mins (quantiles)
    q_left: float = 0.03
    q_center: float = 0.03
    q_right: float = 0.03
    q_global: float = 0.01

    # anti-oscillation (direction stickiness)
    shield_hold_steps: int = 12
    shield_dir_gap_m: float = 0.6
    shield_emergency_vy: float = 1.2

    # lateral repulsion
    shield_repulse_ref_m: float = 3.0
    shield_k_rep: float = 1.0
    shield_vy_cap: float = 1.0
    shield_alpha: float = 0.85  # low-pass for vy bias (non-emergency only)

    # speed scaling
    shield_min_scale: float = 0.30
    shield_slow_d: float = 2.8
    shield_stop_d: float = 1.2

    # corridor mode (both sides close, front clear)
    corridor_left_right_m: float = 2.5
    corridor_front_clear_m: float = 4.0
    shield_corridor_vy_mul: float = 0.35
    shield_corridor_vx_mul: float = 0.55

    # split regions
    left_x1: float = 0.33
    center_x0: float = 0.40
    center_x1: float = 0.60
    right_x0: float = 0.67


@dataclass
class SafetyShieldState:
    # filtered lateral bias
    vy_shield: float = 0.0
    # -1 go left, +1 go right, 0 none
    avoid_dir: int = 0
    # keep direction for N steps
    avoid_hold: int = 0


class SafetyShield:
    """A small reactive safety layer that preserves the original single-file behavior:
    - Emergency: strong slow-down + climb + instant lateral kick (NO filter).
    - Non-emergency: direction stickiness + smooth slow-down + filtered repulsion + corridor mode.
    """

    def __init__(self, cfg: SafetyShieldConfig):
        self.cfg = cfg
        self.state = SafetyShieldState()

    def _qmin(self, arr: np.ndarray, q: float, fallback: float) -> float:
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            return float(fallback)
        return float(np.quantile(v, q))

    def _clamp_xy(self, vx: float, vy: float) -> Tuple[float, float]:
        mv = float(self.cfg.max_vel_xy)
        return float(np.clip(vx, -mv, mv)), float(np.clip(vy, -mv, mv))

    def apply(self, obs: Obs, cmd: Cmd) -> Cmd:
        depth_m = obs.depth_z_m
        vx_cmd = float(cmd.vx)
        vy_cmd = float(cmd.vy)
        z_cmd = float(cmd.z)

        if (not self.cfg.enable) or (depth_m is None):
            cmd.vx, cmd.vy = self._clamp_xy(vx_cmd, vy_cmd)
            return cmd

        d = depth_m
        h, w = d.shape

        xL1 = int(self.cfg.left_x1 * w)
        xC0 = int(self.cfg.center_x0 * w)
        xC1 = int(self.cfg.center_x1 * w)
        xR0 = int(self.cfg.right_x0 * w)

        L = d[:, :xL1]
        C = d[:, xC0:xC1]
        R = d[:, xR0:]

        # IMPORTANT: do NOT use np.nanmax(d) as fallback (it can become NaN if all-NaN).
        fallback = float(self.cfg.depth_max_m)

        left_min = self._qmin(L, self.cfg.q_left, fallback)
        center_min = self._qmin(C, self.cfg.q_center, fallback)
        right_min = self._qmin(R, self.cfg.q_right, fallback)
        min_depth = self._qmin(d, self.cfg.q_global, fallback)

        # ---------------- EMERGENCY ----------------
        # too close ahead -> no filter, strong slow + climb + lateral kick, lock direction
        if center_min <= self.cfg.safety_stop_depth_m:
            scale = 0.15
            vx_cmd *= scale
            vy_cmd *= scale

            # climb (up) => more negative z in NED
            z_cmd = float(self.cfg.target_z_ned - abs(self.cfg.safety_climb_m))

            # escape to the more open side
            escape_dir = +1 if right_min > left_min else -1
            vy_cmd += float(self.cfg.shield_emergency_vy) * float(escape_dir)

            # reset filter + lock direction
            self.state.vy_shield = 0.0
            self.state.avoid_dir = int(escape_dir)
            self.state.avoid_hold = int(self.cfg.shield_hold_steps)

            cmd.vx, cmd.vy = self._clamp_xy(vx_cmd, vy_cmd)
            cmd.z = float(z_cmd)
            return cmd

        # ---------------- Direction stickiness (avoid flip-flop) ----------------
        if self.state.avoid_hold > 0:
            self.state.avoid_hold -= 1
            avoid_dir = int(self.state.avoid_dir)
        else:
            gap = float(right_min - left_min)
            thr = float(self.cfg.shield_dir_gap_m)
            if gap > thr:
                avoid_dir = +1
            elif gap < -thr:
                avoid_dir = -1
            else:
                avoid_dir = 0

            if avoid_dir != 0:
                self.state.avoid_dir = int(avoid_dir)
                self.state.avoid_hold = int(self.cfg.shield_hold_steps)

        # ---------------- SPEED SCALE (smooth) ----------------
        if min_depth < self.cfg.shield_slow_d:
            # map depth -> scale in [min_scale, 1]
            s = (min_depth - self.cfg.shield_stop_d) / (
                float(self.cfg.shield_slow_d - self.cfg.shield_stop_d) + 1e-6
            )
            s = float(np.clip(s, float(self.cfg.shield_min_scale), 1.0))
        else:
            s = 1.0

        vx_cmd *= s
        vy_cmd *= s

        # ---------------- REPULSION (filtered, non-emergency only) ----------------
        d_ref = float(self.cfg.shield_repulse_ref_m)
        blockL = float(np.clip((d_ref - left_min) / max(1e-6, d_ref), 0.0, 1.0))
        blockR = float(np.clip((d_ref - right_min) / max(1e-6, d_ref), 0.0, 1.0))

        vy_bias_raw = float(self.cfg.shield_k_rep) * float(blockL - blockR)
        vy_bias_raw = float(np.clip(vy_bias_raw, -float(self.cfg.shield_vy_cap), float(self.cfg.shield_vy_cap)))

        # if direction locked, enforce sign
        if avoid_dir != 0:
            vy_bias_raw = float(abs(vy_bias_raw)) * float(avoid_dir)

        alpha = float(self.cfg.shield_alpha)
        self.state.vy_shield = alpha * float(self.state.vy_shield) + (1.0 - alpha) * vy_bias_raw

        vy_cmd += float(self.state.vy_shield)

        # ---------------- Corridor mode ----------------
        corridor = (left_min < float(self.cfg.corridor_left_right_m) and
                    right_min < float(self.cfg.corridor_left_right_m) and
                    center_min > float(self.cfg.corridor_front_clear_m))
        if corridor:
            vy_cmd *= float(self.cfg.shield_corridor_vy_mul)
            vx_cmd *= float(self.cfg.shield_corridor_vx_mul)

        # ---------------- FINAL ----------------
        cmd.vx, cmd.vy = self._clamp_xy(vx_cmd, vy_cmd)
        cmd.z = float(z_cmd)
        return cmd
