# controllers/metapinn/wrapper.py
from __future__ import annotations

import os
import math
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from controllers.mp_imports import ensure_metapinn_on_path
ensure_metapinn_on_path(r"E:\wind-aware-main\Meta-PINN") 
from meta_pinn.config import DEFAULT_OPTIONS
from meta_pinn.model import MetaPINN

UAV_mass = float(DEFAULT_OPTIONS["UAV_mass"])

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _safe_torch_load(path: str):
    """兼容 torch.load(weights_only=...) 与旧版本 torch"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        return torch.load(path, map_location="cpu")


class MetaPINNWrapper:
    """
    从 pinn_online_adaptive.py 抽出的完整 MetaPINNWrapper（可直接用于 E2E / Online Adapt / Warm-start）。
    - predict(): 给定特征向量，输出残差力 fhat（单位：N）
    - push()/maybe_update(): 可选在线更新（仅更新当前 task 的 embedding/β）
    - warm-start I/O: import_adapt_state()/save_adapt_state()
    """

    def __init__(
        self,
        feature_keys: Sequence[str],
        scaler_path: Optional[str] = None,
        load_path: Optional[str] = None,
        lr: float = 1e-3,
        buffer_size: int = 4096,
        batch_size: int = 256,
        update_every: int = 10,
        task_id: int = 0,
        device: Optional[str] = None,
        # gating/limit params
        mag_limit_g: float = 0.6,    # <= 0.6g * m
        step_limit_g: float = 0.15,  # 每步变化 <= 0.15g * m
        ood_decay: float = 12.0,     # exp(-d2/ood_decay)
        unc_scale: float = 0.3,      # g_unc = 1/(1+ema_err/unc_scale)
    ):
        self.feature_keys = list(feature_keys)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # --- scaler ---
        self.scaler: Optional[Dict[str, np.ndarray]] = None
        self.in_dim_from_scaler: Optional[int] = None
        if scaler_path and os.path.isfile(scaler_path):
            npz = np.load(scaler_path)
            self.scaler = {
                "mean": npz["x_mean"].astype(np.float32),
                "std": npz["x_std"].astype(np.float32),
            }
            self.in_dim_from_scaler = int(self.scaler["mean"].shape[0])
            print(f"[MetaPINNWrapper] Loaded scaler: {scaler_path} | dim={self.in_dim_from_scaler}")

        # --- pretrained weights ---
        self._sd: Optional[Dict[str, torch.Tensor]] = None
        if load_path and os.path.isfile(load_path):
            sd_raw = _safe_torch_load(load_path)
            # 兼容: {"model_state_dict": ...} 或直接 state_dict
            if isinstance(sd_raw, dict) and ("model_state_dict" in sd_raw):
                self._sd = sd_raw["model_state_dict"]
            elif isinstance(sd_raw, dict):
                self._sd = sd_raw
            else:
                raise ValueError(f"Unsupported checkpoint format: {type(sd_raw)}")
            print(f"[MetaPINNWrapper] Loaded weights: {load_path}")

        # --- model / optimizer ---
        self.model: Optional[MetaPINN] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.lr = float(lr)

        # online buffer
        self.buffer = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.update_every = int(update_every)
        self.step = 0

        # task
        self.task_id = int(task_id)

        # gating / filters
        self.ema_err = 0.0
        self.last_fhat = np.zeros(3, dtype=np.float32)

        self.mag_limit = float(mag_limit_g) * 9.81 * float(UAV_mass)
        self.step_limit = float(step_limit_g) * 9.81 * float(UAV_mass)
        self.ood_decay = float(ood_decay)
        self.unc_scale = float(unc_scale)

        # warm-start cache (inject on first model creation)
        self._warm_state: Optional[Dict[str, np.ndarray]] = None

    # ---------------- internals ----------------
    def _ensure_model(self, in_dim_runtime: int):
        if self.model is not None:
            return

        # 校验 scaler 维度
        if self.in_dim_from_scaler is not None:
            assert in_dim_runtime == self.in_dim_from_scaler, (
                f"online feature dim({in_dim_runtime}) != scaler dim({self.in_dim_from_scaler})"
            )

        # 从 state_dict 推断结构
        if self._sd is not None:
            if "task_embeddings" not in self._sd:
                raise KeyError("state_dict missing 'task_embeddings' (cannot infer num_tasks/task_dim)")
            num_tasks, task_dim = map(int, self._sd["task_embeddings"].shape)

            # fc1: [hidden_dim, input_dim + task_dim]
            if "fc1.weight" not in self._sd:
                raise KeyError("state_dict missing 'fc1.weight' (cannot infer hidden_dim/input_dim)")
            hidden_dim = int(self._sd["fc1.weight"].shape[0])
            input_dim_from_sd = int(self._sd["fc1.weight"].shape[1] - task_dim)

            assert input_dim_from_sd == in_dim_runtime, (
                f"state_dict input_dim({input_dim_from_sd}) != online feature dim({in_dim_runtime})"
            )

            use_cond_mod = ("cond2beta.0.weight" in self._sd)
            cond_dim = int(self._sd["cond2beta.0.weight"].shape[1]) if use_cond_mod else 1
            use_uncertainty = ("log_vars" in self._sd)
        else:
            # 没有离线权重时给默认值（用于纯在线/调试）
            num_tasks, task_dim, hidden_dim = 1, 128, 384
            use_cond_mod, cond_dim, use_uncertainty = True, 1, True

        self.model = MetaPINN(
            input_dim=in_dim_runtime,
            num_tasks=num_tasks,
            task_dim=task_dim,
            hidden_dim=hidden_dim,
            use_uncertainty=use_uncertainty,
            cond_dim=cond_dim,
            use_cond_mod=use_cond_mod,
            cond_mod_from="target",
            beta_min=0.15,
            beta_max=8.0,
        ).to(self.device)

        if self._sd is not None:
            # strict=False 允许你后续改了一点结构也能载入
            self.model.load_state_dict(self._sd, strict=False)

        # 如果提前 import_adapt_state 了，就在此注入 task embedding / beta
        if self._warm_state is not None:
            with torch.no_grad():
                if "task_emb" in self._warm_state:
                    te = torch.tensor(self._warm_state["task_emb"], dtype=torch.float32, device=self.device)
                    self.model.task_embeddings[self.task_id].copy_(te)
                if ("task_beta" in self._warm_state) and hasattr(self.model, "task_beta_logscale"):
                    tb = torch.tensor(self._warm_state["task_beta"], dtype=torch.float32, device=self.device)
                    self.model.task_beta_logscale[self.task_id].copy_(tb)
            print("[Warm-start] Task params injected to model.")
            self._warm_state = None

        # 只学习当前 task 的 embedding/β，其余冻结
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.task_embeddings.requires_grad = True
        params = [self.model.task_embeddings]

        if hasattr(self.model, "task_beta_logscale"):
            self.model.task_beta_logscale.requires_grad = True
            params.append(self.model.task_beta_logscale)

        self.opt = torch.optim.SGD(params, lr=self.lr, momentum=0.0)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return x
        m = self.scaler["mean"]
        s = self.scaler["std"]
        s = np.where(s < 1e-6, 1.0, s)
        return (x - m) / s

    def _gate(self, x_np: np.ndarray) -> float:
        # OOD gating
        if self.scaler is None:
            g_ood = 1.0
        else:
            z = (x_np - self.scaler["mean"]) / np.maximum(self.scaler["std"], 1e-6)
            d2 = float(np.mean(z * z))
            g_ood = math.exp(-d2 / self.ood_decay)

        # uncertainty gating
        g_unc = 1.0 / (1.0 + self.ema_err / max(1e-6, self.unc_scale))

        g = _clip(g_ood * g_unc, 0.0, 1.0)
        return float(g)

    # ---------------- public: predict ----------------
    def predict(self, feat_np: np.ndarray, cond_val: Optional[float] = None) -> np.ndarray:
        """
        输入:
          feat_np: (D,) float32/float64
          cond_val: 可选条件（例如速度标量），用于 cond2beta
        输出:
          fhat: (3,) float32, 单位 N
        """
        feat_np = np.asarray(feat_np, dtype=np.float32).reshape(-1)
        self._ensure_model(in_dim_runtime=int(feat_np.shape[0]))

        assert self.model is not None

        x = torch.tensor(self._norm(feat_np), dtype=torch.float32, device=self.device).unsqueeze(0)

        # 若模型支持 cond2beta，就更新一下当前 task 的 beta（不回传梯度）
        if (cond_val is not None) and hasattr(self.model, "_beta_from_c"):
            c = torch.tensor([[float(cond_val)]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ = self.model._beta_from_c(c, task_id=self.task_id)

        self.model.eval()
        with torch.no_grad():
            fhat = self.model._pred_physical(x, task_id=self.task_id).squeeze(0).detach().cpu().numpy()

        # 限幅（总体幅值）
        fhat = np.clip(fhat, -self.mag_limit, self.mag_limit)

        # 逐步变化限幅 + gating（避免抖动/失配）
        g = self._gate(feat_np)
        delta = np.clip(fhat - self.last_fhat, -self.step_limit, self.step_limit)
        fhat_used = self.last_fhat + g * delta

        self.last_fhat = fhat_used.astype(np.float32)
        return self.last_fhat

    # ---------------- public: online adapt ----------------
    def push(
        self,
        feat_np: np.ndarray,
        target_np: np.ndarray,
        cur_vel: Optional[Sequence[float]] = None,
        att: Optional[Sequence[float]] = None,
        thr: Optional[float] = None,
    ):
        """
        将 (feat, target_fa) 放入 buffer
        target_np: (3,) residual force in world/NED frame, unit N
        cur_vel/att/thr: 用于简单过滤不可信样本（和你原脚本一致）
        """
        # 过滤：低速（近悬停）、大姿态、油门饱和
        if cur_vel is not None and float(np.linalg.norm(np.asarray(cur_vel, dtype=np.float32))) < 0.3:
            return
        if att is not None:
            r = float(att[0]); p = float(att[1])
            if abs(r) > math.radians(30) or abs(p) > math.radians(30):
                return
        if thr is not None and (thr < 0.03 or thr > 0.97):
            return

        feat_np = np.asarray(feat_np, dtype=np.float32).reshape(-1)
        target_np = np.asarray(target_np, dtype=np.float32).reshape(3)

        # 维护无量纲 EMA 误差（按 g*m 归一）
        denom = max(1e-6, 9.81 * float(UAV_mass))
        err = float(np.linalg.norm(target_np - self.last_fhat) / denom)
        self.ema_err = 0.9 * self.ema_err + 0.1 * err

        self.buffer.append((feat_np, target_np))

    def online_update(self) -> Optional[float]:
        """
        从 buffer 采样训练，只更新当前 task 的 embedding/β
        返回 loss（float）或 None（数据不足）
        """
        if (self.model is None) or (self.opt is None):
            return None
        if len(self.buffer) < max(64, self.batch_size // 2):
            return None

        n = min(self.batch_size, len(self.buffer))
        idx = np.random.choice(len(self.buffer), size=n, replace=False)

        xs = np.stack([self.buffer[i][0] for i in idx], axis=0).astype(np.float32)
        ys = np.stack([self.buffer[i][1] for i in idx], axis=0).astype(np.float32)

        x = torch.tensor(self._norm(xs), dtype=torch.float32, device=self.device)
        y = torch.tensor(ys, dtype=torch.float32, device=self.device)

        self.model.train()
        self.opt.zero_grad()

        f_pred = self.model._pred_physical(x, task_id=self.task_id)
        loss = F.smooth_l1_loss(f_pred, y, beta=5.0)  # Huber
        loss.backward()

        # 只允许当前 task 的 embedding/β 更新
        with torch.no_grad():
            if self.model.task_embeddings.grad is not None:
                mask = torch.zeros_like(self.model.task_embeddings)
                mask[self.task_id] = 1.0
                self.model.task_embeddings.grad *= mask

            if hasattr(self.model, "task_beta_logscale") and (self.model.task_beta_logscale.grad is not None):
                maskb = torch.zeros_like(self.model.task_beta_logscale)
                maskb[self.task_id] = 1.0
                self.model.task_beta_logscale.grad *= maskb

        self.opt.step()
        return float(loss.detach().cpu().item())

    def maybe_update(self) -> Optional[float]:
        self.step += 1
        if self.update_every <= 0:
            return None
        if self.step % self.update_every == 0:
            return self.online_update()
        return None

    # ---------------- warm-start I/O ----------------
    def export_adapt_state(self) -> Dict[str, np.ndarray]:
        """
        导出可迁移参数：task embedding + optional beta
        """
        out: Dict[str, np.ndarray] = {}
        if self.model is None:
            return out

        with torch.no_grad():
            out["task_emb"] = self.model.task_embeddings[self.task_id].detach().cpu().numpy()
            if hasattr(self.model, "task_beta_logscale"):
                out["task_beta"] = self.model.task_beta_logscale[self.task_id].detach().cpu().numpy()
        return out

    def import_adapt_state(self, path: Union[str, Path]) -> bool:
        """
        从 npz 导入 task embedding/beta
        若模型未初始化，先缓存到 _warm_state，等第一次 predict 时注入
        """
        p = Path(path)
        if (p is None) or (not p.exists()):
            return False

        npz = np.load(str(p), allow_pickle=True)
        warm: Dict[str, np.ndarray] = {}
        if "task_emb" in npz.files:
            warm["task_emb"] = np.asarray(npz["task_emb"])
        if "task_beta" in npz.files:
            warm["task_beta"] = np.asarray(npz["task_beta"])

        self._warm_state = warm
        print(f"[Warm-start] Loaded task params from {p} (will inject on first model creation)")

        # 若模型已存在，立即注入
        if self.model is not None:
            with torch.no_grad():
                if "task_emb" in warm:
                    te = torch.tensor(warm["task_emb"], dtype=torch.float32, device=self.device)
                    self.model.task_embeddings[self.task_id].copy_(te)
                if ("task_beta" in warm) and hasattr(self.model, "task_beta_logscale"):
                    tb = torch.tensor(warm["task_beta"], dtype=torch.float32, device=self.device)
                    self.model.task_beta_logscale[self.task_id].copy_(tb)
            self._warm_state = None
        return True

    def save_adapt_state(self, path: Union[str, Path], extra: Optional[dict] = None):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = self.export_adapt_state()
        if extra:
            for k, v in extra.items():
                payload[k] = np.asarray(v)

        np.savez(str(p), **payload)
        print(f"[Warm-start] Saved task params to {p}")
