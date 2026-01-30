# controllers/windaware_mixer.py
from __future__ import annotations
import numpy as np
from controllers.interfaces import Obs, Plan, Cmd
from .mp_imports import ensure_metapinn_on_path
ensure_metapinn_on_path(r"E:\wind-aware-main\Meta-PINN") 
from meta_pinn.config import DEFAULT_OPTIONS

UAV_mass = float(DEFAULT_OPTIONS["UAV_mass"])

def clamp_xy(vx, vy, max_xy):
    n = float(np.linalg.norm([vx, vy]))
    if n <= max_xy: return vx, vy
    s = max_xy / max(1e-6, n)
    return vx*s, vy*s

class WindAwareMixer:
    def __init__(self, max_vel_xy=4.0, k_ff=0.5):
        self.max_vel_xy = float(max_vel_xy)
        self.k_ff = float(k_ff)

    def mix(self, obs: Obs, plan: Plan, fhat_ned: np.ndarray | None, dt: float) -> Cmd:
        v = plan.v_ref_ned.astype(np.float32).copy()

        if fhat_ned is not None:
            a_res = fhat_ned.astype(np.float32) / float(UAV_mass)
            v[:2] = v[:2] + self.k_ff * a_res[:2] * float(dt)

        vx, vy = clamp_xy(float(v[0]), float(v[1]), self.max_vel_xy)

        return Cmd(
            vx=vx, vy=vy, z=float(plan.z_cmd_ned), yaw_deg=float(plan.yaw_ref_deg),
            meta={"fhat_ned": None if fhat_ned is None else fhat_ned.copy()}
        )
