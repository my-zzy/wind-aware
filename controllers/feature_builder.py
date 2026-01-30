# controllers/metapinn/feature_builder.py
from __future__ import annotations
import numpy as np
from typing import List
from .interfaces import Obs, Plan

def build_features(keys: List[str], obs: Obs, plan: Plan, cmd_vxy: np.ndarray) -> np.ndarray:
    parts = []
    qw,qx,qy,qz = obs.quat_wxyz
    for k in keys:
        if k == "p":
            parts.append(obs.pos_ned.astype(np.float32))
        elif k == "v":
            parts.append(obs.vel_ned.astype(np.float32))
        elif k == "q":
            parts.append(np.asarray([qw,qx,qy,qz], dtype=np.float32))
        elif k == "w":
            parts.append(obs.ang_vel_ned.astype(np.float32))
        elif k == "R":
            parts.append(obs.R_wb_ned.astype(np.float32).reshape(-1))
        elif k == "v_d":
            parts.append(plan.v_ref_ned.astype(np.float32))
        elif k == "cmd_v":
            parts.append(cmd_vxy.astype(np.float32))
        elif k == "pwm":
            if obs.rotor_speeds is None:
                parts.append(np.zeros(4, dtype=np.float32))
            else:
                parts.append(obs.rotor_speeds.astype(np.float32))
        else:
            parts.append(np.zeros(1, dtype=np.float32))
    return np.concatenate(parts, axis=0)
