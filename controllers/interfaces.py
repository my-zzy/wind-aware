# controllers/interfaces.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class Obs:
    t: float
    pos_ned: np.ndarray          # (3,)
    vel_ned: np.ndarray          # (3,)
    acc_ned: np.ndarray          # (3,)
    R_wb_ned: np.ndarray         # (3,3) body->world in NED
    quat_wxyz: tuple[float,float,float,float]
    ang_vel_ned: np.ndarray      # (3,)
    depth_z_m: Optional[np.ndarray]   # (H,W) Z-depth (meters)
    rotor_speeds: Optional[np.ndarray]# (4,)
    extra: Dict[str, Any]

@dataclass
class Plan:
    v_ref_ned: np.ndarray        # (3,)
    z_cmd_ned: float             # desired Down
    yaw_ref_deg: float
    best_id: int
    best_score: float
    min_depth: float
    subgoal_ned: np.ndarray
    final_goal_ned: np.ndarray
    debug: Dict[str, Any]

@dataclass
class Cmd:
    vx: float
    vy: float
    z: float
    yaw_deg: float
    meta: Dict[str, Any]
