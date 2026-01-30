# planners/subgoal.py
from __future__ import annotations
import numpy as np

def compute_subgoal(pos_ned: np.ndarray, goal_ned: np.ndarray, radius: float) -> np.ndarray:
    vec = goal_ned - pos_ned
    d = float(np.linalg.norm(vec))
    if d < radius:
        return goal_ned.copy()
    return pos_ned + vec / max(1e-6, d) * radius
