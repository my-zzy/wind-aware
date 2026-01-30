# controllers/yaw_controller.py
from __future__ import annotations
import numpy as np

def wrap_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a

class YawController:
    def __init__(self, mode="face_velocity", alpha=0.25, rate_limit_deg_s=90.0):
        self.mode = mode
        self.alpha = float(alpha)
        self.rate_limit = float(rate_limit_deg_s)
        self._yaw = 0.0

    def reset(self, yaw_deg=0.0):
        self._yaw = float(yaw_deg)

    def step(self, vxy: np.ndarray, yaw_ref_deg: float, dt: float):
        if self.mode == "hold":
            target = self._yaw
        else:
            if float(np.linalg.norm(vxy)) > 1e-3:
                target = float(np.degrees(np.arctan2(vxy[1], vxy[0])))
            else:
                target = float(yaw_ref_deg)

        # smooth + rate limit
        err = wrap_deg(target - self._yaw)
        max_step = self.rate_limit * max(1e-6, dt)
        err = float(np.clip(err, -max_step, max_step))
        self._yaw = wrap_deg(self._yaw + self.alpha * err)
        return self._yaw
