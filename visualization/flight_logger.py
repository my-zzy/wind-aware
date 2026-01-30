# visualization/flight_logger.py
from __future__ import annotations
import os
import numpy as np
from controllers.interfaces import Obs, Plan, Cmd

class FlightLogger:
    def __init__(self, out_dir="./yopo_logs"):
        self.out_dir = out_dir
        self.t = []
        self.pos = []
        self.vel = []
        self.v_ref = []
        self.cmd = []
        self.min_depth = []

    def push(self, obs: Obs, plan: Plan, cmd: Cmd):
        self.t.append(float(obs.t))
        self.pos.append(obs.pos_ned.copy())
        self.vel.append(obs.vel_ned.copy())
        self.v_ref.append(plan.v_ref_ned.copy())
        self.cmd.append(np.array([cmd.vx, cmd.vy, cmd.z, cmd.yaw_deg], dtype=np.float32))
        self.min_depth.append(float(plan.min_depth))

    def save(self, name="logs"):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{name}.npz")
        np.savez(
            path,
            t=np.asarray(self.t, dtype=np.float32),
            pos=np.asarray(self.pos, dtype=np.float32),
            vel=np.asarray(self.vel, dtype=np.float32),
            v_ref=np.asarray(self.v_ref, dtype=np.float32),
            cmd=np.asarray(self.cmd, dtype=np.float32),
            min_depth=np.asarray(self.min_depth, dtype=np.float32),
        )
        print(f"[logger] saved {path}")
