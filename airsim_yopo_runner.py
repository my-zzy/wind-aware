# -*- coding: utf-8 -*-
"""
airsim_yopo_runner.py

Run YOPO (depth-image + state) policy online in Microsoft AirSim (SimpleFlight).

This script provides:
  1) Dual-view visualization: [Left] raw depth colormap, [Right] depth + top-K candidate trajectories.
  2) Step-by-step debug prints: measured speed, best idx, best score, min_depth (ROI), safety scale, dist-to-goal.
  3) Safer takeoff and optional safety "shield" (speed scaling + optional climb when obstacles are too close).
  4) Goal can be explicit NED [x,y,z] or a UE Actor name.
  5) On finish (goal reached or ESC): save control curves (png) + logs.npz; optionally show plots.
  6) When goal reached: hover -> land -> disarm.

Coordinate frames:
  - AirSim uses NED (x North, y East, z Down).
  - YOPO repo assumes a right-handed frame with z Up and y Left (NWU/FLU-like).
  - We convert NED(world)/FRD(body) -> NWU(world)/FLU(body) by flipping y & z.

Usage:
  python airsim_yopo_runner.py
Edit SETTINGS at bottom.
"""

from __future__ import annotations

import os
import sys
import time
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch

# --------------------------
# YOPO imports (from the provided zip)
# --------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_YOPO_DIR = os.path.join(THIS_DIR, "YOPO")
YOPO_DIR = os.environ.get("YOPO_DIR", DEFAULT_YOPO_DIR)

if os.path.isdir(YOPO_DIR) and YOPO_DIR not in sys.path:
    sys.path.insert(0, YOPO_DIR)

try:
    import airsim  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Cannot import airsim. Make sure you installed the AirSim Python package (pip install airsim)."
    ) from e

from policy.yopo_network import YopoNetwork
from policy.primitive import LatticePrimitive

# YOPO poly solver name differs across forks; support both.
try:
    from policy.poly_solver import Poly5Solver  # type: ignore
except Exception:
    from policy.poly_solver import Polys5Solver as Poly5Solver  # type: ignore


# --------------------------
# Utilities
# --------------------------
def quat_to_rot_wb_ned(q_wxyz: Tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix mapping body -> world (NED) from quaternion (w,x,y,z)."""
    w, x, y, z = q_wxyz
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )
    return R


# NED -> (N, W, U). Also FRD -> (F, L, U). Keeps right-handedness.
F_NED_TO_NWU = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def ned_to_nwu(v: np.ndarray) -> np.ndarray:
    return (F_NED_TO_NWU @ v.reshape(3, 1)).reshape(3)


def nwu_to_ned(v: np.ndarray) -> np.ndarray:
    return (F_NED_TO_NWU @ v.reshape(3, 1)).reshape(3)


def rot_wb_ned_to_nwu(R_wb_ned: np.ndarray) -> np.ndarray:
    """Change of basis: R_wb_nwu = F * R_wb_ned * F."""
    return F_NED_TO_NWU @ R_wb_ned @ F_NED_TO_NWU


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm:
        return v
    return v * (max_norm / (n + 1e-8))


@dataclass
class GoalSpec:
    kind: str  # "ned" or "ue_actor"
    ned: Optional[np.ndarray] = None
    actor_name: Optional[str] = None


class AirSimYopoRunner:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.airsim_ip = settings.get("airsim_ip", "127.0.0.1")
        self.vehicle_name = settings.get("vehicle_name", "Drone1")
        self._repel_lp = 0.0
        self._scale_lp = 1.0
        # --- shield states (for anti-oscillation + direction stickiness) ---
        self._vy_shield = 0.0           # filtered lateral bias
        self._avoid_dir = 0             # -1 go left, +1 go right, 0 none
        self._avoid_hold = 0            # keep direction for N steps

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Depth preprocessing
        self.img_h = int(settings.get("image_height", 96))
        self.img_w = int(settings.get("image_width", 160))
        self.max_depth_m = float(settings.get("depth_max_m", 20.0))

        # Control loop timing
        self.ctrl_dt = float(settings.get("control_dt", 0.10))
        self.cmd_duration = float(settings.get("command_duration_s", max(0.25, self.ctrl_dt * 2.5)))
        self.lookahead_t = float(settings.get("trajectory_lookahead_s", 0.20))

        # Speed limits
        self.max_vel_xy = float(settings.get("max_vel_xy", 3.0))
        self.max_vel_norm = float(settings.get("max_vel_norm", 6.0))

        # Debug prints
        self.verbose_step = bool(settings.get("verbose_step", True))

        # Safety shield (recommended)
        self.enable_safety_shield = bool(settings.get("enable_safety_shield", True))
        self.safety_slow_depth_m = float(settings.get("safety_slow_depth_m", 3.0))
        self.safety_stop_depth_m = float(settings.get("safety_stop_depth_m", 1.5))
        self.safety_climb_m = float(settings.get("safety_climb_m", 0.8))  # when stop, climb by this (U direction)

        # ROI for min-depth (fractions)
        self.min_depth_roi_x0 = float(settings.get("min_depth_roi_x0", 0.35))
        self.min_depth_roi_x1 = float(settings.get("min_depth_roi_x1", 0.65))
        self.min_depth_roi_y0 = float(settings.get("min_depth_roi_y0", 0.20))
        self.min_depth_roi_y1 = float(settings.get("min_depth_roi_y1", 0.80))

        # Visualization
        self.vis_depth = bool(settings.get("vis_depth", True))
        self.vis_trajs = bool(settings.get("vis_trajs", True))
        self.vis_scale = float(settings.get("vis_scale", 3.0))
        self.vis_show_text = bool(settings.get("vis_show_text", False))
        self.vis_topk_trajs = int(settings.get("vis_topk_trajs", 3))
        self.vis_draw_endpoints = bool(settings.get("vis_draw_endpoints", True))
        self.traj_vis_samples = int(settings.get("traj_vis_samples", 8))

        # --------- Live top-down (visualization only) ---------
        self.live_topdown_history = int(settings.get("live_topdown_history", 350))
        self.live_topdown_size = tuple(settings.get("live_topdown_size", [360, 320]))  # (w,h)
        self.live_topdown_margin = int(settings.get("live_topdown_margin", 18))

        # desired position integrator (plot only)
        self._des_pos_ned: Optional[np.ndarray] = None


        # Camera FOV for projection
        self.h_fov_deg = float(settings.get("h_fov_deg", 90.0))
        self.v_fov_deg = float(settings.get("v_fov_deg", 60.0))

        # Start/goal
        self.target_z_ned = float(settings.get("target_z_ned", -3.0))
        self.takeoff_timeout_s = float(settings.get("takeoff_timeout_s", 8.0))
        self.goal_reached_m = float(settings.get("goal_reached_m", 2.0))
        self.land_on_goal = bool(settings.get("land_on_goal", True))

        # Plot/log
        self.plot_on_exit = bool(settings.get("plot_on_exit", True))
        self.plot_show = bool(settings.get("plot_show", True))
        self.plot_dir = str(settings.get("plot_dir", "./yopo_logs"))

        # Runtime logs
        self._t0 = time.time()
        self._log = {
            "t": [],
            "pos_ned": [],
            "vel_ned": [],
            "vx_cmd": [],
            "vy_cmd": [],
            "z_cmd": [],
            "best_id": [],
            "best_score": [],
            "min_depth": [],
            "shield_scale": [],
            "dist_goal": [],
            # ---- for "Flight Data" plotting (visualization only) ----
            "pos_des_ned": [],
            "yaw_act_deg": [],
            "pitch_act_deg": [],
            "roll_act_deg": [],
            "yaw_des_deg": [],
            "pitch_des_deg": [],
            "roll_des_deg": [],
        }
        # Progress monitoring (to prevent 'fly away' when network is out-of-distribution)
        self._prev_dist: Optional[float] = None
        self._bad_progress_count: int = 0

        # YOPO config / primitives
        from config.config import cfg  # type: ignore

        cfg["train"] = False
        if "yopo_test_velocity" in settings:
            cfg["velocity"] = float(settings["yopo_test_velocity"])

        self.lattice = LatticePrimitive.get_instance()
        self.segment_time = float(self.lattice.segment_time)
        # Training-time goal length (YOPO is a local planner; keep goal vector in-distribution)
        self.yopo_goal_length = float(cfg["goal_length"])  # typically 2*radio_range (e.g., 10m)

        # Load model
        ckpt = settings.get("checkpoint_path")
        if ckpt is None:
            ckpt = os.path.join(YOPO_DIR, "saved", "YOPO_1", "epoch50.pth")
        self.checkpoint_path = ckpt

        self.model = YopoNetwork().to(self.device)
        self._load_checkpoint(self.checkpoint_path)
        self.model.eval()

        # AirSim client
        self.client = airsim.MultirotorClient(ip=self.airsim_ip)
        self.client.confirmConnection()
        print("Connected!")
        

        # Goal spec
        self.goal = self._parse_goal(settings.get("goal"))

        # ---- Subgoal Rolling Planner ----
        self.subgoal_radius = float(settings.get("subgoal_radius", 8.0))  # meters
        self._current_subgoal_ned: Optional[np.ndarray] = None

    def _compute_subgoal(self, pos_ned: np.ndarray, goal_ned: np.ndarray) -> np.ndarray:
        """
        Generate a rolling subgoal at fixed radius toward the final goal.
        """
        vec = goal_ned - pos_ned
        d = float(np.linalg.norm(vec))
        if d < self.subgoal_radius:
            return goal_ned.copy()
        return pos_ned + vec / d * self.subgoal_radius

    # ---------- load / goal ----------
    def _load_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"YOPO checkpoint not found: {path}\n"
                f"Tip: set SETTINGS['checkpoint_path'] or environment YOPO_DIR."
            )
        # PyTorch 2.1+ warns about weights_only; support both versions.
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)

    def _parse_goal(self, goal_cfg: Any) -> GoalSpec:
        if goal_cfg is None:
            return GoalSpec(kind="ned", ned=np.array([20.0, 0.0, self.target_z_ned], dtype=np.float32))
        if isinstance(goal_cfg, (list, tuple)) and len(goal_cfg) == 3:
            return GoalSpec(kind="ned", ned=np.array(goal_cfg, dtype=np.float32))
        if isinstance(goal_cfg, dict):
            t = str(goal_cfg.get("type", "ned")).lower()
            if t == "ned":
                xyz = goal_cfg.get("xyz", goal_cfg.get("goal", None))
                if xyz is None:
                    raise ValueError("goal type=ned requires 'xyz': [x,y,z]")
                return GoalSpec(kind="ned", ned=np.array(xyz, dtype=np.float32))
            if t in ("ue_actor", "ue", "actor"):
                name = goal_cfg.get("name")
                if not name:
                    raise ValueError("goal type=ue_actor requires 'name'")
                return GoalSpec(kind="ue_actor", actor_name=str(name))
        raise ValueError("Unsupported goal format. Use [x,y,z] or {type:'ue_actor',name:'...'}")

    def _resolve_goal_ned(self) -> np.ndarray:
        if self.goal.kind == "ned":
            assert self.goal.ned is not None
            return self.goal.ned.astype(np.float32)
        assert self.goal.actor_name is not None
        pose = self.client.simGetObjectPose(self.goal.actor_name)
        if math.isnan(pose.position.x_val):
            raise RuntimeError(
                f"Cannot find UE actor '{self.goal.actor_name}'. Make sure the Actor name matches exactly."
            )
        return np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=np.float32)

    # ---------- takeoff / land ----------
    def arm_and_takeoff(self) -> None:
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        self.client.takeoffAsync(timeout_sec=self.takeoff_timeout_s, vehicle_name=self.vehicle_name).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.3)

        st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        x0 = st.kinematics_estimated.position.x_val
        y0 = st.kinematics_estimated.position.y_val
        self.client.moveToPositionAsync(
            x0, y0, self.target_z_ned,
            velocity=max(1.0, min(3.0, self.max_vel_xy)),
            vehicle_name=self.vehicle_name,
        ).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)

    def land_and_disarm(self) -> None:
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        try:
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            # Some SimpleFlight builds don't implement landAsync well; fallback to descend
            st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            z = st.kinematics_estimated.position.z_val
            self.client.moveByVelocityZAsync(0, 0, z + 1.5, duration=2.0, vehicle_name=self.vehicle_name).join()

        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        except Exception:
            pass
        try:
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except Exception:
            pass

    def _get_depth_image(self) -> Optional[np.ndarray]:
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)],
            vehicle_name=self.vehicle_name,
        )
        if not responses:
            return None

        d = responses[0]
        depth_euclid = np.array(d.image_data_float, dtype=np.float32).reshape(d.height, d.width)

        # resize
        depth_euclid = cv2.resize(depth_euclid, (self.img_w, self.img_h))

        # === 欧式距离 -> Z 深度 ===
        h, w = depth_euclid.shape
        hfov = np.deg2rad(self.h_fov_deg)
        vfov = np.deg2rad(self.v_fov_deg)

        xs = (np.linspace(-1, 1, w)[None, :]) * np.tan(hfov/2)
        ys = (np.linspace(-1, 1, h)[:, None]) * np.tan(vfov/2)

        cos_theta = 1.0 / np.sqrt(xs**2 + ys**2 + 1.0)

        depth_z = depth_euclid * cos_theta

        # === YOPO 需要“真实相机”的深度 ===
        noise = np.random.normal(0, 0.03, depth_z.shape)
        depth_z = depth_z + noise
        depth_z = cv2.GaussianBlur(depth_z, (3,3), 0)

        depth_z = np.clip(depth_z, 0.0, self.max_depth_m)
        return depth_z

    def _preprocess_depth(self, depth_m: np.ndarray) -> torch.Tensor:
        depth_norm = (np.clip(depth_m, 0.0, self.max_depth_m) / self.max_depth_m).astype(np.float32)
        return torch.from_numpy(depth_norm)[None, None, :, :].to(self.device)

    def _get_state_ned(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = st.kinematics_estimated
        pos = np.array([kin.position.x_val, kin.position.y_val, kin.position.z_val], dtype=np.float32)
        vel = np.array([kin.linear_velocity.x_val, kin.linear_velocity.y_val, kin.linear_velocity.z_val], dtype=np.float32)
        acc = np.array([kin.linear_acceleration.x_val, kin.linear_acceleration.y_val, kin.linear_acceleration.z_val], dtype=np.float32)

        q = kin.orientation
        R_wb = quat_to_rot_wb_ned((q.w_val, q.x_val, q.y_val, q.z_val))
        return pos, vel, acc, R_wb

    def _compute_min_depth(self, depth_m: np.ndarray) -> float:
        h, w = depth_m.shape[:2]
        x0 = int(w * self.min_depth_roi_x0)
        x1 = int(w * self.min_depth_roi_x1)
        y0 = int(h * self.min_depth_roi_y0)
        y1 = int(h * self.min_depth_roi_y1)
        roi = depth_m[y0:y1, x0:x1].astype(np.float32)

        if roi.size == 0:
            roi = depth_m.astype(np.float32)

        # 过滤异常：0/极小值、NaN、Inf
        valid = np.isfinite(roi) & (roi > 0.2)   # 0.2m 以下基本都是噪声/贴脸像素
        vals = roi[valid]

        if vals.size < 50:
            # 用更宽松的过滤
            valid = np.isfinite(roi) & (roi > 1e-3)
            vals = roi[valid]

        if vals.size == 0:
            return float(self.max_depth_m)

        # 用分位数代替 min（更稳）
        return float(np.quantile(vals, 0.05))

    # ---------- projection / traj ----------
    def _project_body_points_to_image(self, pts_b: np.ndarray) -> np.ndarray:
        x = pts_b[:, 0]
        y = pts_b[:, 1]
        z = pts_b[:, 2]
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(z, np.sqrt(x * x + y * y) + 1e-8)
        h_fov = np.deg2rad(self.h_fov_deg)
        v_fov = np.deg2rad(self.v_fov_deg)
        u = (yaw / (h_fov / 2.0) + 1.0) * 0.5 * (self.img_w - 1)
        v = (1.0 - (pitch / (v_fov / 2.0) + 1.0) * 0.5) * (self.img_h - 1)
        return np.stack([u, v], axis=1)

    def _sample_poly_trajectory(self, start_v: np.ndarray, start_a: np.ndarray, end_state: np.ndarray) -> np.ndarray:
        end_p = end_state[0:3]
        end_v = end_state[3:6]
        end_a = end_state[6:9]
        ts = np.linspace(0.0, self.segment_time, self.traj_vis_samples, dtype=np.float32)

        sol_x = Poly5Solver(0.0, float(start_v[0]), float(start_a[0]),
                            float(end_p[0]), float(end_v[0]), float(end_a[0]),
                            self.segment_time)
        sol_y = Poly5Solver(0.0, float(start_v[1]), float(start_a[1]),
                            float(end_p[1]), float(end_v[1]), float(end_a[1]),
                            self.segment_time)
        sol_z = Poly5Solver(0.0, float(start_v[2]), float(start_a[2]),
                            float(end_p[2]), float(end_v[2]), float(end_a[2]),
                            self.segment_time)
        pts = np.stack(
            [[sol_x.get_position(t), sol_y.get_position(t), sol_z.get_position(t)] for t in ts],
            axis=0
        ).astype(np.float32)
        return pts

    # ---------- visualization ----------
    def _render_topdown_panel(
        self,
        pos_hist_ned: np.ndarray,
        final_goal_ned: np.ndarray,
        subgoal_ned: np.ndarray,
    ) -> np.ndarray:
        """
        Top-down panel (visualization only):
        - x (North) -> right
        - y (East)  -> down
        """
        w, h = self.live_topdown_size
        margin = self.live_topdown_margin
        panel = np.zeros((h, w, 3), dtype=np.uint8)

        if pos_hist_ned is None or pos_hist_ned.size == 0:
            return panel

        pts = pos_hist_ned[:, :2].astype(np.float32)
        extra = np.stack([final_goal_ned[:2], subgoal_ned[:2]], axis=0).astype(np.float32)
        all_pts = np.concatenate([pts, extra], axis=0)

        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        dx = max(1e-3, float(xmax - xmin))
        dy = max(1e-3, float(ymax - ymin))

        pad = 0.20
        xmin -= dx * pad; xmax += dx * pad
        ymin -= dy * pad; ymax += dy * pad
        dx = max(1e-3, float(xmax - xmin))
        dy = max(1e-3, float(ymax - ymin))

        def world_to_px(pxy: np.ndarray):
            u = int(margin + (pxy[0] - xmin) / dx * (w - 2 * margin))
            v = int(margin + (pxy[1] - ymin) / dy * (h - 2 * margin))
            return u, v

        # grid
        for gx in range(0, w, 40):
            cv2.line(panel, (gx, 0), (gx, h - 1), (25, 25, 25), 1)
        for gy in range(0, h, 40):
            cv2.line(panel, (0, gy), (w - 1, gy), (25, 25, 25), 1)

        poly = np.array([world_to_px(p) for p in pts], dtype=np.int32)
        if len(poly) >= 2:
            cv2.polylines(panel, [poly], False, (200, 200, 200), 2, lineType=cv2.LINE_AA)

        u0, v0 = world_to_px(pts[0])
        u1, v1 = world_to_px(pts[-1])
        ug, vg = world_to_px(final_goal_ned[:2])
        us, vs = world_to_px(subgoal_ned[:2])

        cv2.circle(panel, (u0, v0), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)   # start
        cv2.circle(panel, (u1, v1), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)       # current
        cv2.circle(panel, (ug, vg), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)       # final goal
        cv2.circle(panel, (us, vs), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)     # subgoal

        cv2.putText(panel, "Top-Down (NED x/y)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return panel

    def _depth_to_colormap(self, depth_m: np.ndarray) -> np.ndarray:
        d = np.clip(depth_m, 0.0, self.max_depth_m)
        u8 = (d / self.max_depth_m * 255.0).astype(np.uint8)
        return cv2.applyColorMap(255 - u8, cv2.COLORMAP_TURBO)

    def _draw_dual_view(
        self,
        depth_m: np.ndarray,
        endstates_b: np.ndarray,
        scores: np.ndarray,
        best_id: int,
        start_v_b: np.ndarray,
        start_a_b: np.ndarray,
        min_depth: float,
        pos_ned: np.ndarray,
        final_goal_ned: np.ndarray,
        subgoal_ned: np.ndarray,
    ) -> None:
        if not (self.vis_depth or self.vis_trajs):
            return

        # Left: depth colormap + traj overlay
        left = self._depth_to_colormap(depth_m)
        if self.vis_trajs and endstates_b is not None and scores is not None:
            order = np.argsort(scores)
            k = int(np.clip(self.vis_topk_trajs, 1, len(order)))
            draw_ids = [int(i) for i in order[:k]]
            if best_id not in draw_ids:
                draw_ids = [best_id] + draw_ids[:-1]

            if self.vis_draw_endpoints:
                pts_uv = self._project_body_points_to_image(endstates_b[:, 0:3])
                for i in draw_ids:
                    u, v = pts_uv[i]
                    if 0 <= u < self.img_w and 0 <= v < self.img_h:
                        col = (0, 255, 0) if i == best_id else (255, 255, 255)
                        r = 4 if i == best_id else 2
                        cv2.circle(left, (int(u), int(v)), r, col, -1, lineType=cv2.LINE_AA)

            for i in draw_ids:
                pts_b = self._sample_poly_trajectory(start_v_b, start_a_b, endstates_b[i])
                poly = np.round(self._project_body_points_to_image(pts_b)).astype(np.int32)
                keep = (
                    (poly[:, 0] >= 0) & (poly[:, 0] < self.img_w) &
                    (poly[:, 1] >= 0) & (poly[:, 1] < self.img_h)
                )
                poly = poly[keep]
                if len(poly) >= 2:
                    col = (0, 255, 0) if i == best_id else (0, 200, 255)
                    cv2.polylines(left, [poly], False, col, 1, lineType=cv2.LINE_AA)

        if self.vis_show_text:
            cv2.putText(
                left,
                f"best={best_id:02d} score={scores[best_id]:.3f} minD={min_depth:.2f}m",
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Right: top-down history
        if len(self._log["pos_ned"]) > 0:
            hist = np.array(self._log["pos_ned"], dtype=np.float32)
            if hist.shape[0] > self.live_topdown_history:
                hist = hist[-self.live_topdown_history:]
        else:
            hist = np.array([pos_ned], dtype=np.float32)

        right = self._render_topdown_panel(hist, final_goal_ned, subgoal_ned)

        # scale left
        if self.vis_scale != 1.0:
            left_s = cv2.resize(
                left,
                (int(left.shape[1] * self.vis_scale), int(left.shape[0] * self.vis_scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            left_s = left

        # resize right to match height
        target_h = left_s.shape[0]
        scale_r = target_h / max(1, right.shape[0])
        right_s = cv2.resize(right, (int(right.shape[1] * scale_r), target_h), interpolation=cv2.INTER_NEAREST)

        vis = np.hstack([left_s, right_s])
        cv2.imshow("YOPO Live (Depth+Traj | Top-Down)", vis)


    # ---------- core step ----------
    @torch.inference_mode()
    def step(self) -> bool:
        depth_m = self._get_depth_image()
        if depth_m is None:
            print("[WARN] No depth image.")
            time.sleep(self.ctrl_dt)
            return True
        
        pos_ned, vel_ned, acc_ned, R_wb_ned = self._get_state_ned()
        final_goal_ned = self._resolve_goal_ned()

        # ---- Subgoal Rolling ----
        subgoal_ned = self._compute_subgoal(pos_ned, final_goal_ned)
        goal_ned = subgoal_ned


        # NED -> NWU
        pos_nwu = ned_to_nwu(pos_ned)
        vel_w_nwu = ned_to_nwu(vel_ned)
        acc_w_nwu = ned_to_nwu(acc_ned)
        goal_w_nwu = ned_to_nwu(goal_ned)

        # Rotation body->world in NWU
        R_wb_nwu = rot_wb_ned_to_nwu(R_wb_ned)
        R_bw_nwu = R_wb_nwu.T

        # body obs in NWU (FLU)
        vel_b = (R_bw_nwu @ vel_w_nwu.reshape(3, 1)).reshape(3)
        acc_b = (R_bw_nwu @ acc_w_nwu.reshape(3, 1)).reshape(3)
        goal_vec_w = goal_w_nwu - pos_nwu
        dist_to_goal = float(np.linalg.norm(goal_vec_w))

        # === 先算 dist，再做 clip ===
        clip = min(1.0, self.yopo_goal_length / max(1e-6, dist_to_goal))
        goal_vec_w_clip = goal_vec_w * clip

        goal_b_full = (R_bw_nwu @ goal_vec_w_clip.reshape(3, 1)).reshape(3)
        # ===== YOPO 正确输入=====
        goal_b = goal_b_full.copy()
        goal_b[2] = 0.0  # YOPO 只接受平面 goal
        acc_b[:] = 0.0

        obs = np.concatenate([vel_b, acc_b, goal_b], axis=0).astype(np.float32)
        obs_t = torch.from_numpy(obs)[None, :].to(self.device)
        depth_t = self._preprocess_depth(depth_m)

        endstate_t, score_t = self.model.inference(depth_t, obs_t)
        endstate = endstate_t[0].permute(1, 2, 0).reshape(-1, 9).detach().cpu().numpy()
        score = score_t[0].reshape(-1).detach().cpu().numpy()

        best_id = int(np.argmin(score))
        best_score = float(score[best_id])

        min_depth = self._compute_min_depth(depth_m)

        # visualize (dual view)
        self._draw_dual_view(
            depth_m, endstate, score, best_id, vel_b, acc_b, min_depth,
            pos_ned=pos_ned, final_goal_ned=final_goal_ned, subgoal_ned=subgoal_ned
        )


        # goal check
        dist_to_goal = float(np.linalg.norm(goal_vec_w))
        if dist_to_goal < self.goal_reached_m:
            print(f"[INFO] Goal reached (dist={dist_to_goal:.2f}m). Hover -> land.")
            
            try:
                self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            except Exception:
                pass
            if self.land_on_goal:
                self.land_and_disarm()
            return False

        # Progress monitor: if we keep moving away from the goal, blend in a simple go-to-goal controller.
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal
        else:
            if dist_to_goal > self._prev_dist + 0.5:  # meters
                self._bad_progress_count += 1
            else:
                self._bad_progress_count = 0
            self._prev_dist = dist_to_goal
        # build desired velocity from best traj at lookahead
        end = endstate[best_id]
        end_p, end_v, end_a = end[0:3], end[3:6], end[6:9]

        sol_x = Poly5Solver(0.0, float(vel_b[0]), float(acc_b[0]),
                            float(end_p[0]), float(end_v[0]), float(end_a[0]),
                            self.segment_time)
        sol_y = Poly5Solver(0.0, float(vel_b[1]), float(acc_b[1]),
                            float(end_p[1]), float(end_v[1]), float(end_a[1]),
                            self.segment_time)
        sol_z = Poly5Solver(0.0, float(vel_b[2]), float(acc_b[2]),
                            float(end_p[2]), float(end_v[2]), float(end_a[2]),
                            self.segment_time)

        t_la = float(np.clip(self.lookahead_t, 0.0, self.segment_time))
        v_des_b = np.array([sol_x.get_velocity(t_la), sol_y.get_velocity(t_la), sol_z.get_velocity(t_la)], dtype=np.float32)
        v_des_b = clamp_norm(v_des_b, self.max_vel_norm)

        # body->world(NWU)->NED
        v_des_w_nwu = (R_wb_nwu @ v_des_b.reshape(3, 1)).reshape(3)
        v_des_w_ned = nwu_to_ned(v_des_w_nwu)

        # clamp XY
        vxy = clamp_norm(np.array([v_des_w_ned[0], v_des_w_ned[1]], dtype=np.float32), self.max_vel_xy)
        vx_cmd, vy_cmd = float(vxy[0]), float(vxy[1])
        # === YOPO 全局方向纠偏 ===
        goal_vec_ned = goal_ned - pos_ned
        gxy = goal_vec_ned[:2]
        gxy_norm = np.linalg.norm(gxy)

        if gxy_norm > 1e-3:
            gxy_dir = gxy / gxy_norm

            v_cmd = np.array([vx_cmd, vy_cmd])
            v_norm = np.linalg.norm(v_cmd)

            if v_norm > 1e-3:
                v_dir = v_cmd / v_norm

                # 如果当前速度方向和目标方向夹角 > 60°，强制拉回
                cosang = np.dot(v_dir, gxy_dir)

                if cosang < 0.5:
                    blend = 0.6  # 拉回强度，0.5~0.7 都行
                    new_dir = (1 - blend) * v_dir + blend * gxy_dir
                    new_dir /= np.linalg.norm(new_dir)

                    vx_cmd = float(new_dir[0] * v_norm)
                    vy_cmd = float(new_dir[1] * v_norm)


        # If we are consistently diverging, override with a conservative go-to-goal velocity.
        if self._bad_progress_count >= int(self.settings.get('fallback_bad_steps', 6)):
            # world-frame goal direction in NED
            goal_vec_ned = goal_ned - pos_ned
            gxy = np.array([goal_vec_ned[0], goal_vec_ned[1]], dtype=np.float32)
            gxy_n = float(np.linalg.norm(gxy))
            if gxy_n > 1e-3:
                gxy = gxy / gxy_n
                v_fallback = float(self.settings.get('fallback_speed', min(1.5, self.max_vel_xy)))
                vx_cmd, vy_cmd = float(gxy[0] * v_fallback), float(gxy[1] * v_fallback)

        # ===================== safety shield (dense obstacles friendly) =====================
        scale = 1.0
        z_cmd = self.target_z_ned

        if self.enable_safety_shield and depth_m is not None:
            d = depth_m
            h, w = d.shape

            # --- region mins (use robust quantile instead of raw min to reduce pixel noise) ---
            def qmin(arr, q=0.03):
                v = arr[np.isfinite(arr)]
                if v.size == 0:
                    return self.max_depth_m
                return float(np.quantile(v, q))

            L = d[:, :int(0.33*w)]
            C = d[:, int(0.40*w):int(0.60*w)]
            R = d[:, int(0.67*w):]

            left_min   = qmin(L, 0.03)
            center_min = qmin(C, 0.03)
            right_min  = qmin(R, 0.03)
            min_depth  = qmin(d, 0.01)

            # --- 0) EMERGENCY: too close in front -> NO FILTER, strong slow + climb + strong lateral push ---
            # 这一步保证“不会因为滤波而撞上柱子”
            if center_min <= self.safety_stop_depth_m:
                # hard slow
                scale = 0.15
                vx_cmd *= scale
                vy_cmd *= scale

                # climb (up)
                z_cmd = self.target_z_ned - abs(self.safety_climb_m)

                # choose escape direction (go to the more open side)
                # direction: +1 means go right in NED? 这里我们只给 vy_cmd 加一个“增量”，方向由现有坐标保持一致即可
                # 规则：哪边更空就往哪边躲
                escape_dir = +1 if right_min > left_min else -1

                # strong instant lateral kick (NO FILTER)
                vy_kick = float(self.settings.get("shield_emergency_vy", 1.2)) * escape_dir
                vy_cmd += vy_kick

                # reset filter + lock direction for a short time to avoid flip-flop
                self._vy_shield = 0.0
                self._avoid_dir = escape_dir
                self._avoid_hold = int(self.settings.get("shield_hold_steps", 10))

            else:
                # --- 1) Direction stickiness (avoid left-right oscillation in dense pillars) ---
                # 如果已经锁定方向，则在 hold 期间不允许换边
                if self._avoid_hold > 0:
                    self._avoid_hold -= 1
                    avoid_dir = self._avoid_dir
                else:
                    # decide direction only when one side is clearly better
                    gap = right_min - left_min
                    thr = float(self.settings.get("shield_dir_gap_m", 0.6))  # 需要“明显更空”才换边
                    if gap > thr:
                        avoid_dir = +1
                    elif gap < -thr:
                        avoid_dir = -1
                    else:
                        avoid_dir = 0
                    # lock if decided
                    if avoid_dir != 0:
                        self._avoid_dir = avoid_dir
                        self._avoid_hold = int(self.settings.get("shield_hold_steps", 10))

                # --- 2) Danger-based speed scaling (smooth) ---
                slow_d = float(self.settings.get("shield_slow_d", self.safety_slow_depth_m))
                stop_d = float(self.settings.get("shield_stop_d", self.safety_stop_depth_m))

                if min_depth < slow_d:
                    # map depth -> scale in [min_scale, 1]
                    min_scale = float(self.settings.get("shield_min_scale", 0.30))
                    s = (min_depth - stop_d) / max(1e-6, (slow_d - stop_d))
                    s = float(np.clip(s, min_scale, 1.0))
                    scale = s
                else:
                    scale = 1.0

                vx_cmd *= scale
                vy_cmd *= scale

                # --- 3) Lateral repulsion (FILTER ONLY in non-emergency) ---
                # repulsion magnitude grows when closer
                d_ref = float(self.settings.get("shield_repulse_ref_m", 3.0))
                k_rep = float(self.settings.get("shield_k_rep", 1.0))     # overall gain
                vy_max = float(self.settings.get("shield_vy_cap", 1.0))   # cap

                # compute “how blocked” each side is (0..1)
                blockL = np.clip((d_ref - left_min) / max(1e-6, d_ref), 0.0, 1.0)
                blockR = np.clip((d_ref - right_min) / max(1e-6, d_ref), 0.0, 1.0)

                # raw bias: push away from the more blocked side
                vy_bias_raw = k_rep * (blockL - blockR)  # >0 => push right, <0 => push left
                vy_bias_raw = float(np.clip(vy_bias_raw, -vy_max, vy_max))

                # if we have a locked direction, enforce sign (prevents flip-flop)
                if avoid_dir != 0:
                    vy_bias_raw = float(abs(vy_bias_raw)) * avoid_dir

                # low-pass filter (only here)
                alpha = float(self.settings.get("shield_alpha", 0.80))  # smaller => more responsive
                self._vy_shield = alpha * self._vy_shield + (1.0 - alpha) * vy_bias_raw

                # apply filtered bias
                vy_cmd += self._vy_shield

                # --- 4) Corridor mode (both sides close, front still ok): suppress lateral to avoid “snake” ---
                corridor = (left_min < 2.5 and right_min < 2.5 and center_min > 4.0)
                if corridor:
                    vy_cmd *= float(self.settings.get("shield_corridor_vy_mul", 0.35))
                    vx_cmd *= float(self.settings.get("shield_corridor_vx_mul", 0.55))

        # final clamp (important)
        vx_cmd = float(np.clip(vx_cmd, -self.max_vel_xy, self.max_vel_xy))
        vy_cmd = float(np.clip(vy_cmd, -self.max_vel_xy, self.max_vel_xy))
        # ===================================================================

        # step print
        if self.verbose_step:
            vxy_meas = float(np.linalg.norm(vel_ned[0:2]))
            v_meas = float(np.linalg.norm(vel_ned))
            # print(
            #     f"[STEP] v_xy={vxy_meas:.2f} v={v_meas:.2f}  best={best_id:02d}  "
            #     f"score={best_score:.3f}  minD={min_depth:.2f}  scale={scale:.2f}  dist={dist_to_goal:.2f}"
            # )
            print(
                f"[STEP] v_xy={vxy_meas:.2f} v={v_meas:.2f}  best={best_id:02d}  "
                f"score={best_score:.3f}  minD={min_depth:.2f}  scale={scale:.2f}  "
                f"dist_sub={np.linalg.norm(subgoal_ned-pos_ned):.2f}  "
                f"dist_final={np.linalg.norm(final_goal_ned-pos_ned):.2f}"
            )


        # log
        t_now = time.time() - self._t0
        self._log["t"].append(float(t_now))
        self._log["pos_ned"].append(pos_ned.astype(np.float32))
        self._log["vel_ned"].append(vel_ned.astype(np.float32))
        self._log["vx_cmd"].append(float(vx_cmd))
        self._log["vy_cmd"].append(float(vy_cmd))
        self._log["z_cmd"].append(float(z_cmd))
        self._log["best_id"].append(int(best_id))
        self._log["best_score"].append(float(best_score))
        self._log["min_depth"].append(float(min_depth))
        self._log["shield_scale"].append(float(scale))
        self._log["dist_goal"].append(float(dist_to_goal))

        # yaw: face velocity direction (optional)
        st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pitch_rad, roll_rad, yaw_ned = airsim.to_eularian_angles(st.kinematics_estimated.orientation)

        if (vx_cmd * vx_cmd + vy_cmd * vy_cmd) > 1e-4:
            yaw_cmd_rad = math.atan2(vy_cmd, vx_cmd)
            yaw_deg = float(np.degrees(yaw_cmd_rad))
        else:
            yaw_deg = float(np.degrees(yaw_ned))
        # ---- plot-only desired state logging (NO control change) ----
        if self._des_pos_ned is None:
            self._des_pos_ned = pos_ned.copy()
        else:
            self._des_pos_ned[0] += float(vx_cmd) * float(self.ctrl_dt)
            self._des_pos_ned[1] += float(vy_cmd) * float(self.ctrl_dt)
            self._des_pos_ned[2] = float(z_cmd)

        self._log["pos_des_ned"].append(self._des_pos_ned.astype(np.float32))

        self._log["yaw_act_deg"].append(float(np.degrees(yaw_ned)))
        self._log["pitch_act_deg"].append(float(np.degrees(pitch_rad)))
        self._log["roll_act_deg"].append(float(np.degrees(roll_rad)))

        self._log["yaw_des_deg"].append(float(yaw_deg))
        self._log["pitch_des_deg"].append(0.0)
        self._log["roll_des_deg"].append(0.0)

        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        # send command
        self.client.moveByVelocityZAsync(
            vx_cmd, vy_cmd, z_cmd,
            duration=self.cmd_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name,
        )

        # ESC to stop
        if cv2.waitKey(1) & 0xFF == 27:
            return False
        return True

    # ---------- plotting ----------
    def _plot_logs(self) -> None:
        if not self._log["t"]:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARN] matplotlib not available, skip plotting: {e}")
            return

        os.makedirs(self.plot_dir, exist_ok=True)

        t = np.array(self._log["t"], dtype=np.float32)

        pos_act = np.array(self._log["pos_ned"], dtype=np.float32)
        vel = np.array(self._log["vel_ned"], dtype=np.float32)

        vx = np.array(self._log["vx_cmd"], dtype=np.float32)
        vy = np.array(self._log["vy_cmd"], dtype=np.float32)
        zc = np.array(self._log["z_cmd"], dtype=np.float32)

        score = np.array(self._log["best_score"], dtype=np.float32)
        min_d = np.array(self._log["min_depth"], dtype=np.float32)
        scale = np.array(self._log["shield_scale"], dtype=np.float32)
        dist = np.array(self._log["dist_goal"], dtype=np.float32)

        # desired (plot-only)
        pos_des = np.array(self._log.get("pos_des_ned", []), dtype=np.float32)
        yaw_act = np.array(self._log.get("yaw_act_deg", []), dtype=np.float32)
        yaw_des = np.array(self._log.get("yaw_des_deg", []), dtype=np.float32)
        pitch_act = np.array(self._log.get("pitch_act_deg", []), dtype=np.float32)
        pitch_des = np.array(self._log.get("pitch_des_deg", []), dtype=np.float32)
        roll_act = np.array(self._log.get("roll_act_deg", []), dtype=np.float32)
        roll_des = np.array(self._log.get("roll_des_deg", []), dtype=np.float32)

        # --- Flight Data (6 subplots) ---
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("Flight Data")

        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(t, pos_act[:, 0], label="Actual")
        if pos_des.size:
            ax1.plot(t, pos_des[:, 0], "r--", label="Desired")
        ax1.set_title("X Position"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("X (m)")
        ax1.grid(True); ax1.legend()

        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(t, pos_act[:, 1], label="Actual")
        if pos_des.size:
            ax2.plot(t, pos_des[:, 1], "r--", label="Desired")
        ax2.set_title("Y Position"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Y (m)")
        ax2.grid(True); ax2.legend()

        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(t, pos_act[:, 2], label="Actual")
        if pos_des.size:
            ax3.plot(t, pos_des[:, 2], "r--", label="Desired")
        ax3.set_title("Z Position"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Z (m)")
        ax3.grid(True); ax3.legend()

        ax4 = plt.subplot(3, 2, 4)
        if yaw_act.size:
            ax4.plot(t, yaw_act, label="Actual")
        if yaw_des.size:
            ax4.plot(t, yaw_des, "r--", label="Desired")
        ax4.set_title("Yaw Angle"); ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Yaw (deg)")
        ax4.grid(True); ax4.legend()

        ax5 = plt.subplot(3, 2, 5)
        if pitch_act.size:
            ax5.plot(t, pitch_act, label="Actual")
        if pitch_des.size:
            ax5.plot(t, pitch_des, "r--", label="Desired")
        ax5.set_title("Pitch Angle"); ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Pitch (deg)")
        ax5.grid(True); ax5.legend()

        ax6 = plt.subplot(3, 2, 6)
        if roll_act.size:
            ax6.plot(t, roll_act, label="Actual")
        if roll_des.size:
            ax6.plot(t, roll_des, "r--", label="Desired")
        ax6.set_title("Roll Angle"); ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Roll (deg)")
        ax6.grid(True); ax6.legend()

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.savefig("flight_data.png", dpi=180)

        # Save raw (keep backward compatible keys + add extra)
        np.savez(
            os.path.join(self.plot_dir, "logs.npz"),
            t=t,
            pos_ned=pos_act,
            vel_ned=vel,
            vx_cmd=vx,
            vy_cmd=vy,
            z_cmd=zc,
            best_id=np.array(self._log["best_id"], dtype=np.int32),
            best_score=score,
            min_depth=min_d,
            shield_scale=scale,
            dist_goal=dist,
            # extras
            pos_des_ned=pos_des if pos_des.size else np.zeros((0, 3), dtype=np.float32),
            yaw_act_deg=yaw_act if yaw_act.size else np.zeros((0,), dtype=np.float32),
            pitch_act_deg=pitch_act if pitch_act.size else np.zeros((0,), dtype=np.float32),
            roll_act_deg=roll_act if roll_act.size else np.zeros((0,), dtype=np.float32),
            yaw_des_deg=yaw_des if yaw_des.size else np.zeros((0,), dtype=np.float32),
            pitch_des_deg=pitch_des if pitch_des.size else np.zeros((0,), dtype=np.float32),
            roll_des_deg=roll_des if roll_des.size else np.zeros((0,), dtype=np.float32),
        )

        if self.plot_show:
            plt.show()
        else:
            plt.close("all")
        print(f"[INFO] Plots saved to: {self.plot_dir}")


    # ---------- run loop ----------
    def run(self) -> None:
        self.arm_and_takeoff()
        print("[INFO] Running YOPO control loop. Press ESC to stop.")
        print(self.client.simGetCameraInfo("0").pose)
        try:
            while True:
                cont = self.step()
                if not cont:
                    break
                time.sleep(self.ctrl_dt)
        finally:
            try:
                self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            except Exception:
                pass
            cv2.destroyAllWindows()
            if self.plot_on_exit:
                self._plot_logs()
            # if user stopped by ESC, also land if requested
            if self.land_on_goal:
                try:
                    self.land_and_disarm()
                except Exception:
                    pass


if __name__ == "__main__":
    SETTINGS: Dict[str, Any] = {
        "airsim_ip": "127.0.0.1",
        "vehicle_name": "Drone1",

        # Goal
        "goal": [20, 3, -3],  # or {"type":"ue_actor","name":"TargetPoint_1"}
        "goal_reached_m": 2.0,

        # altitude hold
        "target_z_ned": -3.0,

        # Depth sizes
        "image_height": 96,
        "image_width": 160,
        "depth_max_m": 20.0,

        # Control (more stable defaults)
        "control_dt": 0.10,
        "command_duration_s": 0.35,
        "trajectory_lookahead_s": 0.20,
        "max_vel_xy": 4.0,
        "max_vel_norm": 6.0,
        "yopo_test_velocity": 6.0,

        # Safety shield (strongly recommended for near-obstacle stability)
        "enable_safety_shield": True,
        "safety_slow_depth_m": 2.5,
        "safety_stop_depth_m": 1.2,
        "safety_climb_m": 0.8,

        "shield_alpha": 0.85,
        "shield_hold_steps": 12,
        "shield_dir_gap_m": 0.6,
        "shield_emergency_vy": 1.2,
        "shield_repulse_ref_m": 3.0,
        "shield_k_rep": 1.0,
        "shield_vy_cap": 1.0,
        "shield_min_scale": 0.30,
        "shield_corridor_vy_mul": 0.35,
        "shield_corridor_vx_mul": 0.55,
        "shield_slow_d": 2.8,
        "shield_stop_d": 1.2,


        # Fallback (prevents 'fly away' if network input is out-of-distribution)
        "fallback_bad_steps": 6,
        "fallback_speed": 1.5,

        # ROI for min depth
        "min_depth_roi_x0": 0.35,
        "min_depth_roi_x1": 0.65,
        "min_depth_roi_y0": 0.20,
        "min_depth_roi_y1": 0.80,

        # Visualization (dual view)
        "vis_depth": True,
        "vis_trajs": True,
        "traj_vis_samples": 8,
        "vis_scale": 3.0,
        "vis_show_text": False,
        "vis_topk_trajs": 3,
        "vis_draw_endpoints": True,
        "live_topdown_history": 350,
        "live_topdown_size": [360, 320],
        "live_topdown_margin": 18,


        # Camera FOV
        "h_fov_deg": 90.0,
        "v_fov_deg": 60.0,

        # Step prints
        "verbose_step": True,

        # Takeoff / landing
        "takeoff_timeout_s": 8.0,
        "land_on_goal": True,

        # Post-run plots
        "plot_on_exit": True,
        "plot_show": True,
        "plot_dir": "./yopo_logs",
        "subgoal_radius": 8.0,

    }

    runner = AirSimYopoRunner(SETTINGS)
    runner.run()
