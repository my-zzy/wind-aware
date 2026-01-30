# -*- coding: utf-8 -*-
"""
runners/e2e_windaware_runner.py

Aligned to configs/e2e_windaware.yaml keys.

Visualization (same spirit as airsim_yopo_runner):
  Left : depth colormap + top-K candidate trajectories overlay
  Right: top-down history (NED x/y), showing start/current/final goal/subgoal

On exit:
  Save flight_data.png in original "Flight Data" 3x2 layout (Actual vs Desired).

YAML knobs for window aspect:
  visualize.depth_scale: scales LEFT panel (depth image) before stacking
  visualize.topdown_size: (optional) base size (w,h) for RIGHT panel; then resized to match left height
"""

from __future__ import annotations

import os
import time
import math
import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.airsim_multirotor_env import AirSimMultirotorEnv
from envs.wind_profiles import WindApplier, PROFILES_ALL

from controllers.interfaces import Obs, Plan, Cmd
from controllers.safety_shield import SafetyShield, SafetyShieldConfig

from planners.yopo.yopo_planner import YOPOPlanner
from planners.yopo.yopo_model import YOPOModel
from planners.yopo.polytraj import build_poly5_traj, sample_poly_positions

# MetaPINN
from controllers.wrapper import MetaPINNWrapper, UAV_mass

try:
    import airsim  # type: ignore
except Exception:
    airsim = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore


# ----------------------------
# math utilities
# ----------------------------
F_NED_TO_NWU = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # NED -> (N,W,U)


def ned_to_nwu(v: np.ndarray) -> np.ndarray:
    return (F_NED_TO_NWU @ np.asarray(v, dtype=np.float32).reshape(3, 1)).reshape(3)


def nwu_to_ned(v: np.ndarray) -> np.ndarray:
    return (F_NED_TO_NWU @ np.asarray(v, dtype=np.float32).reshape(3, 1)).reshape(3)


def rot_wb_ned_to_nwu(R_wb_ned: np.ndarray) -> np.ndarray:
    """Change of basis: R_wb_nwu = F * R_wb_ned * F."""
    R = np.asarray(R_wb_ned, dtype=np.float32).reshape(3, 3)
    return F_NED_TO_NWU @ R @ F_NED_TO_NWU


def quat_to_rot_wb_ned(q_wxyz: Tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix mapping body -> world (NED) from quaternion (w,x,y,z)."""
    w, x, y, z = [float(v) for v in q_wxyz]
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


def _euler_from_R_wb_ned(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (roll, pitch, yaw) in radians from body->world rotation matrix in NED.
    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= float(max_norm):
        return vv
    return (vv * (float(max_norm) / (n + 1e-6))).astype(np.float32)


def _compute_subgoal(pos_ned: np.ndarray, final_goal_ned: np.ndarray, subgoal_radius: float) -> np.ndarray:
    vec = np.asarray(final_goal_ned, dtype=np.float32) - np.asarray(pos_ned, dtype=np.float32)
    d = float(np.linalg.norm(vec))
    if d <= float(subgoal_radius):
        return np.asarray(final_goal_ned, dtype=np.float32).copy()
    return (np.asarray(pos_ned, dtype=np.float32) + vec / max(1e-6, d) * float(subgoal_radius)).astype(np.float32)


# ----------------------------
# Visualization (aligned style)
# ----------------------------
class _VisHook:
    """
    Left : depth colormap + top-K candidate trajectories overlay
    Right: top-down history (NED x/y)

    Works best if plan.debug contains:
      - "endstate": (N,9) endstate candidates in BODY frame (FLU)
      - "score":   (N,)
      - "vel_b":   (3,) start velocity in BODY (FLU)
      - "acc_b":   (3,) start accel in BODY (FLU)
    """

    def __init__(
        self,
        enable: bool,
        depth_scale: float,
        topk: int,
        traj_points: int,
        cam_fov_deg: float,
        topdown_history: int,
        topdown_size: Tuple[int, int],  # (w,h)
        topdown_margin: int,
        win: str = "YOPO Live (Depth+Traj | Top-Down)",
    ):
        self.enable = bool(enable) and (cv2 is not None)
        self.depth_scale = float(depth_scale)
        self.topk = int(max(1, topk))
        self.traj_points = int(max(2, traj_points))
        self.h_fov_deg = float(cam_fov_deg)
        self.v_fov_deg = float(cam_fov_deg * 2.0 / 3.0)

        self.topdown_history = int(max(10, topdown_history))
        self.topdown_size = (int(topdown_size[0]), int(topdown_size[1]))
        self.topdown_margin = int(topdown_margin)

        self.win = win
        if self.enable:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    def _depth_to_colormap(self, depth_m: np.ndarray, depth_max: float) -> np.ndarray:
        d = np.asarray(depth_m, dtype=np.float32)
        d = np.nan_to_num(d, nan=depth_max, posinf=depth_max, neginf=0.0)
        d = np.clip(d, 0.0, float(depth_max))
        u8 = (d / float(depth_max) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(255 - u8, cv2.COLORMAP_TURBO)

    def _project_body_points_to_image(self, pts_b: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        pts_b = np.asarray(pts_b, dtype=np.float32).reshape(-1, 3)
        x = pts_b[:, 0]
        y = pts_b[:, 1]
        z = pts_b[:, 2]
        yaw = np.arctan2(y, x + 1e-8)
        pitch = np.arctan2(z, np.sqrt(x * x + y * y) + 1e-8)

        h_fov = np.deg2rad(self.h_fov_deg)
        v_fov = np.deg2rad(self.v_fov_deg)

        u = (yaw / (h_fov / 2.0) + 1.0) * 0.5 * float(img_w - 1)
        v = (1.0 - (pitch / (v_fov / 2.0) + 1.0) * 0.5) * float(img_h - 1)
        return np.stack([u, v], axis=1)

    def _render_topdown_panel(self, pos_hist_ned: np.ndarray, final_goal_ned: np.ndarray, subgoal_ned: np.ndarray) -> np.ndarray:
        w, h = self.topdown_size
        margin = self.topdown_margin
        panel = np.zeros((h, w, 3), dtype=np.uint8)

        if pos_hist_ned is None or pos_hist_ned.size == 0:
            return panel

        pts = np.asarray(pos_hist_ned, dtype=np.float32)[:, :2]
        extra = np.stack([final_goal_ned[:2], subgoal_ned[:2]], axis=0).astype(np.float32)
        all_pts = np.concatenate([pts, extra], axis=0)

        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        dx = max(1e-3, float(xmax - xmin))
        dy = max(1e-3, float(ymax - ymin))

        pad = 0.20
        xmin -= dx * pad
        xmax += dx * pad
        ymin -= dy * pad
        ymax += dy * pad
        dx = max(1e-3, float(xmax - xmin))
        dy = max(1e-3, float(ymax - ymin))

        def world_to_px(pxy: np.ndarray) -> Tuple[int, int]:
            u = int(margin + (pxy[0] - xmin) / dx * (w - 2 * margin))
            v = int(margin + (pxy[1] - ymin) / dy * (h - 2 * margin))
            return u, v

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

        cv2.circle(panel, (u0, v0), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (u1, v1), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (ug, vg), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (us, vs), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        cv2.putText(panel, "Top-Down (NED x/y)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return panel

    def _overlay_trajs(self, left: np.ndarray, plan: Plan, img_w: int, img_h: int, segment_time: float) -> np.ndarray:
        if not isinstance(getattr(plan, "debug", None), dict):
            return left

        dbg = plan.debug
        endstates = dbg.get("endstate", dbg.get("endstates", None))
        scores = dbg.get("score", dbg.get("scores", None))
        vel_b = dbg.get("vel_b", None)
        acc_b = dbg.get("acc_b", None)

        if endstates is None or scores is None or vel_b is None or acc_b is None:
            return left

        endstates = np.asarray(endstates, dtype=np.float32).reshape(-1, 9)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        vel_b = np.asarray(vel_b, dtype=np.float32).reshape(3)
        acc_b = np.asarray(acc_b, dtype=np.float32).reshape(3)
        if endstates.shape[0] != scores.shape[0]:
            return left

        order = np.argsort(scores)
        k = int(np.clip(self.topk, 1, len(order)))
        draw_ids = [int(i) for i in order[:k]]
        if int(plan.best_id) not in draw_ids:
            draw_ids = [int(plan.best_id)] + draw_ids[:-1]

        out = left.copy()

        pts_uv = self._project_body_points_to_image(endstates[:, 0:3], img_w, img_h)
        for i in draw_ids:
            u, v = pts_uv[i]
            if 0 <= u < img_w and 0 <= v < img_h:
                col = (0, 255, 0) if i == int(plan.best_id) else (255, 255, 255)
                r = 4 if i == int(plan.best_id) else 2
                cv2.circle(out, (int(u), int(v)), r, col, -1, lineType=cv2.LINE_AA)

        for i in draw_ids:
            traj = build_poly5_traj(vel_b, acc_b, endstates[i], float(segment_time))
            pts_b = sample_poly_positions(traj, num=self.traj_points, t0=0.0, t1=traj.T)
            poly = np.round(self._project_body_points_to_image(pts_b, img_w, img_h)).astype(np.int32)

            keep = (
                (poly[:, 0] >= 0) & (poly[:, 0] < img_w) &
                (poly[:, 1] >= 0) & (poly[:, 1] < img_h)
            )
            poly = poly[keep]
            if len(poly) >= 2:
                col = (0, 255, 0) if i == int(plan.best_id) else (0, 200, 255)
                cv2.polylines(out, [poly], False, col, 1, lineType=cv2.LINE_AA)

        return out

    def show(
        self,
        depth_m: np.ndarray,
        depth_max: float,
        plan: Plan,
        pos_hist_ned: np.ndarray,
        final_goal_ned: np.ndarray,
        subgoal_ned: np.ndarray,
        segment_time: float,
        img_w: int,
        img_h: int,
    ) -> bool:
        if not self.enable:
            return True

        left = self._depth_to_colormap(depth_m, depth_max)
        left = self._overlay_trajs(left, plan, img_w, img_h, segment_time)

        right = self._render_topdown_panel(pos_hist_ned, final_goal_ned, subgoal_ned)

        if self.depth_scale != 1.0:
            left_s = cv2.resize(
                left,
                (int(left.shape[1] * self.depth_scale), int(left.shape[0] * self.depth_scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            left_s = left

        target_h = left_s.shape[0]
        scale_r = float(target_h) / max(1, right.shape[0])
        right_s = cv2.resize(
            right,
            (int(right.shape[1] * scale_r), target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        vis = np.hstack([left_s, right_s])
        cv2.imshow(self.win, vis)

        key = cv2.waitKey(1) & 0xFF
        return (key != 27)


# ----------------------------
# Runner
# ----------------------------
class E2EWindAwareRunner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # AirSim
        air = cfg["airsim"]
        self.env = AirSimMultirotorEnv(airsim_ip=air["ip"], vehicle_name=air["vehicle_name"])
        self.client = self.env.client
        self.vehicle_name = str(air["vehicle_name"])

        # Output
        out_cfg = cfg.get("output", {})
        self.out_dir = str(out_cfg.get("dir", "outputs"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.flight_png = os.path.join(self.out_dir, str(out_cfg.get("flight_png", "flight_data.png")))

        # Goal
        gcfg = cfg["goal"]
        self.goal_ned = np.array(gcfg["ned"], dtype=np.float32).reshape(3)
        self.goal_reached_m = float(gcfg.get("goal_reached_m", 2.0))
        self.subgoal_radius = float(gcfg.get("subgoal_radius", 8.0))

        # Perception
        pcfg = cfg["perception"]
        self.depth_H = int(pcfg["image_height"])
        self.depth_W = int(pcfg["image_width"])
        self.depth_max = float(pcfg["depth_max_m"])
        self.camera_name = str(pcfg.get("camera_name", "0"))
        self.depth_vis_enable = bool(pcfg.get("depth_visualize", True)) and (cv2 is not None)

        # Control
        ccfg = cfg["control"]
        self.dt = float(ccfg["dt"])
        self.command_duration_s = float(ccfg["command_duration_s"])
        self.target_z_ned = float(ccfg.get("target_z_ned", -3.0))

        # === FIX(1): progress fallback settings ===
        fbcfg = cfg.get("fallback", {})
        self.fallback_bad_steps = int(fbcfg.get("bad_steps", 6))
        self.fallback_speed = float(fbcfg.get("speed", 1.5))
        self.fallback_dist_eps = float(fbcfg.get("dist_eps", 0.5))
        self._prev_dist: Optional[float] = None
        self._bad_progress_count: int = 0

        # YOPO
        ycfg = cfg["yopo"]
        yopo_model = YOPOModel(ycfg["yopo_dir"], ycfg["checkpoint_path"])
        self.planner = self._make_yopo_planner(yopo_model, ycfg, pcfg, ccfg, gcfg)
        self.segment_time = float(ycfg.get("segment_time", 1.7))

        # Shield
        scfg = cfg.get("shield", {})
        self.shield: Optional[SafetyShield] = None
        if bool(scfg.get("enable", True)):
            ss_cfg = SafetyShieldConfig(
                enable=bool(scfg.get("enable", True)),
                max_vel_xy=float(scfg.get("max_vel_xy", ccfg["max_vel_xy"])),
                target_z_ned=float(scfg.get("target_z_ned", self.target_z_ned)),
                depth_max_m=float(scfg.get("depth_max_m", pcfg["depth_max_m"])),

                safety_slow_depth_m=float(scfg.get("safety_slow_depth_m", 2.8)),
                safety_stop_depth_m=float(scfg.get("safety_stop_depth_m", 1.2)),
                safety_climb_m=float(scfg.get("safety_climb_m", 0.8)),

                q_left=float(scfg.get("q_left", 0.03)),
                q_center=float(scfg.get("q_center", 0.03)),
                q_right=float(scfg.get("q_right", 0.03)),
                q_global=float(scfg.get("q_global", 0.01)),

                shield_hold_steps=int(scfg.get("shield_hold_steps", 12)),
                shield_dir_gap_m=float(scfg.get("shield_dir_gap_m", 0.6)),
                shield_emergency_vy=float(scfg.get("shield_emergency_vy", 1.2)),

                shield_repulse_ref_m=float(scfg.get("shield_repulse_ref_m", 3.0)),
                shield_k_rep=float(scfg.get("shield_k_rep", 1.0)),
                shield_vy_cap=float(scfg.get("shield_vy_cap", 1.0)),
                shield_alpha=float(scfg.get("shield_alpha", 0.85)),

                shield_min_scale=float(scfg.get("shield_min_scale", 0.30)),
                shield_slow_d=float(scfg.get("shield_slow_d", 2.8)),
                shield_stop_d=float(scfg.get("shield_stop_d", 1.2)),

                corridor_left_right_m=float(scfg.get("corridor_left_right_m", 2.5)),
                corridor_front_clear_m=float(scfg.get("corridor_front_clear_m", 4.0)),
                shield_corridor_vy_mul=float(scfg.get("shield_corridor_vy_mul", 0.35)),
                shield_corridor_vx_mul=float(scfg.get("shield_corridor_vx_mul", 0.55)),

                left_x1=float(scfg.get("left_x1", 0.33)),
                center_x0=float(scfg.get("center_x0", 0.40)),
                center_x1=float(scfg.get("center_x1", 0.60)),
                right_x0=float(scfg.get("right_x0", 0.67)),
            )
            self.shield = SafetyShield(ss_cfg)

        # Wind
        wcfg = cfg.get("wind", {})
        self.wind_enable = bool(wcfg.get("enable", False))
        self.wind_name = str(wcfg.get("profile", "0mps"))
        self.wind_dt = float(wcfg.get("dt", self.dt))
        self.wind_applier = WindApplier()
        self.wind_profile = PROFILES_ALL.get(self.wind_name, PROFILES_ALL["0mps"])

        # MetaPINN
        mcfg = cfg.get("metapinn", {})
        self.meta_enable = bool(mcfg.get("enable", False))
        self.meta: Optional[MetaPINNWrapper] = None
        self.meta_keys: list[str] = []
        if self.meta_enable:
            model = mcfg["model"]
            keys = [s.strip() for s in str(model["feature_keys"]).split(",") if s.strip()]
            self.meta_keys = keys
            self.meta = MetaPINNWrapper(
                feature_keys=keys,
                scaler_path=model.get("mp_scaler", None),
                load_path=model.get("mp_load", None),
                lr=float(mcfg.get("online", {}).get("lr", 1e-3)),
                update_every=int(mcfg.get("online", {}).get("update_every", 10)),
            )

            fusion = mcfg.get("fusion", {})
            self.fusion_type = str(fusion.get("type", "vel_ff")).lower()
            self.k_ff = float(fusion.get("k_ff", 0.35))
            self.max_delta_v = float(fusion.get("max_delta_v", 1.2))
            self.cond_from = str(fusion.get("cond_from", "speed")).lower()
        else:
            self.fusion_type = "none"
            self.k_ff = 0.0
            self.max_delta_v = 0.0
            self.cond_from = "speed"

        # Visualize (YAML aligned)
        vcfg = cfg.get("visualize", {})
        self.vis = _VisHook(
            enable=bool(vcfg.get("enable", True)),
            depth_scale=float(vcfg.get("depth_scale", 1.0)),
            topk=int(vcfg.get("topk", 8)),
            traj_points=int(vcfg.get("traj_points", 20)),
            cam_fov_deg=float(vcfg.get("cam_fov_deg", 90.0)),
            topdown_history=int(vcfg.get("topdown_history", 350)),
            topdown_size=tuple(vcfg.get("topdown_size", [360, 320])),
            topdown_margin=int(vcfg.get("topdown_margin", 18)),
        )

        # Debug
        self.print_each_step = bool(cfg.get("debug", {}).get("print_each_step", False))

        # Plot-only desired position integrator
        self._des_pos_ned: Optional[np.ndarray] = None

        # === FIX(2): wall-time bases for logs/sim time ===
        self._t0_wall = time.time()
        self._last_wall = self._t0_wall

        # Logs (for original Flight Data plotting)
        self._log: Dict[str, list] = {
            "t": [],
            "pos_ned": [],
            "vel_ned": [],
            "pos_des_ned": [],
            "yaw_act_deg": [],
            "pitch_act_deg": [],
            "roll_act_deg": [],
            "yaw_des_deg": [],
            "pitch_des_deg": [],
            "roll_des_deg": [],
        }

    def _make_yopo_planner(
        self,
        yopo_model: YOPOModel,
        ycfg: Dict[str, Any],
        pcfg: Dict[str, Any],
        ccfg: Dict[str, Any],
        gcfg: Dict[str, Any],
    ) -> YOPOPlanner:
        sig = inspect.signature(YOPOPlanner.__init__)
        kwargs = dict(
            yopo_model=yopo_model,
            image_height=pcfg["image_height"],
            image_width=pcfg["image_width"],
            depth_max_m=pcfg["depth_max_m"],
            segment_time=ycfg["segment_time"],
            max_vel_xy=ccfg["max_vel_xy"],
            max_vel_norm=ccfg["max_vel_norm"],
            goal_length=gcfg.get("yopo_goal_length", 10.0),
            subgoal_radius=gcfg.get("subgoal_radius", 8.0),
            target_z_ned=ccfg.get("target_z_ned", -3.0),
            lookahead_s=float(ccfg.get("trajectory_lookahead_s", 0.20)),
        )
        if "control_dt" in sig.parameters:
            kwargs["control_dt"] = float(ccfg.get("dt", 0.10))
        return YOPOPlanner(**kwargs)

    def _get_depth_image(self) -> np.ndarray:
        """
        Depth fetch:
          - read DepthPerspective float
          - NaN/Inf -> depth_max
          - clip
          - resize to (W,H)
        """
        if airsim is None:
            raise RuntimeError("Cannot import airsim. Please `pip install airsim`.")

        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)],
            vehicle_name=self.vehicle_name,
        )
        if (not responses) or (len(responses[0].image_data_float) == 0):
            raise RuntimeError("Depth image is empty. Check camera_name / ImageRequest.")

        d = responses[0]
        depth = np.array(d.image_data_float, dtype=np.float32).reshape(d.height, d.width)
        depth = np.nan_to_num(depth, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        depth = np.clip(depth, 0.0, self.depth_max)

        if (depth.shape[0] != self.depth_H) or (depth.shape[1] != self.depth_W):
            if cv2 is not None:
                depth = cv2.resize(depth, (self.depth_W, self.depth_H), interpolation=cv2.INTER_NEAREST)
            else:
                ys = (np.linspace(0, depth.shape[0] - 1, self.depth_H)).astype(np.int32)
                xs = (np.linspace(0, depth.shape[1] - 1, self.depth_W)).astype(np.int32)
                depth = depth[ys][:, xs]

        return depth.astype(np.float32)

    # ---------------- MetaPINN ----------------
    def _build_meta_features(self, obs: Obs, cmd_vxy: Optional[np.ndarray] = None) -> np.ndarray:
        parts = []
        for key in self.meta_keys:
            if key == "v":
                parts.append(np.asarray(obs.vel_ned, dtype=np.float32).reshape(3))
            elif key == "q":
                qw, qx, qy, qz = obs.quat_wxyz
                parts.append(np.asarray([qw, qx, qy, qz], dtype=np.float32))
            elif key == "pwm":
                if obs.rotor_speeds is None:
                    parts.append(np.zeros(4, dtype=np.float32))
                else:
                    parts.append(np.asarray(obs.rotor_speeds, dtype=np.float32).reshape(4))
            elif key == "p":
                parts.append(np.asarray(obs.pos_ned, dtype=np.float32).reshape(3))
            elif key == "cmd_vxy":
                if cmd_vxy is None:
                    parts.append(np.zeros(2, dtype=np.float32))
                else:
                    parts.append(np.asarray(cmd_vxy, dtype=np.float32).reshape(2))
            else:
                parts.append(np.zeros(1, dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _fusion(self, v_yopo: np.ndarray, obs: Obs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        v_yopo = np.asarray(v_yopo, dtype=np.float32).reshape(3)

        if (not self.meta_enable) or (self.meta is None) or (self.fusion_type == "none"):
            return v_yopo, None

        cond_val: Optional[float] = None
        if self.cond_from == "speed":
            cond_val = float(np.linalg.norm(np.asarray(obs.vel_ned, dtype=np.float32)[:2]))

        feat = self._build_meta_features(obs, cmd_vxy=v_yopo[:2])
        fhat = self.meta.predict(feat, cond_val=cond_val)  # (3,) N

        a_hat = np.asarray(fhat, dtype=np.float32).reshape(3) / float(UAV_mass)
        v_cmd = v_yopo.copy()

        if self.fusion_type in ("vel_ff", "acc_ff"):
            # NOTE: you set k_ff negative in YAML to get "-" effect
            dv = float(self.k_ff) * a_hat * float(self.dt)
            dv_xy = _clamp_norm(dv[:2], float(self.max_delta_v))
            v_cmd[0] += float(dv_xy[0])
            v_cmd[1] += float(dv_xy[1])
            v_cmd[2] = 0.0

        v_cmd[:2] = _clamp_norm(v_cmd[:2], float(self.cfg["control"]["max_vel_xy"]))
        v_cmd = _clamp_norm(v_cmd, float(self.cfg["control"]["max_vel_norm"]))
        v_cmd[2] = 0.0
        return v_cmd.astype(np.float32), np.asarray(fhat, dtype=np.float32).reshape(3)

    # ---------------- Plot (original Flight Data 3x2) ----------------
    def _save_flight_png(self) -> None:
        if len(self._log["t"]) < 2:
            print("[E2E] no logs, skip saving flight_data.png")
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[E2E] matplotlib not found, skip saving flight_data.png")
            return

        t = np.array(self._log["t"], dtype=np.float32)
        pos_act = np.array(self._log["pos_ned"], dtype=np.float32)
        pos_des = np.array(self._log["pos_des_ned"], dtype=np.float32)

        yaw_act = np.array(self._log["yaw_act_deg"], dtype=np.float32)
        pitch_act = np.array(self._log["pitch_act_deg"], dtype=np.float32)
        roll_act = np.array(self._log["roll_act_deg"], dtype=np.float32)

        yaw_des = np.array(self._log["yaw_des_deg"], dtype=np.float32)
        pitch_des = np.array(self._log["pitch_des_deg"], dtype=np.float32)
        roll_des = np.array(self._log["roll_des_deg"], dtype=np.float32)

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("Flight Data")

        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(t, pos_act[:, 0], label="Actual")
        ax1.plot(t, pos_des[:, 0], "r--", label="Desired")
        ax1.set_title("X Position"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("X (m)")
        ax1.grid(True); ax1.legend()

        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(t, pos_act[:, 1], label="Actual")
        ax2.plot(t, pos_des[:, 1], "r--", label="Desired")
        ax2.set_title("Y Position"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Y (m)")
        ax2.grid(True); ax2.legend()

        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(t, pos_act[:, 2], label="Actual")
        ax3.plot(t, pos_des[:, 2], "r--", label="Desired")
        ax3.set_title("Z Position"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Z (m)")
        ax3.grid(True); ax3.legend()

        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(t, yaw_act, label="Actual")
        ax4.plot(t, yaw_des, "r--", label="Desired")
        ax4.set_title("Yaw Angle"); ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Yaw (deg)")
        ax4.grid(True); ax4.legend()

        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(t, pitch_act, label="Actual")
        ax5.plot(t, pitch_des, "r--", label="Desired")
        ax5.set_title("Pitch Angle"); ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Pitch (deg)")
        ax5.grid(True); ax5.legend()

        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(t, roll_act, label="Actual")
        ax6.plot(t, roll_des, "r--", label="Desired")
        ax6.set_title("Roll Angle"); ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Roll (deg)")
        ax6.grid(True); ax6.legend()

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(self.flight_png, dpi=180)
        plt.close(fig)
        print(f"[E2E] saved {self.flight_png}")

    def run(self) -> None:
        print("[E2E] enable + takeoff ...")
        self.env.enable()
        self.env.takeoff(target_z_ned=float(self.target_z_ned), timeout=10.0, vel=2.0)

        # === FIX(2): use wall-time sim clock ===
        t_sim = 0.0
        self._t0_wall = time.time()
        self._last_wall = self._t0_wall

        try:
            while True:
                # --- wall dt ---
                now = time.time()
                dt_wall = float(now - self._last_wall)
                self._last_wall = now
                dt_wall = float(np.clip(dt_wall, 0.0, 0.5))
                t_sim = float(now - self._t0_wall)

                # depth
                depth = self._get_depth_image()

                # obs
                obs = self.env.get_obs(depth_z_m=depth)

                # wind (use wall-time)
                if self.wind_enable:
                    self.wind_applier.apply_wind(self.client, self.wind_profile, t_sim, dt=self.wind_dt)

                pos_ned = np.asarray(obs.pos_ned, dtype=np.float32).reshape(3)
                final_goal_ned = np.asarray(self.goal_ned, dtype=np.float32).reshape(3)

                # --- goal check uses FINAL goal ---
                dist_final = float(np.linalg.norm(final_goal_ned - pos_ned))
                if dist_final <= float(self.goal_reached_m):
                    print(f"[E2E] goal reached: dist={dist_final:.2f}m <= {self.goal_reached_m:.2f}m")
                    break

                # === FIX(3): rolling subgoal fed into planner (like original runner) ===
                subgoal_ned = _compute_subgoal(pos_ned, final_goal_ned, self.subgoal_radius)
                goal_for_plan = subgoal_ned

                # planner
                plan = self.planner.plan(obs, goal_for_plan)

                # for visualization: compute vel_b/acc_b (body FLU)
                R_wb_ned = quat_to_rot_wb_ned(tuple(obs.quat_wxyz))
                R_wb_nwu = rot_wb_ned_to_nwu(R_wb_ned)
                R_bw_nwu = R_wb_nwu.T
                vel_w_nwu = ned_to_nwu(np.asarray(obs.vel_ned, dtype=np.float32).reshape(3))
                vel_b = (R_bw_nwu @ vel_w_nwu.reshape(3, 1)).reshape(3)
                acc_b = np.zeros(3, dtype=np.float32)

                if isinstance(getattr(plan, "debug", None), dict):
                    plan.debug["vel_b"] = vel_b.astype(np.float32)
                    plan.debug["acc_b"] = acc_b.astype(np.float32)

                # metapinn fusion
                v_cmd, fhat = self._fusion(plan.v_ref_ned, obs)

                cmd_pre = Cmd(
                    vx=float(v_cmd[0]),
                    vy=float(v_cmd[1]),
                    z=float(plan.z_cmd_ned),
                    yaw_deg=float(plan.yaw_ref_deg),
                    meta={
                        "best_id": int(plan.best_id),
                        "best_score": float(plan.best_score),
                        "min_depth": float(plan.min_depth),
                        "dist_final": float(dist_final),
                        "dist_sub": float(np.linalg.norm(subgoal_ned - pos_ned)),
                        "fhat": fhat.tolist() if fhat is not None else None,
                    },
                )

                # shield
                cmd = cmd_pre
                shield_scale = 1.0
                if self.shield is not None:
                    n0 = float(np.linalg.norm([cmd_pre.vx, cmd_pre.vy]))
                    cmd = self.shield.apply(obs, cmd_pre)
                    n1 = float(np.linalg.norm([cmd.vx, cmd.vy]))
                    shield_scale = (n1 / max(1e-6, n0)) if n0 > 1e-6 else 1.0
                    cmd.meta["shield_scale"] = float(shield_scale)

                # === FIX(1): progress monitor fallback (use FINAL goal distance) ===
                if self._prev_dist is None:
                    self._prev_dist = dist_final
                else:
                    if dist_final > float(self._prev_dist) + float(self.fallback_dist_eps):
                        self._bad_progress_count += 1
                    else:
                        self._bad_progress_count = 0
                    self._prev_dist = dist_final

                if self._bad_progress_count >= int(self.fallback_bad_steps):
                    g = (final_goal_ned - pos_ned).astype(np.float32)
                    gxy = g[:2]
                    gn = float(np.linalg.norm(gxy))
                    if gn > 1e-3:
                        gdir = gxy / gn
                        # override vxy to pull back (keep z/yaw as-is)
                        cmd.vx = float(gdir[0] * float(self.fallback_speed))
                        cmd.vy = float(gdir[1] * float(self.fallback_speed))
                        cmd.meta["fallback"] = True
                    else:
                        cmd.meta["fallback"] = False
                else:
                    cmd.meta["fallback"] = False

                # send cmd
                self.env.send_cmd(cmd, duration=float(self.command_duration_s))

                # ------- logs for Flight Data plotting (wall-time) -------
                t_now = float(now - self._t0_wall)
                self._log["t"].append(t_now)
                self._log["pos_ned"].append(pos_ned.astype(np.float32).tolist())
                self._log["vel_ned"].append(np.asarray(obs.vel_ned, dtype=np.float32).tolist())

                # === FIX(4): desired pos integrator uses dt_eff aligned to duration/wall ===
                if self._des_pos_ned is None:
                    self._des_pos_ned = pos_ned.copy()
                else:
                    dt_eff = max(float(dt_wall), float(self.command_duration_s))
                    self._des_pos_ned[0] += float(cmd.vx) * dt_eff
                    self._des_pos_ned[1] += float(cmd.vy) * dt_eff
                    self._des_pos_ned[2] = float(cmd.z)
                self._log["pos_des_ned"].append(self._des_pos_ned.astype(np.float32).tolist())

                # attitude actual/desired
                roll_rad, pitch_rad, yaw_rad = _euler_from_R_wb_ned(R_wb_ned)
                self._log["roll_act_deg"].append(float(np.degrees(roll_rad)))
                self._log["pitch_act_deg"].append(float(np.degrees(pitch_rad)))
                self._log["yaw_act_deg"].append(float(np.degrees(yaw_rad)))

                self._log["yaw_des_deg"].append(float(cmd.yaw_deg))
                self._log["pitch_des_deg"].append(0.0)
                self._log["roll_des_deg"].append(0.0)

                if self.print_each_step:
                    print(
                        f"[step] dist_final={dist_final:.2f}  dist_sub={np.linalg.norm(subgoal_ned-pos_ned):.2f}  "
                        f"minD={plan.min_depth:.2f}  best={plan.best_id}/{plan.best_score:.3f}  "
                        f"v_ref=({plan.v_ref_ned[0]:.2f},{plan.v_ref_ned[1]:.2f})  "
                        f"v_cmd=({cmd.vx:.2f},{cmd.vy:.2f})  "
                        f"scale={shield_scale:.2f}  fb={cmd.meta.get('fallback', False)}"
                    )

                # visualize
                if self.depth_vis_enable:
                    hist = np.array(self._log["pos_ned"], dtype=np.float32)
                    if hist.shape[0] > self.vis.topdown_history:
                        hist = hist[-self.vis.topdown_history :]

                    ok = self.vis.show(
                        depth_m=depth,
                        depth_max=self.depth_max,
                        plan=plan,
                        pos_hist_ned=hist,
                        final_goal_ned=final_goal_ned,
                        subgoal_ned=subgoal_ned,
                        segment_time=float(self.segment_time),
                        img_w=self.depth_W,
                        img_h=self.depth_H,
                    )
                    if not ok:
                        print("[E2E] user exit (ESC)")
                        break

                # pacing: keep roughly dt (optional)
                # 注意：command_duration_s 可能 > dt，这里只是让循环不“空转”
                sleep_s = max(0.0, float(self.dt) - float(dt_wall))
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\n[E2E] interrupted by user")
        finally:
            print("[E2E] landing ...")
            try:
                self.env.land()
            except Exception:
                pass
            try:
                self.env.disable()
            except Exception:
                pass
            if cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            try:
                self._save_flight_png()
            except Exception:
                pass


# Optional CLI entry (if you want to run directly)
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/e2e_windaware.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runner = E2EWindAwareRunner(cfg)
    runner.run()
