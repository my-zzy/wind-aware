# planners/yopo/yopo_planner.py
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from controllers.interfaces import Obs, Plan
from planners.subgoal import compute_subgoal
from planners.yopo.polytraj import build_poly5_traj, clamp_norm as clamp_norm_vec
from perception.roi_depth import compute_min_depth_roi

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
    return clamp_norm_vec(v, float(max_norm))


class YOPOPlanner:
    """
    Rolling MPC-like step: take v at t = control_dt (small step), like original runner.
    Also produces visualization overlays (traj_px) in depth image pixels.
    """

    def __init__(
        self,
        yopo_model,
        image_height: int,
        image_width: int,
        depth_max_m: float,
        segment_time: float,
        control_dt: float,          # rolling step dt
        max_vel_xy: float = 4.0,
        max_vel_norm: float = 6.0,
        goal_length: float = 10.0,
        subgoal_radius: float = 8.0,
        target_z_ned: float = -3.0,
        lookahead_s: float = 0.20,  # kept for debugging
        # ---- visualization (approx pinhole for overlay) ----
        cam_fov_deg: float = 90.0,
        vis_topk: int = 8,
        vis_num_points: int = 20,
    ):
        self.model = yopo_model
        self.device = yopo_model.device

        self.H = int(image_height)
        self.W = int(image_width)
        self.depth_max_m = float(depth_max_m)

        self.segment_time = float(segment_time)
        self.control_dt = float(control_dt)
        self.lookahead_s = float(lookahead_s)

        self.max_vel_xy = float(max_vel_xy)
        self.max_vel_norm = float(max_vel_norm)
        self.goal_length = float(goal_length)
        self.subgoal_radius = float(subgoal_radius)
        self.target_z_ned = float(target_z_ned)

        self.cam_fov_deg = float(cam_fov_deg)
        self.vis_topk = int(vis_topk)
        self.vis_num_points = int(vis_num_points)

    def _preprocess_depth(self, depth_z_m) -> torch.Tensor:
        """
        depth_z_m must be (H,W) ndarray in meters.
        Add strong guards to catch upstream bugs early.
        """
        d = np.asarray(depth_z_m)

        if d.ndim == 0:
            raise TypeError(
                f"Obs.depth_z_m is a scalar ({d.dtype}). "
                f"It must be a (H,W) depth image. "
                f"Likely you passed min_depth into get_obs(depth_z_m=...)."
            )

        if d.ndim == 1:
            if d.size == self.H * self.W:
                d = d.reshape(self.H, self.W)
            else:
                raise TypeError(
                    f"Obs.depth_z_m is 1D with size={d.size}, expected H*W={self.H*self.W}."
                )

        if d.ndim > 2:
            d = np.squeeze(d)
        if d.ndim != 2:
            raise TypeError(f"Obs.depth_z_m has invalid shape {d.shape}, expected (H,W).")

        d = d.astype(np.float32)
        d = np.clip(d, 0.0, float(self.depth_max_m))
        d = d / float(self.depth_max_m)
        return torch.from_numpy(d)[None, None, :, :].to(self.device)

    # ---------------- visualization helpers ----------------
    def _project_flu_points_to_px(self, p_flu: np.ndarray) -> np.ndarray:
        """
        Project body-frame FLU points to depth image pixels using a simple pinhole model.

        Assumptions (for visualization only):
          - traj.pos(t) is in BODY frame (FLU): x forward, y left, z up (per your polytraj.py docstring)
          - camera optical axis is forward (+x), image u right, v down
          - Convert FLU -> FRD for pinhole: (x_f, y_r, z_d) = (x, -y, -z)
        """
        p = np.asarray(p_flu, dtype=np.float32).reshape(-1, 3)
        x = p[:, 0]
        y = -p[:, 1]
        z = -p[:, 2]

        # avoid behind / singular
        x = np.maximum(x, 1e-3)

        fov = np.deg2rad(self.cam_fov_deg)
        fx = 0.5 * float(self.W) / np.tan(0.5 * fov)
        fy = fx
        cx = 0.5 * float(self.W)
        cy = 0.5 * float(self.H)

        u = cx + fx * (y / x)
        v = cy + fy * (z / x)

        px = np.stack([u, v], axis=1).astype(np.float32)
        # clip to bounds
        px[:, 0] = np.clip(px[:, 0], 0, self.W - 1)
        px[:, 1] = np.clip(px[:, 1], 0, self.H - 1)
        return px

    def _build_vis_traj_px(
        self,
        vel_b: np.ndarray,
        acc_b: np.ndarray,
        endstate: np.ndarray,
        score: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build topK poly5 trajectories and project sampled points to depth image pixels.
        Returns:
          traj_px: (K,M,2) float32
          traj_ids: (K,) int32 (candidate indices)
          traj_score: (K,) float32
        """
        score = np.asarray(score).reshape(-1)
        N = int(score.shape[0])
        if N <= 0:
            return None, None, None

        K = int(min(self.vis_topk, N))
        M = int(max(5, self.vis_num_points))

        ids = np.argsort(score)[:K].astype(np.int32)

        traj_px_list = []
        for cid in ids:
            traj = build_poly5_traj(
                start_v=vel_b,
                start_a=acc_b,
                end_state=endstate[int(cid)],
                segment_time=self.segment_time,
            )
            ts = np.linspace(0.0, traj.T, M, dtype=np.float32)
            pts = np.stack([traj.pos(float(t)) for t in ts], axis=0).astype(np.float32)  # (M,3) in FLU
            px = self._project_flu_points_to_px(pts)  # (M,2)
            traj_px_list.append(px)

        traj_px = np.stack(traj_px_list, axis=0).astype(np.float32)
        traj_score = score[ids].astype(np.float32)
        return traj_px, ids, traj_score

    # ---------------- main plan ----------------
    def plan(self, obs: Obs, final_goal_ned: np.ndarray) -> Plan:
        assert obs.depth_z_m is not None, "YOPO needs depth_z_m in Obs"
        assert obs.pos_ned is not None and obs.vel_ned is not None and obs.acc_ned is not None and obs.R_wb_ned is not None

        # rolling subgoal
        subgoal_ned = compute_subgoal(obs.pos_ned, final_goal_ned, self.subgoal_radius)
        goal_ned = subgoal_ned

        # --- NED -> NWU ---
        pos_nwu = ned_to_nwu(obs.pos_ned)
        vel_w_nwu = ned_to_nwu(obs.vel_ned)
        acc_w_nwu = ned_to_nwu(obs.acc_ned)
        goal_w_nwu = ned_to_nwu(goal_ned)

        # body->world in NWU, then world->body
        R_wb_nwu = rot_wb_ned_to_nwu(obs.R_wb_ned)
        R_bw_nwu = R_wb_nwu.T

        # body obs in NWU (FLU)
        vel_b = (R_bw_nwu @ vel_w_nwu.reshape(3, 1)).reshape(3)
        acc_b = (R_bw_nwu @ acc_w_nwu.reshape(3, 1)).reshape(3)

        # goal vector (world) -> clip -> body
        goal_vec_w = goal_w_nwu - pos_nwu
        dist_to_goal = float(np.linalg.norm(goal_vec_w))
        clip = min(1.0, self.goal_length / max(1e-6, dist_to_goal))
        goal_vec_w_clip = goal_vec_w * clip
        goal_b_full = (R_bw_nwu @ goal_vec_w_clip.reshape(3, 1)).reshape(3)

        # YOPO: planar goal only + often acc not used
        goal_b = goal_b_full.copy()
        goal_b[2] = 0.0
        acc_b[:] = 0.0

        yopo_obs = np.concatenate([vel_b, acc_b, goal_b], axis=0).astype(np.float32)
        obs_t = torch.from_numpy(yopo_obs)[None, :].to(self.device)
        depth_t = self._preprocess_depth(obs.depth_z_m)

        # inference
        endstate_t, score_t = self.model.inference(depth_t, obs_t)
        endstate = endstate_t[0].permute(1, 2, 0).reshape(-1, 9).detach().cpu().numpy()
        score = score_t[0].reshape(-1).detach().cpu().numpy()

        best_id = int(np.argmin(score))
        best_score = float(score[best_id])

        # rolling step: build poly and take v at control_dt
        traj = build_poly5_traj(
            start_v=vel_b,
            start_a=acc_b,
            end_state=endstate[best_id],
            segment_time=self.segment_time,
        )
        t_step = float(np.clip(self.control_dt, 0.0, traj.T))
        v_des_b = traj.vel(t_step)
        v_des_b = clamp_norm(v_des_b, self.max_vel_norm)

        # BODY -> WORLD(NWU) -> NED
        v_ref_w_nwu = (R_wb_nwu @ v_des_b.reshape(3, 1)).reshape(3)
        v_ref_ned = nwu_to_ned(v_ref_w_nwu).astype(np.float32)
        v_ref_ned[2] = 0.0

        # clamp XY speed
        vxy = v_ref_ned[:2].astype(np.float32)
        nxy = float(np.linalg.norm(vxy))
        if nxy > self.max_vel_xy:
            vxy = vxy / max(1e-6, nxy) * float(self.max_vel_xy)
        v_ref_ned = np.array([float(vxy[0]), float(vxy[1]), 0.0], dtype=np.float32)

        # depth stats for shield / debug
        min_depth = compute_min_depth_roi(obs.depth_z_m, self.depth_max_m)

        # yaw ref (downstream yaw_controller will smooth)
        yaw_ref_deg = float(np.degrees(np.arctan2(vxy[1], vxy[0]))) if (nxy > 1e-6) else 0.0

        # build vis overlay data (topK)
        traj_px, traj_ids, traj_score = self._build_vis_traj_px(vel_b, acc_b, endstate, score)

        debug: Dict[str, Any] = {
            "endstate": endstate,
            "score": score,
            "v_des_b": v_des_b.astype(np.float32),
            "t_step": t_step,
            # overlay
            "traj_px": traj_px,          # (K,M,2) or None
            "traj_ids": traj_ids,        # (K,) or None
            "traj_score": traj_score,    # (K,) or None
            "cam_fov_deg": float(self.cam_fov_deg),
        }

        return Plan(
            v_ref_ned=v_ref_ned,
            z_cmd_ned=float(self.target_z_ned),
            yaw_ref_deg=float(yaw_ref_deg),
            best_id=int(best_id),
            best_score=float(best_score),
            min_depth=float(min_depth),
            subgoal_ned=subgoal_ned.astype(np.float32),
            final_goal_ned=final_goal_ned.astype(np.float32),
            debug=debug,
        )
