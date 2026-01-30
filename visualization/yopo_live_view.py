# visualization/yopo_live_view.py
from __future__ import annotations

import cv2
import numpy as np

from controllers.interfaces import Obs, Plan, Cmd
from planners.yopo.polytraj import build_poly5_traj, sample_poly_positions


# NED -> (N, W, U). Also FRD -> (F, L, U). Keeps right-handedness.
F_NED_TO_NWU = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def ned_to_nwu(v: np.ndarray) -> np.ndarray:
    return (F_NED_TO_NWU @ v.reshape(3, 1)).reshape(3)


def rot_wb_ned_to_nwu(R_wb_ned: np.ndarray) -> np.ndarray:
    """Change of basis: R_wb_nwu = F * R_wb_ned * F."""
    return F_NED_TO_NWU @ R_wb_ned @ F_NED_TO_NWU


class YopoLiveView:
    """Dual-view visualization aligned with the original single-file runner.

    Left: Depth colormap + Top-K candidate trajectories (endpoints + sampled polylines)
    Right: Live top-down XY history (NED x/y)
    """

    def __init__(
        self,
        enabled: bool = True,
        image_width: int = 160,
        image_height: int = 96,
        depth_max_m: float = 20.0,
        h_fov_deg: float = 90.0,
        v_fov_deg: float = 60.0,
        vis_scale: float = 3.0,
        vis_topk_trajs: int = 3,
        traj_vis_samples: int = 8,
        vis_draw_endpoints: bool = True,
        live_topdown_history: int = 350,
        live_topdown_size: tuple[int, int] = (360, 320),  # (w,h)
        live_topdown_margin: int = 18,
        segment_time: float = 0.35,
        vis_show_text: bool = False,
    ):
        self.enabled = bool(enabled)
        self.W = int(image_width)
        self.H = int(image_height)
        self.maxd = float(depth_max_m)

        self.h_fov_deg = float(h_fov_deg)
        self.v_fov_deg = float(v_fov_deg)

        self.vis_scale = float(vis_scale)
        self.vis_topk = int(vis_topk_trajs)
        self.traj_samples = int(traj_vis_samples)
        self.vis_draw_endpoints = bool(vis_draw_endpoints)
        self.vis_show_text = bool(vis_show_text)

        self.topdown_history = int(live_topdown_history)
        self.topdown_size = (int(live_topdown_size[0]), int(live_topdown_size[1]))
        self.topdown_margin = int(live_topdown_margin)

        self.segment_time = float(segment_time)
        self._pos_hist: list[np.ndarray] = []

    # ---------- helpers ----------
    def _depth_to_colormap(self, depth_m: np.ndarray) -> np.ndarray:
        d = np.clip(depth_m, 0.0, self.maxd)
        u8 = (d / self.maxd * 255.0).astype(np.uint8)
        return cv2.applyColorMap(255 - u8, cv2.COLORMAP_TURBO)

    def _project_body_points_to_image(self, pts_b: np.ndarray) -> np.ndarray:
        # pts_b: (N,3) in body frame
        x = pts_b[:, 0]
        y = pts_b[:, 1]
        z = pts_b[:, 2]
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(z, np.sqrt(x * x + y * y) + 1e-8)

        h_fov = np.deg2rad(self.h_fov_deg)
        v_fov = np.deg2rad(self.v_fov_deg)

        u = (yaw / (h_fov / 2.0) + 1.0) * 0.5 * (self.W - 1)
        v = (1.0 - (pitch / (v_fov / 2.0) + 1.0) * 0.5) * (self.H - 1)
        return np.stack([u, v], axis=1)

    def _render_topdown_panel(self, final_goal_ned: np.ndarray, subgoal_ned: np.ndarray) -> np.ndarray:
        """Top-down panel (visualization only):
        - x (North) -> right
        - y (East)  -> down
        """
        w, h = self.topdown_size
        margin = self.topdown_margin
        panel = np.zeros((h, w, 3), dtype=np.uint8)

        if len(self._pos_hist) == 0:
            return panel

        pts = np.stack(self._pos_hist, axis=0)[:, :2].astype(np.float32)
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

        cv2.circle(panel, (u0, v0), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)  # start
        cv2.circle(panel, (u1, v1), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)      # current
        cv2.circle(panel, (ug, vg), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)      # final goal
        cv2.circle(panel, (us, vs), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)    # subgoal

        cv2.putText(panel, "Top-Down (NED x/y)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        return panel

    # ---------- main ----------
    def render(self, obs: Obs, plan: Plan, cmd: Cmd) -> bool:
        if (not self.enabled) or (obs.depth_z_m is None):
            return True

        # update history
        if obs.pos_ned is not None:
            self._pos_hist.append(obs.pos_ned.astype(np.float32))
            if len(self._pos_hist) > self.topdown_history:
                self._pos_hist = self._pos_hist[-self.topdown_history :]

        # left panel: depth + traj overlay
        left = self._depth_to_colormap(obs.depth_z_m)

        # draw top-K trajectories if available
        try:
            endstate = plan.debug.get("endstate", None)
            score = plan.debug.get("score", None)
        except Exception:
            endstate, score = None, None

        if (endstate is not None) and (score is not None) and (obs.vel_ned is not None) and (obs.acc_ned is not None) and (obs.R_wb_ned is not None):
            endstate = np.asarray(endstate, dtype=np.float32).reshape(-1, 9)
            score = np.asarray(score, dtype=np.float32).reshape(-1)

            best_id = int(getattr(plan, "best_id", int(np.argmin(score))))

            # compute start v/a in BODY frame (FLU) like planner
            R_wb_nwu = rot_wb_ned_to_nwu(obs.R_wb_ned)
            R_bw_nwu = R_wb_nwu.T
            vel_b = (R_bw_nwu @ ned_to_nwu(obs.vel_ned).reshape(3, 1)).reshape(3)
            acc_b = (R_bw_nwu @ ned_to_nwu(obs.acc_ned).reshape(3, 1)).reshape(3)
            acc_b[:] = 0.0  # align with planner

            order = np.argsort(score)
            k = int(np.clip(self.vis_topk, 1, len(order)))
            draw_ids = [int(i) for i in order[:k]]
            if best_id not in draw_ids:
                draw_ids = [best_id] + draw_ids[:-1]

            if self.vis_draw_endpoints:
                pts_uv = self._project_body_points_to_image(endstate[:, 0:3])
                for i in draw_ids:
                    u, v = pts_uv[i]
                    if 0 <= u < self.W and 0 <= v < self.H:
                        col = (0, 255, 0) if i == best_id else (255, 255, 255)
                        r = 4 if i == best_id else 2
                        cv2.circle(left, (int(u), int(v)), r, col, -1, lineType=cv2.LINE_AA)

            # polyline samples
            for i in draw_ids:
                traj = build_poly5_traj(vel_b, acc_b, endstate[i], self.segment_time)
                pts_b = sample_poly_positions(traj, self.traj_samples)
                poly = np.round(self._project_body_points_to_image(pts_b)).astype(np.int32)
                keep = (
                    (poly[:, 0] >= 0) & (poly[:, 0] < self.W) &
                    (poly[:, 1] >= 0) & (poly[:, 1] < self.H)
                )
                poly = poly[keep]
                if len(poly) >= 2:
                    col = (0, 255, 0) if i == best_id else (0, 200, 255)
                    cv2.polylines(left, [poly], False, col, 1, lineType=cv2.LINE_AA)

            if self.vis_show_text:
                md = float(getattr(plan, "min_depth", 0.0))
                bs = float(getattr(plan, "best_score", score[best_id]))
                cv2.putText(
                    left,
                    f"best={best_id:02d} score={bs:.3f} minD={md:.2f}m",
                    (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # right panel: top-down
        right = self._render_topdown_panel(plan.final_goal_ned, plan.subgoal_ned)

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

        # ESC to stop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        return True
