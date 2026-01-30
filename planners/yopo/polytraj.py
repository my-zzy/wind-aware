# planners/yopo/polytraj.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

def _get_poly5_solver():
    # local import to avoid import-time crash
    try:
        from policy.poly_solver import Poly5Solver  # type: ignore
        return Poly5Solver
    except Exception:
        try:
            from policy.poly_solver import Polys5Solver as Poly5Solver  # type: ignore
            return Poly5Solver
        except Exception as e:
            raise ModuleNotFoundError(
                "Cannot import YOPO poly solver. "
                "Make sure YOPO repo is on PYTHONPATH (see ensure_yopo_on_path)."
            ) from e


@dataclass
class PolyTrajConfig:
    segment_time: float = 0.35  # seconds
    vis_samples: int = 8        # number of samples for visualization polyline


@dataclass
class Poly5Traj:
    """
    A 3D 5th-order polynomial trajectory defined in BODY frame (usually FLU in your YOPO pipeline).
    It is parameterized by start velocity/acceleration and an end_state (p,v,a) at t=T.

    NOTE:
      The solver in YOPO typically assumes p(0)=0 and uses v(0), a(0) as boundary.
      End boundary uses p(T), v(T), a(T).
    """
    sol_x: any
    sol_y: any
    sol_z: any
    T: float

    def pos(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, self.T))
        return np.array(
            [self.sol_x.get_position(t), self.sol_y.get_position(t), self.sol_z.get_position(t)],
            dtype=np.float32,
        )

    def vel(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, self.T))
        return np.array(
            [self.sol_x.get_velocity(t), self.sol_y.get_velocity(t), self.sol_z.get_velocity(t)],
            dtype=np.float32,
        )

    def acc(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, self.T))
        return np.array(
            [self.sol_x.get_acceleration(t), self.sol_y.get_acceleration(t), self.sol_z.get_acceleration(t)],
            dtype=np.float32,
        )


def build_poly5_traj(
    start_v: np.ndarray,
    start_a: np.ndarray,
    end_state: np.ndarray,
    segment_time: float,
) -> Poly5Traj:
    """
    Build 3D Poly5 trajectory for a given end_state.

    Args:
      start_v: (3,) start velocity in body frame
      start_a: (3,) start acceleration in body frame
      end_state: (9,) = [px,py,pz, vx,vy,vz, ax,ay,az] at t=T, in body frame
      segment_time: T

    Returns:
      Poly5Traj
    """
    Poly5Solver = _get_poly5_solver()
    sv = np.asarray(start_v, dtype=np.float32).reshape(3)
    sa = np.asarray(start_a, dtype=np.float32).reshape(3)
    es = np.asarray(end_state, dtype=np.float32).reshape(9)

    end_p = es[0:3]
    end_v = es[3:6]
    end_a = es[6:9]
    T = float(segment_time)

    sol_x = Poly5Solver(
        0.0, float(sv[0]), float(sa[0]),
        float(end_p[0]), float(end_v[0]), float(end_a[0]),
        T,
    )
    sol_y = Poly5Solver(
        0.0, float(sv[1]), float(sa[1]),
        float(end_p[1]), float(end_v[1]), float(end_a[1]),
        T,
    )
    sol_z = Poly5Solver(
        0.0, float(sv[2]), float(sa[2]),
        float(end_p[2]), float(end_v[2]), float(end_a[2]),
        T,
    )

    return Poly5Traj(sol_x=sol_x, sol_y=sol_y, sol_z=sol_z, T=T)


def sample_poly_positions(
    traj: Poly5Traj,
    num: int,
    t0: float = 0.0,
    t1: Optional[float] = None,
) -> np.ndarray:
    """
    Sample positions along trajectory for visualization.

    Returns:
      pts: (num,3)
    """
    if t1 is None:
        t1 = traj.T
    num = int(max(2, num))
    ts = np.linspace(float(t0), float(t1), num, dtype=np.float32)
    pts = np.stack([traj.pos(float(t)) for t in ts], axis=0).astype(np.float32)
    return pts


def desired_velocity_at_lookahead(
    start_v: np.ndarray,
    start_a: np.ndarray,
    end_state: np.ndarray,
    segment_time: float,
    lookahead_t: float,
) -> Tuple[np.ndarray, Poly5Traj]:
    """
    Convenience: build trajectory and return v_des at t=lookahead_t.

    Returns:
      v_des: (3,)
      traj: Poly5Traj
    """
    traj = build_poly5_traj(start_v, start_a, end_state, segment_time)
    t_la = float(np.clip(lookahead_t, 0.0, traj.T))
    v_des = traj.vel(t_la)
    return v_des, traj


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """
    Clamp vector magnitude.
    """
    vv = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= float(max_norm):
        return vv.astype(np.float32)
    return (vv * (float(max_norm) / (n + 1e-8))).astype(np.float32)
