# runners/metapinn_runner.py
from __future__ import annotations

import os
import time
import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Any

import numpy as np

from envs.airsim_multirotor_env import AirSimMultirotorEnv
from envs.wind_profiles import PROFILES_ALL, WindApplier

from controllers.wrapper import MetaPINNWrapper, UAV_mass
from controllers.online_adapt import OnlineAdaptConfig, OnlineAdapter

# ----------------------------
# IO
# ----------------------------
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------
# Math utils
# ----------------------------
G = 9.81


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * cp * cy
    return [qw, qx, qy, qz]


# ----------------------------
# Trajectories (NED, z is negative for up)
# ----------------------------
def traj_fig8(t: float) -> Tuple[float, float, float, float]:
    x = 10.0 * math.sin(0.5 * t)
    y = 10.0 * math.sin(0.5 * t) * math.cos(0.5 * t)
    z = -10.0
    yaw = 0.0
    return x, y, z, yaw


def traj_circle(t: float) -> Tuple[float, float, float, float]:
    x = 6.0 * math.cos(0.4 * t)
    y = 6.0 * math.sin(0.4 * t)
    z = -10.0
    yaw = 0.0
    return x, y, z, yaw


def traj_ellipse(t: float) -> Tuple[float, float, float, float]:
    x = 8.0 * math.cos(0.35 * t)
    y = 14.0 * math.sin(0.35 * t)
    z = -10.0
    yaw = 0.0
    return x, y, z, yaw


def traj_random_spline(total_time: float = 60.0, seed: int = 0) -> Callable[[float], Tuple[float, float, float, float]]:
    # same spirit as your original: cubic spline random waypoints
    import numpy as _np
    from scipy.interpolate import CubicSpline

    rng = _np.random.RandomState(seed)
    num_points = 15
    t_points = _np.linspace(0, total_time, num_points)
    x_points = rng.uniform(-15, 15, num_points)
    y_points = rng.uniform(-15, 15, num_points)
    z_points = rng.uniform(-15, -5, num_points)
    yaw_points = rng.uniform(-_np.pi, _np.pi, num_points)

    x_s = CubicSpline(t_points, x_points)
    y_s = CubicSpline(t_points, y_points)
    z_s = CubicSpline(t_points, z_points)
    yaw_s = CubicSpline(t_points, yaw_points)

    def f(t: float):
        tt = float(_np.clip(t, 0, total_time))
        return float(x_s(tt)), float(y_s(tt)), float(z_s(tt)), 0.0

    return f


# ----------------------------
# Adaptive controller (integrated residual -> adaptive)
# This is essentially your function, packed for runner usage.
# ----------------------------
@dataclass
class AdaptiveGains:
    # you can tune these in YAML if needed
    cx: float = 1.6
    cy: float = 1.6
    cz: float = 1.8
    cu: float = 2.2
    cv: float = 2.2
    cw: float = 2.6
    lamx: float = 1.2
    lamy: float = 1.2
    lamz: float = 1.5


def adaptive_controller(
    gains: AdaptiveGains,
    pos_hist, vel_hist, att_hist, ang_hist,
    posd_hist, attd_hist,
    dhat, jifen,
    dt_eff: float,
    t: float,
    *,
    fhat: Optional[np.ndarray] = None,
):
    fx, fy, fz = (0.0, 0.0, 0.0) if (fhat is None) else (float(fhat[0]), float(fhat[1]), float(fhat[2]))

    x, y, z = pos_hist[0][-1], pos_hist[1][-1], pos_hist[2][-1]
    u, v, w = vel_hist[0][-1], vel_hist[1][-1], vel_hist[2][-1]
    phi, theta, psi = att_hist[0][-1], att_hist[1][-1], att_hist[2][-1]
    xd, yd, zd, psid = posd_hist[0][-1], posd_hist[1][-1], posd_hist[2][-1], attd_hist[2][-1]

    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen

    # desired vel/acc (finite diff)
    dt = float(max(1e-6, dt_eff))
    if len(posd_hist[0]) >= 2 and dt_eff > 0:
        xd_dot = (posd_hist[0][-1] - posd_hist[0][-2]) / dt_eff
        yd_dot = (posd_hist[1][-1] - posd_hist[1][-2]) / dt_eff
        zd_dot = (posd_hist[2][-1] - posd_hist[2][-2]) / dt_eff
    else:
        xd_dot = yd_dot = zd_dot = 0.0

    if len(posd_hist[0]) >= 3 and dt_eff > 0:
        xd_dot2 = ((posd_hist[0][-1] - posd_hist[0][-2]) / dt_eff - (posd_hist[0][-2] - posd_hist[0][-3]) / dt_eff) / dt_eff
        yd_dot2 = ((posd_hist[1][-1] - posd_hist[1][-2]) / dt_eff - (posd_hist[1][-2] - posd_hist[1][-3]) / dt_eff) / dt_eff
        zd_dot2 = ((posd_hist[2][-1] - posd_hist[2][-2]) / dt_eff - (posd_hist[2][-2] - posd_hist[2][-3]) / dt_eff) / dt_eff
    else:
        xd_dot2 = yd_dot2 = zd_dot2 = 0.0

    # errors
    ex = x - xd
    ey = y - yd
    ez = z - zd

    eu = u - xd_dot + gains.cx * ex
    ev = v - yd_dot + gains.cy * ey
    ew = w - zd_dot + gains.cz * ez

    # remove residual force (world) as accel proxy
    eu_tilde = eu - fx / float(UAV_mass)
    ev_tilde = ev - fy / float(UAV_mass)
    ew_tilde = ew - fz / float(UAV_mass)

    # error derivatives
    ex_dot = eu - gains.cx * ex
    ey_dot = ev - gains.cy * ey
    ez_dot = ew - gains.cz * ez

    # desired accel (deduct dhat)
    u_dot = -gains.cu * eu - ex + xd_dot2 - gains.cx * ex_dot
    v_dot = -gains.cv * ev - ey + yd_dot2 - gains.cy * ey_dot
    w_dot = -gains.cw * ew - ez + zd_dot2 - gains.cz * ez_dot

    # adaptive law (freeze by dt_eff=0)
    dz_hat += gains.lamz * ew_tilde * dt_eff
    dx_hat += gains.lamx * eu_tilde * dt_eff
    dy_hat += gains.lamy * ev_tilde * dt_eff

    # throttle (SimpleFlight: 0..1)
    thrust_force = -(w_dot - dz_hat - G) * float(UAV_mass) / (math.cos(phi) * math.cos(theta) + 1e-6)
    throttle = clip((thrust_force / (float(UAV_mass) * G)) * 0.5 + 0.5, 0.0, 1.0)

    # lateral accel -> roll/pitch
    accel_x_desired = clip(u_dot - dx_hat, -5.0, 5.0)
    accel_y_desired = clip(-(v_dot - dy_hat), -5.0, 5.0)

    roll_des = -(accel_y_desired * math.cos(psi) - accel_x_desired * math.sin(psi)) / G
    pitch_des = (accel_x_desired * math.cos(psi) + accel_y_desired * math.sin(psi)) / G

    max_angle = math.radians(30)
    roll_des = clip(roll_des, -max_angle, max_angle)
    pitch_des = clip(pitch_des, -max_angle, max_angle)
    yaw_des = psid

    dhat_new = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_new = [xphi, xtheta, xpsi]
    return throttle, roll_des, pitch_des, yaw_des, dhat_new, jifen_new, (u_dot, v_dot, w_dot)


# ----------------------------
# Runner config
# ----------------------------
@dataclass
class MetaPINNRunnerConfig:
    airsim_ip: str = "127.0.0.1"
    vehicle_name: str = "Drone1"

    # timing
    dt: float = 0.01
    total_time: float = 60.0
    freeze_T: float = 2.0

    # wind
    wind_profile: str = "10mps"   # key in PROFILES_ALL
    wind_condition_name: str = "10wind"  # used for warm-start naming

    # trajectory
    traj: str = "fig8"            # fig8/circle/ellipse/random
    random_seed: int = 0

    # metapinn model/scaler
    feature_keys: str = "v,q,pwm"
    mp_load: str = "saved_models/meta_pinn_offline/meta_pinn_last.pth"
    mp_scaler: str = "saved_models/meta_pinn_offline/x_scaler.npz"

    # online adapt
    enable_meta: bool = True
    enable_online_learn: bool = True
    update_every: int = 10
    lr: float = 1e-3
    warm_dir: str = "warm_start"

    # SimpleFlight mapping
    hover_throttle: float = 0.594
    feedforward_g: float = 0.3

    # logging
    out_dir: str = "result"
    save_csv: bool = True


# ----------------------------
# Runner
# ----------------------------
class MetaPINNRunner:
    """
    AirSim(SimpleFlight) + Wind + (MetaPINN predict + optional online adapt + warm-start)
    Control: roll/pitch/yaw/throttle, with adaptive_controller (your original).
    """

    def __init__(self, cfg: Dict[str, Any]):
        # merge dict into dataclass defaults
        base = MetaPINNRunnerConfig()
        for k, v in cfg.items():
            if hasattr(base, k):
                setattr(base, k, v)
        self.cfg = base

        self.env = AirSimMultirotorEnv(airsim_ip=self.cfg.airsim_ip, vehicle_name=self.cfg.vehicle_name)
        self.wind = WindApplier()

        # meta wrapper + online adapter
        keys = [s.strip() for s in str(self.cfg.feature_keys).split(",") if s.strip()]
        self.feature_keys = keys

        self.wrapper = MetaPINNWrapper(
            feature_keys=keys,
            scaler_path=self.cfg.mp_scaler,
            load_path=self.cfg.mp_load,
            lr=float(self.cfg.lr),
            update_every=int(self.cfg.update_every),
        )

        oa_cfg = OnlineAdaptConfig(
            enable_online_learn=bool(self.cfg.enable_online_learn),
            update_every=int(self.cfg.update_every),
            warm_dir=str(self.cfg.warm_dir),
        )
        self.online = OnlineAdapter(self.wrapper, oa_cfg, warm_condition=str(self.cfg.wind_condition_name))

        # state
        self.pos_hist = [[], [], []]
        self.vel_hist = [[], [], []]
        self.att_hist = [[], [], []]
        self.ang_hist = [[], [], []]
        self.posd_hist = [[], [], []]
        self.attd_hist = [[], [], []]
        self.prev_vel = None
        self.prev_des_pos = None
        self._a_lp = np.zeros(3, dtype=np.float32)

        # adaptive state
        self.dhat = [0.0] * 6
        self.jifen = [0.0] * 3
        self.gains = AdaptiveGains()

        # logs
        self.log_t = []
        self.log_pos = []
        self.log_posd = []
        self.log_vel = []
        self.log_cmd = []   # thr, roll, pitch, yaw
        self.log_fhat = []
        self.log_fa = []
        self.log_dhat = []

    def _update_hist(self, pos, vel, att, ang, posd, attd, maxlen=10):
        for i in range(3):
            self.pos_hist[i].append(float(pos[i]))
            self.vel_hist[i].append(float(vel[i]))
            self.att_hist[i].append(float(att[i]))
            self.ang_hist[i].append(float(ang[i]))
            self.posd_hist[i].append(float(posd[i]))
            self.attd_hist[i].append(float(attd[i]))
        for buf in (self.pos_hist, self.vel_hist, self.att_hist, self.ang_hist, self.posd_hist, self.attd_hist):
            for i in range(3):
                if len(buf[i]) > maxlen:
                    buf[i] = buf[i][-maxlen:]

    def _build_features(
        self,
        *,
        obs,
        des_pos: Sequence[float],
        des_att: Sequence[float],
        v_d: Sequence[float],
        q_sp: Sequence[float],
        thr: float,
    ) -> np.ndarray:
        parts = []
        for key in self.feature_keys:
            if key == "p":
                parts.append(obs.pos_ned.astype(np.float32))
            elif key == "v":
                parts.append(obs.vel_ned.astype(np.float32))
            elif key == "q":
                qw, qx, qy, qz = obs.quat_wxyz
                parts.append(np.asarray([qw, qx, qy, qz], dtype=np.float32))
            elif key == "w":
                parts.append(obs.ang_vel_ned.astype(np.float32))
            elif key == "R":
                parts.append(obs.R_wb_ned.astype(np.float32).reshape(-1))
            elif key == "p_d":
                parts.append(np.asarray(des_pos, dtype=np.float32))
            elif key == "att":
                # roll,pitch,yaw
                qw, qx, qy, qz = obs.quat_wxyz
                r, p, y = quaternion_to_euler(qx, qy, qz, qw)
                parts.append(np.asarray([r, p, y], dtype=np.float32))
            elif key == "att_d":
                parts.append(np.asarray(des_att, dtype=np.float32))
            elif key == "v_d":
                parts.append(np.asarray(v_d, dtype=np.float32))
            elif key == "q_sp":
                parts.append(np.asarray(q_sp, dtype=np.float32))
            elif key == "T_sp":
                parts.append(np.asarray([float(thr)], dtype=np.float32))
            elif key == "pwm":
                if obs.rotor_speeds is None:
                    parts.append(np.zeros(4, dtype=np.float32))
                else:
                    parts.append(obs.rotor_speeds.astype(np.float32))
            else:
                parts.append(np.zeros(1, dtype=np.float32))

        return np.concatenate(parts, axis=0).astype(np.float32)

    def _select_traj(self) -> Callable[[float], Tuple[float, float, float, float]]:
        if self.cfg.traj == "fig8":
            return traj_fig8
        if self.cfg.traj == "circle":
            return traj_circle
        if self.cfg.traj == "ellipse":
            return traj_ellipse
        if self.cfg.traj == "random":
            return traj_random_spline(total_time=float(self.cfg.total_time), seed=int(self.cfg.random_seed))
        return traj_fig8

    def _save_csv(self, out_dir: Path, base: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{base}.csv"
        with open(p, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "pos", "pos_d", "vel", "cmd(thr,r,p,y)", "fhat", "fa_obs", "dhat_xyz"])
            for i in range(len(self.log_t)):
                wr.writerow([
                    f"{self.log_t[i]:.6f}",
                    str(self.log_pos[i]),
                    str(self.log_posd[i]),
                    str(self.log_vel[i]),
                    str(self.log_cmd[i]),
                    str(self.log_fhat[i]),
                    str(self.log_fa[i]),
                    str(self.log_dhat[i]),
                ])
        print(f"[metapinn_runner] saved csv: {p}")

    def run(self):
        # ---- warm-start load ----
        dhat_init = None
        ok, dh = self.online.load_warm_start(self.cfg.wind_condition_name)
        if ok:
            dhat_init = dh
            if dhat_init is not None and dhat_init.size >= 3:
                self.dhat[0] = float(dhat_init[0])
                self.dhat[1] = float(dhat_init[1])
                self.dhat[2] = float(dhat_init[2])
            print(f"[metapinn_runner] warm-start loaded for {self.cfg.wind_condition_name}: dhat_init={dhat_init}")

        # ---- connect + takeoff ----
        self.env.enable()
        self.env.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)
        self.env.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)

        target_z = -10.0  # will follow traj z anyway; takeoff to a safe height
        self.env.takeoff(target_z_ned=target_z, timeout=10.0, vel=2.0)

        traj_fn = self._select_traj()

        # wind
        profile = PROFILES_ALL.get(self.cfg.wind_profile, None)
        if profile is None:
            raise KeyError(f"Unknown wind_profile '{self.cfg.wind_profile}'. Available: {list(PROFILES_ALL.keys())}")
        print(f"[metapinn_runner] wind: {self.cfg.wind_profile} | condition={self.cfg.wind_condition_name}")

        t_sim = 0.0
        dt = float(self.cfg.dt)
        hover_thr = float(self.cfg.hover_throttle)

        try:
            while t_sim < float(self.cfg.total_time):
                t_wall = time.time()

                # apply wind
                self.wind.apply_wind(self.env.client, profile, t_sim, dt=max(0.01, dt))

                # read obs (no depth needed here)
                obs = self.env.get_obs(depth_z_m=None)

                # current attitude (roll,pitch,yaw)
                qw, qx, qy, qz = obs.quat_wxyz
                r_cur, p_cur, y_cur = quaternion_to_euler(qx, qy, qz, qw)
                att_cur = [r_cur, p_cur, y_cur]

                # desired state from trajectory
                xd, yd, zd, yawd = traj_fn(t_sim)
                des_pos = [xd, yd, zd]
                des_att = [0.0, 0.0, yawd]

                # desired vel (finite diff)
                if self.prev_des_pos is None:
                    v_d = [0.0, 0.0, 0.0]
                else:
                    v_d = [(des_pos[i] - self.prev_des_pos[i]) / dt for i in range(3)]

                # update history buffers
                self._update_hist(obs.pos_ned, obs.vel_ned, att_cur, obs.ang_vel_ned, des_pos, des_att)

                # only start control after enough history
                if len(self.pos_hist[0]) >= 3:
                    # feature vector
                    q_sp = euler_to_quat(0.0, -0.0, yawd)
                    feat = self._build_features(obs=obs, des_pos=des_pos, des_att=des_att, v_d=v_d, q_sp=q_sp, thr=0.5)

                    # predict residual force
                    fhat = None
                    if bool(self.cfg.enable_meta):
                        speed = float(np.linalg.norm(obs.vel_ned))
                        fhat = self.wrapper.predict(feat, cond_val=speed)  # N in world(NED)

                    # freeze phase for dhat update
                    dt_eff = 0.0 if (t_sim < float(self.cfg.freeze_T)) else dt

                    # adaptive controller (feed fhat inside)
                    thr, r_des, p_des, y_des, self.dhat, self.jifen, _ = adaptive_controller(
                        self.gains,
                        self.pos_hist, self.vel_hist, self.att_hist, self.ang_hist,
                        self.posd_hist, self.attd_hist,
                        self.dhat, self.jifen,
                        dt_eff, t_sim,
                        fhat=fhat,
                    )

                    # optional vertical feedforward using fhat projected to -b_z
                    if bool(self.cfg.enable_meta) and (fhat is not None):
                        g_ff = float(self.cfg.feedforward_g)
                        b_z = obs.R_wb_ned[:, 2].astype(np.float32)  # body z axis in world(NED)
                        thrust_add = g_ff * float(np.dot(fhat.astype(np.float32), -b_z))  # add along thrust axis
                        thrust_force = (thr / hover_thr) * float(UAV_mass) * G + thrust_add
                        thr = clip((thrust_force / (float(UAV_mass) * G)) * 0.5 + 0.5, 0.0, 1.0)

                    # send command (SimpleFlight)
                    self.env.client.moveByRollPitchYawThrottleAsync(
                        float(r_des), float(p_des), float(y_des), float(thr),
                        duration=dt, vehicle_name=self.cfg.vehicle_name
                    )

                    # build fa_obs for online learning (same spirit as your original script)
                    fa_obs = None
                    if bool(self.cfg.enable_meta) and bool(self.cfg.enable_online_learn) and (self.prev_vel is not None):
                        a_raw = (obs.vel_ned.astype(np.float32) - self.prev_vel.astype(np.float32)) / dt
                        alpha = float(np.clip(2.0 * math.pi * 4.0 * dt, 0.0, 1.0))  # ~4Hz LP
                        self._a_lp = (1 - alpha) * self._a_lp + alpha * a_raw
                        a_world = self._a_lp

                        b_z = obs.R_wb_ned[:, 2].astype(np.float32)
                        thrust_world = -(thr / hover_thr) * float(UAV_mass) * G * b_z  # NED
                        fa_obs = float(UAV_mass) * a_world - thrust_world - np.array([0.0, 0.0, float(UAV_mass) * G], dtype=np.float32)

                        # push + maybe update
                        loss = self.online.step(
                            t_sim,
                            feat_np=feat,
                            target_fa_np=fa_obs.astype(np.float32),
                            cur_vel=obs.vel_ned.tolist(),
                            att=att_cur,
                            thr=float(thr),
                            dhat_xyz=self.dhat[:3],
                        )
                        if loss is not None:
                            print(f"[online_update] loss={loss:.5f}")

                    # logs
                    self.log_t.append(t_sim)
                    self.log_pos.append(obs.pos_ned.astype(float).tolist())
                    self.log_posd.append([float(xd), float(yd), float(zd)])
                    self.log_vel.append(obs.vel_ned.astype(float).tolist())
                    self.log_cmd.append([float(thr), float(r_des), float(p_des), float(y_des)])
                    self.log_fhat.append((None if fhat is None else fhat.astype(float).tolist()))
                    self.log_fa.append((None if fa_obs is None else fa_obs.astype(float).tolist()))
                    self.log_dhat.append([float(self.dhat[0]), float(self.dhat[1]), float(self.dhat[2])])

                self.prev_vel = obs.vel_ned.copy()
                self.prev_des_pos = des_pos[:]

                t_sim += dt
                # real-time pacing
                time.sleep(max(0.0, dt - (time.time() - t_wall)))

        finally:
            # land + disable
            try:
                self.env.client.landAsync(vehicle_name=self.cfg.vehicle_name).join()
            except Exception:
                pass
            try:
                self.env.client.armDisarm(False, vehicle_name=self.cfg.vehicle_name)
                self.env.client.enableApiControl(False, vehicle_name=self.cfg.vehicle_name)
            except Exception:
                pass

            # save warm-start (task emb/beta + dhat_steady)
            dhat_steady = self.online.estimate_dhat_steady(now_t=(self.log_t[-1] if self.log_t else None))
            self.online.save_warm_start(
                condition=self.cfg.wind_condition_name,
                extra=None,
                dhat_steady=dhat_steady,
            )
            print(f"[metapinn_runner] warm-start saved: condition={self.cfg.wind_condition_name} dhat_steady={dhat_steady}")

            # save csv
            out_dir = Path(self.cfg.out_dir)
            base = f"simpleflight_{self.cfg.traj}_metapinn_{self.cfg.wind_condition_name}"
            if bool(self.cfg.save_csv):
                self._save_csv(out_dir, base)

            self.env.disable()


# ----------------------------
# CLI
# ----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="path to metapinn yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    MetaPINNRunner(cfg).run()
