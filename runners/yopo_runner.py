#runners/yopo_runner.py
from __future__ import annotations
import time, yaml
import numpy as np

from envs.airsim_multirotor_env import AirSimMultirotorEnv
from envs.wind_profiles import PROFILES_ALL, WindApplier
from perception.depth_sensor import DepthSensor
from planners.yopo.yopo_model import YOPOModel
from planners.yopo.yopo_planner import YOPOPlanner

from controllers.windaware_mixer import WindAwareMixer
from controllers.safety_shield import SafetyShield
from controllers.yaw_controller import YawController
from controllers.interfaces import Cmd

from visualization.yopo_live_view import YopoLiveView
from visualization.flight_logger import FlightLogger

from controllers.feature_builder import build_features
from controllers.wrapper import MetaPINNWrapper

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def merge_cfg(base: dict, override: dict) -> dict:
    out = dict(base)
    out.update(override)
    return out

class YopoRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg["control_dt"])
        self.cmd_dur = float(cfg["command_duration_s"])
        self.target_z = float(cfg["target_z_ned"])

        self.env = AirSimMultirotorEnv(cfg["airsim_ip"], cfg["vehicle_name"])
        self.depth = DepthSensor(
            self.env.client,
            vehicle_name=cfg["vehicle_name"],
            camera_name=cfg.get("camera_name", "0"),
            image_height=cfg["image_height"],
            image_width=cfg["image_width"],
            depth_max_m=cfg["depth_max_m"],
            h_fov_deg=cfg.get("h_fov_deg", 90.0),
            v_fov_deg=cfg.get("v_fov_deg", 60.0),
            add_noise=True
        )

        yopo = YOPOModel(cfg["yopo_dir"], cfg["checkpoint_path"])
        self.planner = YOPOPlanner(
            yopo, cfg["image_height"], cfg["image_width"], cfg["depth_max_m"],
            cfg["segment_time"],
            control_dt=cfg["control_dt"], 
            max_vel_xy=cfg["max_vel_xy"],
            max_vel_norm=cfg["max_vel_norm"],
            subgoal_radius=cfg["subgoal_radius"],
            goal_length=cfg["yopo_goal_length"],
            target_z_ned=self.target_z,
            lookahead_s=float(cfg.get("trajectory_lookahead_s", 0.20)),
        )

        self.use_meta = bool(cfg.get("use_meta", False))
        self.meta = None
        self.meta_keys = []
        self.mixer = WindAwareMixer(max_vel_xy=cfg["max_vel_xy"], k_ff=float(cfg.get("meta_k_ff", 0.5)))

        if self.use_meta:
            self.meta_keys = [s.strip() for s in cfg.get("meta_features","v,q,pwm").split(",") if s.strip()]
            self.meta = MetaPINNWrapper(
                feature_keys=self.meta_keys,
                scaler_path=cfg.get("mp_scaler"),
                load_path=cfg.get("mp_load"),
                lr=1e-3, update_every=int(cfg.get("meta_update_every", 10))
            )

        self.yaw_ctl = YawController(
            mode=str(cfg.get("yaw_mode", "face_velocity")),
            alpha=float(cfg.get("yaw_alpha", 0.25)),
            rate_limit_deg_s=float(cfg.get("yaw_rate_limit_deg_s", 90.0))
        )

        # self.shield = SafetyShield(depth_max_m=cfg["depth_max_m"],
        #                            enable=bool(cfg.get("enable_safety_shield", True)))
        from controllers.safety_shield import SafetyShield, SafetyShieldConfig

        shield_cfg = SafetyShieldConfig(
            enable=bool(cfg.get("enable_safety_shield", True)),
            max_vel_xy=float(cfg["max_vel_xy"]),
            target_z_ned=float(cfg["target_z_ned"]),

            safety_slow_depth_m=float(cfg.get("safety_slow_depth_m", 2.8)),
            safety_stop_depth_m=float(cfg.get("safety_stop_depth_m", 1.2)),
            safety_climb_m=float(cfg.get("safety_climb_m", 0.8)),

            shield_alpha=float(cfg.get("shield_alpha", 0.85)),
            shield_hold_steps=int(cfg.get("shield_hold_steps", 12)),
            shield_dir_gap_m=float(cfg.get("shield_dir_gap_m", 0.6)),
            shield_emergency_vy=float(cfg.get("shield_emergency_vy", 1.2)),

            shield_repulse_ref_m=float(cfg.get("shield_repulse_ref_m", 3.0)),
            shield_k_rep=float(cfg.get("shield_k_rep", 1.0)),
            shield_vy_cap=float(cfg.get("shield_vy_cap", 1.0)),

            shield_min_scale=float(cfg.get("shield_min_scale", 0.30)),
            shield_slow_d=float(cfg.get("shield_slow_d", 2.8)),
            shield_stop_d=float(cfg.get("shield_stop_d", 1.2)),

            shield_corridor_vy_mul=float(cfg.get("shield_corridor_vy_mul", 0.35)),
            shield_corridor_vx_mul=float(cfg.get("shield_corridor_vx_mul", 0.55)),
        )

        self.shield = SafetyShield(shield_cfg)


        self.viz = YopoLiveView(enabled=bool(cfg.get("vis_depth", True)), depth_max_m=cfg["depth_max_m"])
        self.logger = FlightLogger(out_dir=str(cfg.get("log_dir", "./yopo_logs")))

        self.goal = np.array(cfg["goal"], dtype=np.float32)
        self.goal_reached = float(cfg.get("goal_reached_m", 2.0))

        self.apply_wind = bool(cfg.get("apply_wind", False))
        self.wind_tag = cfg.get("wind_profile_tag", "0mps")
        self.wind_profile = PROFILES_ALL.get(self.wind_tag, PROFILES_ALL["0mps"])
        self.wind = WindApplier()

    def run(self):
        self.env.takeoff(self.target_z)

        try:
            while True:
                t0 = time.time()

                depth_z = self.depth.get_depth_z()
                obs = self.env.get_obs(depth_z_m=depth_z)

                if self.apply_wind:
                    self.wind.apply_wind(self.env.client, self.wind_profile, obs.t, dt=self.dt)

                dist = float(np.linalg.norm(self.goal - obs.pos_ned))
                if dist < self.goal_reached:
                    print(f"[goal] reached dist={dist:.2f}m")
                    break

                plan = self.planner.plan(obs, self.goal)

                # metapinn fhat
                fhat = None
                if self.use_meta and (self.meta is not None):
                    # cmd_vxy 还没生成，先用 v_ref 作为 proxy（更稳）
                    feat = build_features(self.meta_keys, obs, plan, cmd_vxy=plan.v_ref_ned[:2])
                    speed = float(np.linalg.norm(obs.vel_ned))
                    fhat = self.meta.predict(feat, cond_val=speed)

                cmd = self.mixer.mix(obs, plan, fhat_ned=fhat, dt=self.dt)

                # yaw smoothing
                yaw = self.yaw_ctl.step(np.array([cmd.vx, cmd.vy], dtype=np.float32), cmd.yaw_deg, dt=self.dt)
                cmd = Cmd(vx=cmd.vx, vy=cmd.vy, z=cmd.z, yaw_deg=yaw, meta=cmd.meta)

                # shield
                cmd = self.shield.apply(obs, cmd)

                # send
                self.env.send_cmd(cmd, duration=self.cmd_dur)

                # viz + log
                self.viz.render(obs, plan, cmd)
                self.logger.push(obs, plan, cmd)

                sleep = self.dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        finally:
            self.logger.save(name="logs")
            try: self.env.land()
            except Exception: pass
            self.env.disable()

def build_cfg(cfg_path: str) -> dict:
    cfg = load_yaml(cfg_path)
    if "base" in cfg:
        base = load_yaml(cfg["base"])
        cfg = merge_cfg(base, cfg)
    return cfg
