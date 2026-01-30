# get_obs(), send_cmd(), takeoff(), land()
# envs/airsim_multirotor_env.py
from __future__ import annotations
import time
import numpy as np
import airsim
from typing import Optional
from envs.airsim_client import AirSimClient
from controllers.interfaces import Obs, Cmd

def quat_to_R_wb_ned(qw, qx, qy, qz) -> np.ndarray:
    # body->world rotation in NED (same as your previous helper)
    n = float(np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float32)
    return R

class AirSimMultirotorEnv:
    def __init__(self, airsim_ip="127.0.0.1", vehicle_name="Drone1"):
        self.vehicle_name = vehicle_name
        self.asim = AirSimClient(airsim_ip)
        self.client = self.asim.client
        self._t0 = time.time()

    def enable(self):
        self.asim.enable_api(self.vehicle_name)

    def disable(self):
        self.asim.disable_api(self.vehicle_name)

    def takeoff(self, target_z_ned: float, timeout=8.0, vel=2.0):
        self.enable()
        self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.vehicle_name).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.3)
        st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        x0 = st.kinematics_estimated.position.x_val
        y0 = st.kinematics_estimated.position.y_val
        self.client.moveToPositionAsync(x0, y0, target_z_ned, velocity=vel, vehicle_name=self.vehicle_name).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)

    def land(self):
        try:
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass

    def get_rotor_speeds(self) -> Optional[np.ndarray]:
        try:
            rs = self.client.getRotorStates(vehicle_name=self.vehicle_name)
            spds = [rs.rotors[i]['speed'] for i in range(4)]
            return np.asarray(spds, dtype=np.float32)
        except Exception:
            return None
        
    def get_depth_z_m(
        self,
        image_height: int,
        image_width: int,
        depth_max_m: float,
        camera_name: str = "0",
    ) -> Optional[np.ndarray]:
        """
        Return (H,W) depth in meters (DepthPerspective, image_data_float).
        Robust: reshape, resize, NaN handling.
        """
        try:
            import cv2
        except Exception:
            cv2 = None

        try:
            responses = self.client.simGetImages(
                [airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)],
                vehicle_name=self.vehicle_name,
            )
            if (not responses) or (len(responses) == 0) or (len(responses[0].image_data_float) == 0):
                return None

            r = responses[0]
            d = np.asarray(r.image_data_float, dtype=np.float32)

            # expected size = r.height * r.width
            if d.ndim != 1 or d.size != int(r.height * r.width):
                # fallback: try reshape anyway
                try:
                    d = d.reshape(int(r.height), int(r.width))
                except Exception:
                    return None
            else:
                d = d.reshape(int(r.height), int(r.width))

            # resize to planner input size
            H, W = int(image_height), int(image_width)
            if (d.shape[0] != H) or (d.shape[1] != W):
                if cv2 is not None:
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    # no cv2: very simple nearest resize
                    ys = (np.linspace(0, d.shape[0]-1, H)).astype(np.int32)
                    xs = (np.linspace(0, d.shape[1]-1, W)).astype(np.int32)
                    d = d[ys][:, xs]

            # NaN/Inf -> depth_max_m
            dm = float(depth_max_m)
            bad = ~np.isfinite(d)
            if np.any(bad):
                d[bad] = dm

            # clip
            d = np.clip(d, 0.0, dm).astype(np.float32)
            return d

        except Exception:
            return None


    def get_obs(self, depth_z_m: Optional[np.ndarray] = None) -> Obs:
        st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = st.kinematics_estimated

        p = kin.position
        v = kin.linear_velocity
        a = kin.linear_acceleration
        w = kin.angular_velocity
        q = kin.orientation

        pos = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)
        vel = np.array([v.x_val, v.y_val, v.z_val], dtype=np.float32)
        acc = np.array([a.x_val, a.y_val, a.z_val], dtype=np.float32)
        ang = np.array([w.x_val, w.y_val, w.z_val], dtype=np.float32)
        quat = (q.w_val, q.x_val, q.y_val, q.z_val)

        R_wb = quat_to_R_wb_ned(*quat)
        rotor = self.get_rotor_speeds()

        return Obs(
            t=time.time() - self._t0,
            pos_ned=pos, vel_ned=vel, acc_ned=acc,
            R_wb_ned=R_wb, quat_wxyz=quat, ang_vel_ned=ang,
            depth_z_m=depth_z_m,
            rotor_speeds=rotor,
            extra={}
        )

    def send_cmd(self, cmd: Cmd, duration: float):
        # E2E 第一版：速度接口
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=float(cmd.yaw_deg))
        self.client.moveByVelocityZAsync(
            float(cmd.vx), float(cmd.vy), float(cmd.z),
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name
        )
