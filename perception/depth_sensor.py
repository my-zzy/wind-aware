# perception/depth_sensor.py
from __future__ import annotations
import numpy as np
import cv2
import airsim

class DepthSensor:
    def __init__(self, client, vehicle_name="Drone1", camera_name="0",
                 image_height=96, image_width=160,
                 depth_max_m=20.0, h_fov_deg=90.0, v_fov_deg=60.0,
                 add_noise=True):
        self.client = client
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.H = int(image_height)
        self.W = int(image_width)
        self.maxd = float(depth_max_m)
        self.hfov = np.deg2rad(float(h_fov_deg))
        self.vfov = np.deg2rad(float(v_fov_deg))
        self.add_noise = bool(add_noise)

    def get_depth_z(self):
        resp = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)],
            vehicle_name=self.vehicle_name
        )
        if not resp or len(resp[0].image_data_float) == 0:
            return None

        d = resp[0]
        depth_e = np.array(d.image_data_float, dtype=np.float32).reshape(d.height, d.width)
        depth_e = cv2.resize(depth_e, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_e[np.isnan(depth_e)] = np.nanmax(depth_e)

        xs = (np.linspace(-1, 1, self.W)[None, :]) * np.tan(self.hfov/2)
        ys = (np.linspace(-1, 1, self.H)[:, None]) * np.tan(self.vfov/2)
        cos_theta = 1.0 / np.sqrt(xs**2 + ys**2 + 1.0)
        depth_z = depth_e * cos_theta

        if self.add_noise:
            depth_z = depth_z + np.random.normal(0, 0.03, depth_z.shape).astype(np.float32)
            depth_z = cv2.GaussianBlur(depth_z, (3,3), 0)

        depth_z = np.clip(depth_z, 0.0, self.maxd)
        return depth_z
