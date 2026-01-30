# 连接、API control、通用封装
# envs/airsim_client.py
from __future__ import annotations
import time
import airsim

class AirSimClient:
    def __init__(self, airsim_ip="127.0.0.1"):
        self.client = airsim.MultirotorClient(ip=airsim_ip)
        self.client.confirmConnection()

    def reset(self, vehicle_name="Drone1"):
        try:
            self.client.reset()
            time.sleep(1.5)
        except Exception:
            pass

    def enable_api(self, vehicle_name="Drone1"):
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)

    def disable_api(self, vehicle_name="Drone1"):
        try: self.client.hoverAsync(vehicle_name=vehicle_name).join()
        except Exception: pass
        try: self.client.armDisarm(False, vehicle_name=vehicle_name)
        except Exception: pass
        try: self.client.enableApiControl(False, vehicle_name=vehicle_name)
        except Exception: pass
