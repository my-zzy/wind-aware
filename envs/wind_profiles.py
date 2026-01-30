 # PROFILES_ALL + apply_wind(profile,t)
# envs/wind_profiles.py
from __future__ import annotations
import numpy as np
import airsim

PROFILES_ALL = {
    "0mps":  {"tag":"0mps",  "kind":"const", "dir":(0,1,0), "mag":0.0},
    "5mps":  {"tag":"5mps",  "kind":"const", "dir":(0,1,0), "mag":5.0},
    "10mps": {"tag":"10mps", "kind":"const", "dir":(0,1,0), "mag":10.0},
    "12mps": {"tag":"12mps", "kind":"const", "dir":(0,1,0), "mag":12.0},
    "15mps": {"tag":"15mps", "kind":"const", "dir":(0,1,0), "mag":15.0},
    "sinusoidal_0to10mps": {"tag":"sinusoidal_0to10mps","kind":"sin","dir":(0,1,0),"mag_mean":5.0,"mag_amp":5.0,"freq_hz":0.33},
    "ou15": {"tag":"ou15", "kind":"ou3d","mean":(0,15,0), "sigma":(1.5,1.5,0.5), "tau":(2.0,2.0,3.0)},
}

class WindApplier:
    def __init__(self):
        self._ou_state = None

    def apply_wind(self, client, profile: dict, t: float, dt: float = 0.02):
        kind = profile.get("kind", "const")

        if kind == "const":
            mag = float(profile.get("mag", 0.0))
            d = profile.get("dir", (0,1,0))
            wind = airsim.Vector3r(d[0]*mag, d[1]*mag, d[2]*mag)
            client.simSetWind(wind)
            return

        if kind == "sin":
            mag = float(profile["mag_mean"] + profile["mag_amp"] * np.sin(2*np.pi*profile["freq_hz"]*t))
            d = profile["dir"]
            wind = airsim.Vector3r(d[0]*mag, d[1]*mag, d[2]*mag)
            client.simSetWind(wind)
            return

        if kind == "ou3d":
            if self._ou_state is None:
                self._ou_state = np.array(profile["mean"], dtype=float)
            mu  = np.array(profile["mean"], dtype=float)
            tau = np.array(profile["tau"], dtype=float)
            sig = np.array(profile["sigma"], dtype=float)
            dW = np.random.normal(size=3)
            self._ou_state += (mu - self._ou_state) * (dt / tau) + np.sqrt(2.0*dt / tau) * sig * dW
            X,Y,Z = self._ou_state.tolist()
            client.simSetWind(airsim.Vector3r(X,Y,Z))
            return
