# perception/roi_depth.py
from __future__ import annotations
import numpy as np

def quantile_min(depth: np.ndarray, q: float, default: float) -> float:
    v = depth[np.isfinite(depth)]
    if v.size == 0: return float(default)
    return float(np.quantile(v, q))

def compute_min_depth_roi(depth_m: np.ndarray, default: float,
                          x0=0.35, x1=0.65, y0=0.20, y1=0.80) -> float:
    h, w = depth_m.shape[:2]
    ix0, ix1 = int(w*x0), int(w*x1)
    iy0, iy1 = int(h*y0), int(h*y1)
    roi = depth_m[iy0:iy1, ix0:ix1]
    if roi.size == 0:
        roi = depth_m
    valid = np.isfinite(roi) & (roi > 0.2)
    vals = roi[valid]
    if vals.size == 0:
        return float(default)
    return float(np.quantile(vals, 0.05))

def split_lcr_quantiles(depth_m: np.ndarray, default: float):
    h, w = depth_m.shape
    L = depth_m[:, :int(0.33*w)]
    C = depth_m[:, int(0.40*w):int(0.60*w)]
    R = depth_m[:, int(0.67*w):]
    return (
        quantile_min(L, 0.03, default),
        quantile_min(C, 0.03, default),
        quantile_min(R, 0.03, default),
        quantile_min(depth_m, 0.01, default),
    )
