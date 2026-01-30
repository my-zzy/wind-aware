# planners/yopo/yopo_model.py
from __future__ import annotations
import os, sys
import torch

class YOPOModel:
    def __init__(self, yopo_dir: str, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.isdir(yopo_dir) and yopo_dir not in sys.path:
            sys.path.insert(0, yopo_dir)

        from policy.yopo_network import YopoNetwork  # type: ignore

        self.model = YopoNetwork().to(self.device)
        self._load(checkpoint_path)
        self.model.eval()

    def _load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"YOPO ckpt not found: {path}")
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
        except TypeError:
            sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.model.load_state_dict(sd, strict=True)

    @torch.inference_mode()
    def inference(self, depth_t: torch.Tensor, obs_t: torch.Tensor):
        endstate_t, score_t = self.model.inference(depth_t, obs_t)
        return endstate_t, score_t
