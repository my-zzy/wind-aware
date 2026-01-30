# planners/yopo/yopo_imports.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def ensure_yopo_on_path(yopo_dir: Optional[str] = None) -> str:
    """
    Make sure the YOPO repo root is on sys.path so that:
      import policy.xxx
      import config.xxx
    can work.

    Priority:
      1) explicit yopo_dir
      2) env var YOPO_DIR
      3) default: <project_root>/YOPO
    """
    if yopo_dir is None or str(yopo_dir).strip() == "":
        yopo_dir = os.environ.get("YOPO_DIR", "")

    if yopo_dir.strip() == "":
        # project_root = .../WIND-AWARE-MAIN
        project_root = Path(__file__).resolve().parents[2]
        yopo_dir = str(project_root / "YOPO")

    yopo_path = Path(yopo_dir).resolve()
    if not yopo_path.exists():
        raise FileNotFoundError(
            f"YOPO dir not found: {yopo_path}\n"
            f"Set configs.yopo.yopo_dir or env YOPO_DIR to your YOPO repo folder."
        )

    yopo_str = str(yopo_path)
    if yopo_str not in sys.path:
        sys.path.insert(0, yopo_str)

    return yopo_str
