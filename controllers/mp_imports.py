# controllers/metapinn/mp_imports.py
from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_metapinn_on_path(mp_dir: str | None = None) -> str:
    """
    Ensure Meta-PINN repo (containing meta_pinn/) is on sys.path.

    Priority:
      1) explicit mp_dir
      2) env var META_PINN_DIR
      3) default: <project_root>/Meta-PINN
    """
    if mp_dir is None or str(mp_dir).strip() == "":
        mp_dir = os.environ.get("META_PINN_DIR", "")

    if mp_dir.strip() == "":
        project_root = Path(__file__).resolve().parents[2]
        mp_dir = str(project_root / "Meta-PINN")

    mp_path = Path(mp_dir).resolve()
    if not mp_path.exists():
        raise FileNotFoundError(
            f"Meta-PINN dir not found: {mp_path}\n"
            f"Set configs.metapinn.metapinn_dir or env META_PINN_DIR."
        )

    mp_str = str(mp_path)
    if mp_str not in sys.path:
        sys.path.insert(0, mp_str)

    return mp_str
