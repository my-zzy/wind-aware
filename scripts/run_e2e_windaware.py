from __future__ import annotations

import argparse

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML not found. Please `pip install pyyaml`.") from e

from runners.e2e_windaware_runner import E2EWindAwareRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to yaml config")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a dict.")

    # minimal banner (optional)
    print(f"[run_e2e_windaware] cfg={args.cfg}")

    runner = E2EWindAwareRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()