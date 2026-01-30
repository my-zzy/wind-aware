# scripts/run_yopo.py
import argparse
from runners.yopo_runner import YopoRunner, build_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="./configs/yopo_airsim.yaml")
    args = ap.parse_args()
    cfg = build_cfg(args.cfg)
    YopoRunner(cfg).run()

if __name__ == "__main__":
    main()
