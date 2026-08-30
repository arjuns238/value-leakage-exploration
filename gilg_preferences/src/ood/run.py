"""Run OOD pairwise measurement experiments.

Usage:
    python -m src.ood.run configs/ood/category_preference.yaml
    python -m src.ood.run configs/ood/*.yaml
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("VLLM_API_KEY", "dummy")

from src.ood.config import load_ood_config
from src.ood.measurement import run_config


def main():
    parser = argparse.ArgumentParser(description="Run OOD measurement experiments")
    parser.add_argument("configs", nargs="+", type=Path, help="Config file(s) to run")
    args = parser.parse_args()

    for path in args.configs:
        config = load_ood_config(path)
        run_config(config)

    print("\nAll done.")


if __name__ == "__main__":
    main()
