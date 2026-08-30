"""Run noise baselines for probe benchmarking.

Usage:
    python -m src.probes.run_baselines --config configs/probes/example.yaml [--n-seeds 10] [--output-dir dir]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.probes.baselines import run_noise_baselines, aggregate_noise_baselines
from src.probes.experiments.run_dir_probes import RunDirProbeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noise baselines for probe benchmarking")
    parser.add_argument("--config", type=Path, required=True, help="RunDirProbeConfig YAML path")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of random seeds per baseline")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: config.output_dir/baselines)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunDirProbeConfig.from_yaml(args.config)
    output_dir = args.output_dir or config.output_dir / "baselines"

    print("Noise Baselines")
    print(f"Config: {args.config}")
    print(f"Run dir: {config.run_dir}")
    if config.eval_run_dir:
        print(f"Eval run dir: {config.eval_run_dir}")
    print(f"Layers: {config.layers}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {output_dir}")
    print()

    results = run_noise_baselines(config, n_seeds=args.n_seeds)
    print(f"\nCollected {len(results)} baseline results")

    aggregated = aggregate_noise_baselines(results)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": config.experiment_name,
        "config_path": str(args.config),
        "run_dir": str(config.run_dir),
        "eval_run_dir": str(config.eval_run_dir) if config.eval_run_dir else None,
        "created_at": datetime.now().isoformat(),
        "n_seeds": args.n_seeds,
        "standardize": config.standardize,
        "demean_confounds": config.demean_confounds,
        "baselines": {
            "shuffled_labels": [
                r for r in aggregated if r["baseline_type"] == "shuffled_labels"
            ],
            "random_activations": [
                r for r in aggregated if r["baseline_type"] == "random_activations"
            ],
        },
        "raw_results": [r.to_dict() for r in results],
    }

    manifest_path = output_dir / "baselines_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path}")

    # Print summary
    print("\n=== Baseline Summary ===")
    heldout = any("heldout_r_mean" in r for r in aggregated)

    for baseline_type in ["shuffled_labels", "random_activations"]:
        entries = manifest["baselines"][baseline_type]
        if not entries:
            print(f"\n{baseline_type}: no results")
            continue

        print(f"\n{baseline_type}:")
        for entry in entries:
            if heldout:
                r_str = f"r={entry['heldout_r_mean']:.4f} (±{entry['heldout_r_std']:.4f})"
                acc_str = ""
                if "heldout_acc_mean" in entry:
                    acc_str = f", acc={entry['heldout_acc_mean']:.4f} (±{entry['heldout_acc_std']:.4f})"
                print(f"  Layer {entry['layer']}: {r_str}{acc_str}")
            else:
                print(f"  Layer {entry['layer']}: R²={entry['cv_r2_mean']:.4f} "
                      f"(±{entry['cv_r2_std']:.4f}, n_seeds={entry['n_seeds']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
