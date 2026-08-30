"""Plot consistency vs mean rating scatter.

Usage:
    python -m src.task_data.consistency.plot --model gemma3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.measurement.storage import EXPERIMENTS_DIR
from src.measurement.storage.run_parsing import (
    extract_model_from_run_dir,
    normalize_score,
    parse_scale_tag,
)
from src.task_data.consistency.compute import load_consistency_index

OUTPUT_DIR = Path(__file__).parent / "analysis"


def load_mean_ratings(
    experiment_id: str, model_filter: str
) -> dict[str, tuple[float, str]]:
    """Load mean ratings per task. Returns task_id -> (mean_rating, origin)."""
    exp_dir = EXPERIMENTS_DIR / experiment_id
    stated_dir = exp_dir / "post_task_stated"
    if not stated_dir.exists():
        return {}

    task_ratings: dict[str, list[float]] = defaultdict(list)
    task_origins: dict[str, str] = {}

    for run_dir in stated_dir.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.yaml"
        measurements_path = run_dir / "measurements.yaml"
        if not config_path.exists() or not measurements_path.exists():
            continue

        model = extract_model_from_run_dir(run_dir.name)
        if model != model_filter:
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        scale_tag = config.get("template_tags", {}).get("scale")
        if not scale_tag:
            continue
        scale = parse_scale_tag(scale_tag)
        if not scale:
            continue

        with open(measurements_path) as f:
            measurements = yaml.safe_load(f)

        if not measurements:
            continue

        for m in measurements:
            if "score" not in m or not isinstance(m["score"], (int, float)):
                continue
            task_id = m["task_id"]
            # Normalize score to 0-1 for comparability across templates
            norm_score = normalize_score(float(m["score"]), scale)
            task_ratings[task_id].append(norm_score)
            if task_id not in task_origins:
                task_origins[task_id] = m.get("origin", "UNKNOWN")

    results: dict[str, tuple[float, str]] = {}
    for task_id, ratings in task_ratings.items():
        results[task_id] = (float(np.mean(ratings)), task_origins[task_id])

    return results


def plot_consistency_vs_mean(
    consistency_key: str,
    experiment_id: str,
    model: str,
    output_path: Path,
) -> None:
    index = load_consistency_index(consistency_key)
    mean_ratings = load_mean_ratings(experiment_id, model)

    if not mean_ratings:
        print(f"No ratings found for {model} in {experiment_id}")
        return

    # Merge data
    origins_colors = {
        "WILDCHAT": "C0",
        "ALPACA": "C2",
        "MATH": "C1",
        "BAILBENCH": "C3",
        "STRESS_TEST": "C4",
    }

    by_origin: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for task_id, (mean_rating, origin) in mean_ratings.items():
        if task_id not in index.scores:
            continue
        consistency = index.scores[task_id]
        by_origin[origin].append((consistency, mean_rating))

    fig, ax = plt.subplots(figsize=(10, 7))

    for origin, points in sorted(by_origin.items()):
        if not points:
            continue
        xs, ys = zip(*points)
        color = origins_colors.get(origin, "gray")
        ax.scatter(xs, ys, c=color, label=f"{origin} ({len(points)})", alpha=0.6, s=50)

    ax.set_xlabel("Consistency (1 - normalized cross-seed std)")
    ax.set_ylabel("Mean Rating")
    ax.set_title(f"Task Consistency vs Mean Rating by Origin ({model})")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.05, 1.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name for filtering runs")
    parser.add_argument("--consistency-key", type=str, default=None, help="Consistency index key (default: derived from model)")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID (default: {model}_consistency)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    consistency_key = args.consistency_key or args.model.replace("-", "_").replace(".", "_")
    experiment_id = args.experiment_id or f"{args.model.replace('-', '_')}_consistency"

    date_str = datetime.now().strftime("%m%d%y")
    output_path = args.output or OUTPUT_DIR / f"plot_{date_str}_consistency_vs_mean_{consistency_key}.png"

    plot_consistency_vs_mean(consistency_key, experiment_id, args.model, output_path)


if __name__ == "__main__":
    main()
