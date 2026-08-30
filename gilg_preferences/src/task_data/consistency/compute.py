"""Task consistency filter based on cross-seed variance.

Computes consistency metrics from experiment data to filter out tasks where models
don't show consistent preferences across random seeds.

Usage:
    # Compute consistency for a model (run on many tasks, e.g., 2000)
    python -m src.task_data.consistency.compute \
        --experiment-dirs results/experiments/multi_model_discrimination_v1 \
        --model gemma-2-27b

    # Filter tasks by consistency (keeps top 70% by default)
    from src.task_data.consistency import make_consistency_filter
    filter_fn = make_consistency_filter("gemma2", keep_ratio=0.7)
    tasks = load_tasks(n=200, origins=[...], filter_fn=filter_fn)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.measurement.storage.run_parsing import (
    extract_model_from_run_dir,
    extract_template_from_run_dir,
    normalize_score,
    parse_scale_tag,
)
from src.task_data.task import Task
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset


DATA_DIR = Path(__file__).parent / "data"
ANALYSIS_DIR = Path(__file__).parent / "analysis"


@dataclass
class ConsistencyIndex:
    scores: dict[str, float]  # task_id -> consistency_score
    percentiles: dict[int, float]  # percentile -> threshold value


@dataclass
class _RunMeasurements:
    task_scores: dict[str, float]
    model: str
    template: str
    seed: int


def _load_run_measurements(run_dir: Path) -> _RunMeasurements | None:
    config_path = run_dir / "config.yaml"
    measurements_path = run_dir / "measurements.yaml"

    if not config_path.exists() or not measurements_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = extract_model_from_run_dir(run_dir.name)
    template = extract_template_from_run_dir(run_dir.name)
    if not model or not template:
        return None

    if "qualitative" in template:
        return None

    scale_tag = config.get("template_tags", {}).get("scale")
    if not scale_tag:
        return None
    scale = parse_scale_tag(scale_tag)
    if not scale:
        return None

    seed = config["rating_seed"]

    with open(measurements_path) as f:
        measurements = yaml.safe_load(f)

    if not measurements:
        return None

    task_scores: dict[str, float] = {}
    for m in measurements:
        if "score" not in m or not isinstance(m["score"], (int, float)):
            continue
        task_scores[m["task_id"]] = normalize_score(float(m["score"]), scale)

    if not task_scores:
        return None

    return _RunMeasurements(task_scores, model, template, seed)


def _load_all_runs(
    experiment_dirs: list[Path], model_filter: str | None = None
) -> list[_RunMeasurements]:
    runs = []
    for exp_dir in experiment_dirs:
        stated_dir = exp_dir / "post_task_stated"
        if not stated_dir.exists():
            continue
        for run_dir in stated_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = _load_run_measurements(run_dir)
            if run:
                if model_filter and run.model != model_filter:
                    continue
                runs.append(run)
    return runs


def compute_consistency(
    experiment_dirs: list[Path], model_filter: str | None = None
) -> dict[str, float]:
    """Compute consistency scores for tasks. Returns task_id -> consistency_score."""
    runs = _load_all_runs(experiment_dirs, model_filter)
    if not runs:
        return {}

    all_task_ids: set[str] = set()
    for run in runs:
        all_task_ids.update(run.task_scores.keys())

    # Group by template for cross-seed variance
    by_template: dict[str, dict[int, _RunMeasurements]] = defaultdict(dict)
    for run in runs:
        by_template[run.template][run.seed] = run

    # Collect all stds for normalization
    all_stds: list[float] = []
    task_stds: dict[str, list[float]] = defaultdict(list)

    for task_id in all_task_ids:
        for template, seed_runs in by_template.items():
            scores = []
            for seed, run in seed_runs.items():
                if task_id in run.task_scores:
                    scores.append(run.task_scores[task_id])
            if len(scores) >= 2:
                std = float(np.std(scores))
                task_stds[task_id].append(std)
                all_stds.append(std)

    # Fixed max_std: 0.5 is the theoretical max on a 0-1 normalized scale
    max_std = 0.5

    # Compute consistency scores
    results: dict[str, float] = {}
    for task_id in all_task_ids:
        stds = task_stds.get(task_id, [])
        if not stds:
            continue
        # Skip tasks with invalid stds (from bad data, e.g. score=850 on 1-5 scale)
        if any(s > max_std * 2 for s in stds):
            continue
        mean_std = float(np.mean(stds))
        results[task_id] = max(0.0, 1 - (mean_std / max_std))

    return results


def save_consistency_index(scores: dict[str, float], path: Path) -> None:
    sorted_scores = sorted(scores.values())
    percentiles = {p: float(np.percentile(sorted_scores, p)) for p in range(5, 100, 5)}

    data = {
        "scores": scores,
        "percentiles": percentiles,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved consistency index ({len(scores)} tasks) to {path}")


def load_consistency_index(model: str) -> ConsistencyIndex:
    path = DATA_DIR / f"consistency_{model}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No consistency index for model {model} at {path}. "
            f"Run: python -m src.task_data.consistency --experiment-dirs <path> --model <model>"
        )
    with open(path) as f:
        data = json.load(f)
    return ConsistencyIndex(
        scores=data["scores"],
        percentiles={int(k): v for k, v in data["percentiles"].items()},
    )


def make_consistency_filter(
    model: str,
    keep_ratio: float = 0.7,
) -> Callable[[Task], bool]:
    """Create a filter that keeps the top `keep_ratio` of tasks by consistency.

    Args:
        model: Model key (e.g., "gemma2", "qwen_think", "claude_haiku")
        keep_ratio: Fraction of tasks to keep (0.7 = keep top 70%, filter bottom 30%)
    """
    index = load_consistency_index(model)
    filter_pct = int((1 - keep_ratio) * 100)

    # Round to nearest 5 (our stored percentiles)
    filter_pct_rounded = max(5, min(95, 5 * round(filter_pct / 5)))
    threshold = index.percentiles[filter_pct_rounded]

    def filter_fn(task: Task) -> bool:
        score = index.scores.get(task.id)
        if score is None:
            return True  # Keep tasks not in index
        return score >= threshold

    return filter_fn


def save_ranked_consistency(scores: dict[str, float], path: Path) -> None:
    """Save human-readable ranked list with prompts."""
    all_tasks = []
    for origin in OriginDataset:
        if origin == OriginDataset.SYNTHETIC:
            continue
        all_tasks.extend(load_tasks(n=100000, origins=[origin]))
    task_lookup = {t.id: t for t in all_tasks}

    ranked = []
    for task_id, score in scores.items():
        task = task_lookup.get(task_id)
        if not task:
            continue
        ranked.append({
            "task_id": task_id,
            "origin": task.origin.name,
            "prompt": task.prompt,
            "consistency_score": round(score, 4),
        })

    ranked.sort(key=lambda x: x["consistency_score"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"Saved {len(ranked)} ranked tasks to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute task consistency metrics from experiment data"
    )
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Experiment directories to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to compute consistency for (e.g., gemma-2-27b)",
    )
    parser.add_argument(
        "--output-key",
        type=str,
        default=None,
        help="Output key for files (default: derived from model name)",
    )
    parser.add_argument(
        "--output-ranked",
        action="store_true",
        help="Also save human-readable ranked JSON",
    )
    args = parser.parse_args()

    experiment_dirs = [Path(d) for d in args.experiment_dirs]
    for d in experiment_dirs:
        if not d.exists():
            print(f"Warning: {d} does not exist")

    # Derive output key from model name if not specified
    output_key = args.output_key
    if not output_key:
        output_key = args.model.replace("-", "_").replace(".", "_")

    print(f"Computing consistency for {args.model}...")
    scores = compute_consistency(experiment_dirs, args.model)

    if not scores:
        print("No metrics computed")
        return

    score_values = list(scores.values())
    print(f"\nComputed consistency for {len(scores)} tasks:")
    print(f"  Mean: {np.mean(score_values):.3f}, Std: {np.std(score_values):.3f}")
    print(f"  Min: {min(score_values):.3f}, Max: {max(score_values):.3f}")

    # Show percentile distribution
    print("\n  Percentiles:")
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        val = np.percentile(score_values, p)
        print(f"    {p}th: {val:.3f}")

    # Save compact index
    index_path = DATA_DIR / f"consistency_{output_key}.json"
    save_consistency_index(scores, index_path)

    # Save ranked list if requested
    if args.output_ranked:
        ranked_path = ANALYSIS_DIR / f"consistency_{output_key}_ranked.json"
        save_ranked_consistency(scores, ranked_path)


if __name__ == "__main__":
    main()
