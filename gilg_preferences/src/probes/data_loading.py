"""Shared measurement and activation loading utilities for probe training."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from src.measurement.storage.loading import load_raw_scores, load_run_utilities, load_yaml
from src.probes.residualization import demean_scores
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


def load_thurstonian_scores(run_dir: Path) -> dict[str, float]:
    """Load task_id -> mu mapping from Thurstonian fit."""
    mu_array, task_ids = load_run_utilities(run_dir)
    return dict(zip(task_ids, mu_array))


def load_pairwise_measurements(run_dir: Path) -> list[BinaryPreferenceMeasurement]:
    """Load measurements and reconstruct as BinaryPreferenceMeasurement objects."""
    measurements_path = run_dir / "measurements.yaml"
    if not measurements_path.exists():
        return []
    raw = load_yaml(measurements_path)

    measurements = []
    for m in raw:
        task_a = Task(
            id=m["task_a"],
            prompt="",
            origin=OriginDataset[m["origin_a"]],
            metadata={},
        )
        task_b = Task(
            id=m["task_b"],
            prompt="",
            origin=OriginDataset[m["origin_b"]],
            metadata={},
        )
        measurements.append(BinaryPreferenceMeasurement(
            task_a=task_a,
            task_b=task_b,
            choice=m["choice"],
            preference_type=PreferenceType.POST_TASK_REVEALED,
        ))

    return measurements


def load_eval_data(
    eval_run_dir: Path,
    train_task_ids: set[str],
    demean_confounds: list[str] | None = None,
    topics_json: Path | None = None,
) -> tuple[dict[str, float], list[BinaryPreferenceMeasurement]]:
    """Load eval scores and measurements, removing overlap with train tasks.

    Returns (eval_scores, eval_measurements) with train-overlapping tasks removed
    and optional demeaning applied.
    """
    eval_scores = load_thurstonian_scores(eval_run_dir)
    eval_measurements = load_pairwise_measurements(eval_run_dir)
    print(f"  Eval: {len(eval_scores)} scores, {len(eval_measurements)} comparisons")

    overlap = set(eval_scores.keys()) & train_task_ids
    if overlap:
        eval_scores = {k: v for k, v in eval_scores.items() if k not in overlap}
        eval_measurements = [
            m for m in eval_measurements
            if m.task_a.id not in overlap and m.task_b.id not in overlap
        ]
        print(f"  Removed {len(overlap)} overlapping train tasks"
              f" -> {len(eval_scores)} eval scores, {len(eval_measurements)} comparisons")

    if demean_confounds and eval_scores:
        assert topics_json is not None
        eval_scores, eval_stats = demean_scores(
            eval_scores, topics_json, confounds=demean_confounds,
        )
        print(f"  Eval demeaned R²={eval_stats['metadata_r2']:.4f}")

    return eval_scores, eval_measurements


def load_measurements_for_templates(
    experiment_dir: Path,
    templates: list[str],
    seeds: list[int],
) -> dict[str, float]:
    """Load measurements and average scores per task_id."""
    raw_measurements = []
    for template in templates:
        task_type = "pre_task" if template.startswith("pre_task") else "post_task"
        measurement_dir = experiment_dir / f"{task_type}_stated"
        raw_measurements.extend(load_raw_scores(measurement_dir, [template], seeds))

    # Average scores per task_id
    scores_by_task: dict[str, list[float]] = {}
    for task_id, score in raw_measurements:
        if task_id not in scores_by_task:
            scores_by_task[task_id] = []
        scores_by_task[task_id].append(score)

    return {tid: np.mean(scores) for tid, scores in scores_by_task.items()}


def filter_task_ids_by_datasets(
    task_ids: set[str],
    datasets: list[str] | None,
    origins_cache: dict[str, set[str]],
) -> set[str]:
    """Filter task IDs to only those in specified datasets. Returns all if datasets is None."""
    if not datasets:
        return task_ids
    target_ids = set()
    for dataset in datasets:
        target_ids.update(origins_cache.get(dataset.upper(), set()))
    return task_ids & target_ids


def expand_training_combinations(
    template_combos: list[list[str]],
    seed_combos: list[list[int]],
    layers: list[int],
    dataset_combos: list[list[str]] | None = None,
) -> list[dict]:
    """Returns list of dicts with keys: templates, datasets, seeds, layer."""
    ds_combos = dataset_combos or [None]

    return [
        {
            "templates": templates,
            "datasets": datasets,
            "seeds": seeds,
            "layer": layer,
        }
        for templates, seeds, datasets, layer in product(
            template_combos,
            seed_combos,
            ds_combos,
            layers,
        )
    ]


def collect_measurements_cache(
    experiment_dir: Path,
    combinations: list[dict],
) -> dict[tuple, dict[str, float]]:
    """Load and cache averaged measurements for all unique template/seed combinations."""
    cache: dict[tuple, dict[str, float]] = {}
    for combo in combinations:
        key = (tuple(combo["templates"]), tuple(combo["seeds"]))
        if key not in cache:
            cache[key] = load_measurements_for_templates(
                experiment_dir, combo["templates"], combo["seeds"]
            )
    return cache


def collect_needed_task_ids(
    measurements_cache: dict[tuple, dict[str, float]],
    combinations: list[dict],
    origins_cache: dict[str, set[str]],
) -> set[str]:
    """Collect union of all task IDs needed across all combinations."""
    all_task_ids: set[str] = set()

    for combo in combinations:
        key = (tuple(combo["templates"]), tuple(combo["seeds"]))
        task_ids = set(measurements_cache[key].keys())
        task_ids = filter_task_ids_by_datasets(task_ids, combo["datasets"], origins_cache)
        all_task_ids.update(task_ids)

    return all_task_ids
