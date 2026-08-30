"""Core correlation utilities.

This module contains low-level correlation functions used across the codebase.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

ReliabilityMethod = Literal["pearson", "spearman", "discrimination", "informative"]

if TYPE_CHECKING:
    from src.task_data import Task
    from src.types import BinaryPreferenceMeasurement, TaskScore


def compute_discrimination_ratio(task_ratings: dict[str, list[float]]) -> float:
    """Compute discrimination ratio: Var(between tasks) / Var(total).

    This metric rewards templates that:
    1. Produce different ratings for different tasks (high between-task variance)
    2. Are stable across seeds for the same task (low within-task variance)

    A template that always says "4" will have ~0 between-task variance and score low.
    A template with high seed noise will have high within-task variance and score low.

    Args:
        task_ratings: task_id -> list of ratings across seeds

    Returns:
        Ratio in [0, 1] where 1 = perfect discrimination, 0 = no discrimination
    """
    if len(task_ratings) < 2:
        return 0.0

    # Need at least 2 seeds per task to estimate within-task variance
    tasks_with_multiple = {k: v for k, v in task_ratings.items() if len(v) >= 2}
    if len(tasks_with_multiple) < 2:
        return 0.0

    # Compute task means and within-task variances
    task_means = []
    within_variances = []
    for ratings in tasks_with_multiple.values():
        task_means.append(np.mean(ratings))
        within_variances.append(np.var(ratings, ddof=1))

    task_means = np.array(task_means)
    var_between = np.var(task_means, ddof=1)
    var_within = np.mean(within_variances)

    var_total = var_between + var_within
    if var_total < 1e-10:
        return 0.0

    return float(var_between / var_total)


def compute_informative_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Compute correlation on non-modal responses only.

    Returns (correlation, discrimination_rate) where:
    - correlation: Pearson r on tasks where at least one rating != mode
    - discrimination_rate: fraction of tasks that pass this filter

    This answers: "When the model discriminates, is it consistent?"
    """
    if len(a) < 2:
        return 0.0, 0.0

    # Find mode (most common value, rounded to handle float imprecision)
    pooled = np.concatenate([a, b])
    rounded = np.round(pooled)
    values, counts = np.unique(rounded, return_counts=True)
    mode = values[np.argmax(counts)]

    # Mask: include task if either rating differs from mode
    mask = (np.round(a) != mode) | (np.round(b) != mode)
    n_informative = mask.sum()
    discrimination_rate = n_informative / len(a)

    if n_informative < 2:
        return float("nan"), discrimination_rate

    a_filtered = a[mask]
    b_filtered = b[mask]

    # Check for zero variance in filtered data
    if np.std(a_filtered) < 1e-10 or np.std(b_filtered) < 1e-10:
        return float("nan"), discrimination_rate

    r, _ = pearsonr(a_filtered, b_filtered)
    return float(r) if not np.isnan(r) else float("nan"), discrimination_rate


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """Compute correlation, returning 0.0 if insufficient data or zero variance."""
    if len(a) < 2:
        return 0.0

    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0

    corr_fn = pearsonr if method == "pearson" else spearmanr
    r, _ = corr_fn(a, b)
    return float(r) if not np.isnan(r) else 0.0


def utility_vector_correlation(
    mu_a: np.ndarray,
    task_ids_a: list[str],
    mu_b: np.ndarray,
    task_ids_b: list[str],
    method: ReliabilityMethod = "pearson",
    min_overlap: int = 10,
) -> float:
    """Compute reliability metric between utility vectors on overlapping tasks.

    Args:
        method: "pearson", "spearman", or "discrimination" (variance ratio)

    Returns NaN if overlap is less than min_overlap or data is insufficient.
    """
    common_ids = set(task_ids_a) & set(task_ids_b)
    if len(common_ids) < min_overlap:
        return float("nan")

    id_to_idx_a = {tid: i for i, tid in enumerate(task_ids_a)}
    id_to_idx_b = {tid: i for i, tid in enumerate(task_ids_b)}

    common_list = sorted(common_ids)
    vals_a = np.array([mu_a[id_to_idx_a[tid]] for tid in common_list])
    vals_b = np.array([mu_b[id_to_idx_b[tid]] for tid in common_list])

    if method == "discrimination":
        # For pairwise comparison, estimate discrimination from just two measurements
        # Build task_ratings dict with two entries per task
        task_ratings = {tid: [vals_a[i], vals_b[i]] for i, tid in enumerate(common_list)}
        return compute_discrimination_ratio(task_ratings)

    if method == "informative":
        r, _ = compute_informative_correlation(vals_a, vals_b)
        return r

    if np.std(vals_a) < 1e-10 or np.std(vals_b) < 1e-10:
        return float("nan")

    corr_fn = pearsonr if method == "pearson" else spearmanr
    r, _ = corr_fn(vals_a, vals_b)
    return float(r) if not np.isnan(r) else float("nan")


def compute_pairwise_correlations(
    results: dict[str, tuple[np.ndarray, list[str]]],
    tags: dict[str, dict[str, str]] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
    min_overlap: int = 10,
) -> list[dict]:
    """Compute pairwise correlations between all result sets.

    Args:
        results: template_id -> (values, task_ids)
        tags: optional template_id -> tag dict for sensitivity analysis
        method: "pearson" or "spearman"
        min_overlap: minimum number of overlapping tasks required
    """
    correlations = []
    for (id_a, (vals_a, ids_a)), (id_b, (vals_b, ids_b)) in combinations(results.items(), 2):
        corr = utility_vector_correlation(vals_a, ids_a, vals_b, ids_b, method, min_overlap)
        entry = {
            "template_a": id_a,
            "template_b": id_b,
            "correlation": float(corr),
        }
        if tags is not None:
            entry["tags_a"] = tags[id_a]
            entry["tags_b"] = tags[id_b]
        correlations.append(entry)
    return correlations


# --- Stated preference helpers ---


def build_score_map(scores: list[TaskScore]) -> dict[str, float]:
    """Build task_id -> mean score map, averaging repeated samples."""
    id_to_scores: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        id_to_scores[s.task.id].append(s.score)
    return {tid: float(np.mean(vals)) for tid, vals in id_to_scores.items()}


def compute_per_task_std(scores: list[TaskScore]) -> dict[str, float]:
    """Compute std of scores across repeated samples for each task."""
    id_to_scores: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        id_to_scores[s.task.id].append(s.score)
    return {tid: float(np.std(vals)) for tid, vals in id_to_scores.items()}


def compute_mean_std_across_tasks(scores: list[TaskScore]) -> float:
    """Average std across all tasks (summary metric for sample consistency)."""
    per_task = compute_per_task_std(scores)
    return float(np.mean(list(per_task.values())))


def scores_to_vector(scores: list[TaskScore], tasks: list[Task]) -> tuple[np.ndarray, list[str]]:
    """Convert TaskScore list to (values, task_ids) for use with compute_pairwise_correlations."""
    score_map = build_score_map(scores)
    task_ids = [t.id for t in tasks if t.id in score_map]
    values = np.array([score_map[tid] for tid in task_ids])
    return values, task_ids


# --- Revealed preference helpers ---


def build_win_rate_vector(
    measurements: list[BinaryPreferenceMeasurement],
    tasks: list[Task],
) -> np.ndarray:
    """Win rate of task i over task j for all pairs i < j, flattened."""
    n = len(tasks)
    id_to_idx = {t.id: i for i, t in enumerate(tasks)}

    wins = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    for m in measurements:
        i = id_to_idx[m.task_a.id]
        j = id_to_idx[m.task_b.id]
        if i > j:
            i, j = j, i
            choice = "b" if m.choice == "a" else "a"
        else:
            choice = m.choice

        counts[i, j] += 1
        if choice == "a":
            wins[i, j] += 1

    rates = []
    for i in range(n):
        for j in range(i + 1, n):
            if counts[i, j] > 0:
                rates.append(wins[i, j] / counts[i, j])
            else:
                rates.append(0.5)

    return np.array(rates)


def win_rate_correlation(
    measurements_a: list[BinaryPreferenceMeasurement],
    measurements_b: list[BinaryPreferenceMeasurement],
    tasks: list[Task],
) -> float:
    """Pearson correlation of win rates between two sets of measurements."""
    rates_a = build_win_rate_vector(measurements_a, tasks)
    rates_b = build_win_rate_vector(measurements_b, tasks)
    return safe_correlation(rates_a, rates_b, "pearson")


# --- YAML save helpers ---


def save_correlations_yaml(
    correlations: list[dict],
    summary_keys: list[str],
    path: Path | str,
) -> None:
    """Save correlations to YAML with mean summary for specified keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"n_pairs": len(correlations)}
    for key in summary_keys:
        values = [c[key] for c in correlations]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else 0.0

    output = {
        "summary": summary,
        "pairwise": correlations,
    }

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def save_experiment_config(
    templates: list,
    model_name: str,
    temperature: float,
    n_tasks: int,
    path: Path | str,
) -> None:
    """Save experiment configuration to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "temperature": temperature,
        "n_tasks": n_tasks,
        "templates": [
            {
                "name": t.name,
                "tags": list(t.tags),
                "prompt": t.template,
            }
            for t in templates
        ],
    }

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
