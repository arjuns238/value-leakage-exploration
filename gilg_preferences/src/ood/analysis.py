"""Delta computation and correlation utilities for OOD generalization experiments."""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def _split_probe(probe_path: Path) -> tuple[np.ndarray, float]:
    """Load probe .npy and split into (weights, bias)."""
    probe = np.load(probe_path)
    return probe[:-1], float(probe[-1])


def _score_activations(
    npz_path: Path,
    layer: int,
    weights: np.ndarray,
    bias: float,
) -> dict[str, float]:
    """Load activations from npz, score with probe, return {task_id: score}."""
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return {tid: float(s) for tid, s in zip(task_ids, scores)}


def compute_p_choose_from_pairwise(
    results: list[dict],
    task_ids: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-task choice rates from pairwise results.

    Returns {condition_id: {task_id: p_choose}}.
    Each task's p_choose = total_wins / total_non_refusal_comparisons.
    """
    # Accumulate per (condition, task)
    wins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in results:
        cid = entry["condition_id"]
        ta, tb = entry["task_a"], entry["task_b"]
        non_refusal = entry["n_a"] + entry["n_b"]

        wins[cid][ta] += entry["n_a"]
        wins[cid][tb] += entry["n_b"]
        total[cid][ta] += non_refusal
        total[cid][tb] += non_refusal

    rates: dict[str, dict[str, float]] = {}
    for cid in wins:
        rates[cid] = {}
        tids = task_ids if task_ids is not None else set(wins[cid].keys())
        for tid in tids:
            w = wins[cid].get(tid, 0)
            t = total[cid].get(tid, 0)
            rates[cid][tid] = w / t if t > 0 else float("nan")

    return rates


def _build_rate_lookup(measurements: list[dict]) -> dict[str, dict[str, float]]:
    """Build {condition_id: {task_id: p_choose}} from flat measurements list."""
    lookup: dict[str, dict[str, float]] = {}
    for m in measurements:
        cid = m["condition_id"]
        if cid not in lookup:
            lookup[cid] = {}
        lookup[cid][m["task_id"]] = m["p_choose"]
    return lookup


def compute_deltas(
    rates: dict[str, dict[str, float]],
    activations_dir: Path,
    probe_path: Path,
    layer: int,
    baseline_activations_key: str = "baseline",
    activations_filename: str = "activations_turn_boundary_-1.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute behavioral and probe deltas for all conditions vs baseline.

    Args:
        rates: {condition_id: {task_id: p_choose}} — from compute_p_choose_from_pairwise()
            or _build_rate_lookup().
        activations_dir: Dir with {condition_id}/{activations_filename}.
        probe_path: Path to probe .npy file.
        layer: Layer number (e.g. 31).
        baseline_activations_key: Subdirectory name for baseline activations.
        activations_filename: Name of the npz file in each condition dir.

    Returns:
        (behavioral_deltas, probe_deltas, condition_labels) — arrays pooled
        across all non-baseline conditions × tasks.
    """
    weights, bias = _split_probe(probe_path)
    baseline_rates = rates["baseline"]

    baseline_npz = activations_dir / baseline_activations_key / activations_filename
    baseline_scores = _score_activations(baseline_npz, layer, weights, bias)

    all_behavioral = []
    all_probe = []
    all_labels = []

    for cid in rates:
        if cid == "baseline":
            continue

        cond_npz = activations_dir / cid / activations_filename
        if not cond_npz.exists():
            warnings.warn(f"Missing activations for condition '{cid}': {cond_npz}")
            continue
        cond_scores = _score_activations(cond_npz, layer, weights, bias)

        for tid, cond_rate in rates[cid].items():
            if tid not in baseline_rates or tid not in baseline_scores or tid not in cond_scores:
                continue

            b_delta = cond_rate - baseline_rates[tid]
            p_delta = cond_scores[tid] - baseline_scores[tid]

            all_behavioral.append(b_delta)
            all_probe.append(p_delta)
            all_labels.append(cid)

    return np.array(all_behavioral), np.array(all_probe), np.array(all_labels)


def correlate_deltas(
    behavioral_deltas: np.ndarray,
    probe_deltas: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Pearson r, sign agreement, permutation test."""
    r, p = stats.pearsonr(behavioral_deltas, probe_deltas)
    spearman_r, spearman_p = stats.spearmanr(behavioral_deltas, probe_deltas)

    # Sign agreement (exclude near-zero behavioral deltas)
    threshold = 0.02
    mask = np.abs(behavioral_deltas) >= threshold
    if mask.sum() > 0:
        sign_agree = float((np.sign(behavioral_deltas[mask]) == np.sign(probe_deltas[mask])).mean())
        sign_n = int(mask.sum())
    else:
        sign_agree = float("nan")
        sign_n = 0

    # Permutation test
    rng = np.random.RandomState(seed)
    perm_rs = np.array([
        stats.pearsonr(behavioral_deltas[rng.permutation(len(behavioral_deltas))], probe_deltas)[0]
        for _ in range(n_permutations)
    ])
    perm_p = float((perm_rs >= r).mean())

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "sign_agreement": sign_agree,
        "sign_n": sign_n,
        "n": len(behavioral_deltas),
        "permutation_p": perm_p,
        "permutation_mean_r": float(perm_rs.mean()),
    }


def per_condition_correlations(
    behavioral_deltas: np.ndarray,
    probe_deltas: np.ndarray,
    condition_labels: np.ndarray,
    min_n: int = 5,
) -> dict[str, dict]:
    """Per-condition Pearson r breakdown."""
    results = {}
    for cid in np.unique(condition_labels):
        mask = condition_labels == cid
        if mask.sum() < min_n:
            continue
        b = behavioral_deltas[mask]
        p = probe_deltas[mask]
        r, pval = stats.pearsonr(b, p)
        results[cid] = {
            "pearson_r": float(r),
            "pearson_p": float(pval),
            "n": int(mask.sum()),
        }
    return results
