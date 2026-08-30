"""Runner for noise baselines using RunDirProbeConfig."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_eval_data, load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.experiments.run_dir_probes import RunDirProbeConfig, train_ridge_heldout
from src.probes.residualization import demean_scores

from .types import BaselineResult, BaselineType


def run_noise_baselines(
    config: RunDirProbeConfig,
    n_seeds: int = 10,
) -> list[BaselineResult]:
    """Run shuffled-labels and random-activations baselines.

    Uses the same data loading path as run_dir_probes.py.
    Evaluates on heldout data (Pearson r + pairwise acc).
    """
    scores = load_thurstonian_scores(config.run_dir)
    print(f"Loaded {len(scores)} Thurstonian scores")

    if config.demean_confounds and scores:
        assert config.topics_json is not None, "topics_json required for demeaning"
        scores, metadata_stats = demean_scores(
            scores, config.topics_json, confounds=config.demean_confounds,
        )
        print(f"Demeaned: R²={metadata_stats['metadata_r2']:.4f}, "
              f"{metadata_stats['n_tasks_demeaned']} tasks retained")

    # Load eval data (eval_run_dir is mandatory)
    eval_scores, eval_measurements = load_eval_data(
        config.eval_run_dir, set(scores.keys()),
        demean_confounds=config.demean_confounds,
        topics_json=config.topics_json,
    )

    task_id_filter = set(scores.keys()) | set(eval_scores.keys())

    print(f"\nBaseline mode: heldout")

    results: list[BaselineResult] = []

    for layer in config.layers:
        print(f"\n--- Layer {layer} ---")
        task_ids, activations = load_activations(
            config.activations_path,
            task_id_filter=task_id_filter,
            layers=[layer],
        )

        indices, y = build_ridge_xy(task_ids, scores)
        if len(indices) < 20:
            print(f"  Skipping: insufficient samples ({len(indices)})")
            continue

        X = activations[layer][indices]

        results.extend(_run_heldout_baselines(
            X, y, activations[layer], task_ids, layer,
            eval_scores, eval_measurements,
            config, n_seeds,
        ))

    return results


def _run_heldout_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    activations: np.ndarray,
    task_ids: np.ndarray,
    layer: int,
    eval_scores: dict[str, float],
    eval_measurements: list,
    config: RunDirProbeConfig,
    n_seeds: int,
) -> list[BaselineResult]:
    """Run both noise baselines with heldout evaluation."""
    results: list[BaselineResult] = []
    n_samples = len(y_train)
    print(f"  {n_samples} train samples, running {n_seeds} seeds (heldout eval)...")

    # Pre-compute stats for random activations (constant across seeds)
    act_mean = X_train.mean(axis=0)
    act_std = X_train.std(axis=0)

    for seed in tqdm(range(n_seeds), desc=f"  L{layer} baselines", leave=False):
        rng = np.random.default_rng(seed)

        # Shuffled labels: real X, shuffled y
        y_shuffled = rng.permutation(y_train)
        metrics = train_ridge_heldout(
            X_train, y_shuffled, activations, task_ids,
            eval_scores, eval_measurements, layer,
            standardize=config.standardize,
            alpha_sweep_size=config.alpha_sweep_size,
            eval_split_seed=config.eval_split_seed,
        )
        results.append(BaselineResult.from_heldout_result(
            metrics, BaselineType.SHUFFLED_LABELS, layer, n_samples, seed,
        ))

        # Random activations: random X, real y
        X_noise = rng.normal(loc=act_mean, scale=act_std, size=X_train.shape)
        metrics = train_ridge_heldout(
            X_noise, y_train, activations, task_ids,
            eval_scores, eval_measurements, layer,
            standardize=config.standardize,
            alpha_sweep_size=config.alpha_sweep_size,
            eval_split_seed=config.eval_split_seed,
        )
        results.append(BaselineResult.from_heldout_result(
            metrics, BaselineType.RANDOM_ACTIVATIONS, layer, n_samples, seed,
        ))

    return results


def aggregate_noise_baselines(results: list[BaselineResult]) -> list[dict]:
    """Aggregate noise baselines across seeds, computing mean/std.

    Returns one entry per (baseline_type, layer) with aggregated stats.
    """
    grouped: dict[tuple, list[BaselineResult]] = defaultdict(list)
    for r in results:
        key = (r.baseline_type, r.layer)
        grouped[key].append(r)

    aggregated = []
    for (baseline_type, layer), items in grouped.items():
        entry: dict = {
            "baseline_type": baseline_type.value,
            "layer": layer,
            "n_samples": items[0].n_samples,
            "n_seeds": len(items),
        }

        # Heldout metrics
        if items[0].heldout_r is not None:
            rs = [r.heldout_r for r in items if r.heldout_r is not None]
            entry["heldout_r_mean"] = float(np.mean(rs))
            entry["heldout_r_std"] = float(np.std(rs))
        if items[0].heldout_acc is not None:
            accs = [r.heldout_acc for r in items if r.heldout_acc is not None]
            entry["heldout_acc_mean"] = float(np.mean(accs))
            entry["heldout_acc_std"] = float(np.std(accs))

        aggregated.append(entry)

    return aggregated
