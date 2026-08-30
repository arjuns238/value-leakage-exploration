"""Train probes from a measurement run directory.

Supports two training modes:
1. Ridge regression on Thurstonian mu values (utility scores)
2. Bradley-Terry on pairwise comparison data

Usage:
    python -m src.probes.experiments.run_dir_probes --config configs/probes/example.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from src.probes.experiments.plot_hoo import plot_hoo_summary
except ModuleNotFoundError:
    plot_hoo_summary = lambda *a, **kw: None  # matplotlib not available
from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe
from src.probes.core.storage import load_probe, save_probe, save_manifest
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import pairwise_accuracy_from_scores, train_bt
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements, load_eval_data
from src.probes.experiments import hoo_ridge, hoo_bt
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.residualization import build_task_groups, demean_scores


class ProbeMode(Enum):
    RIDGE = "ridge"
    BRADLEY_TERRY = "bradley_terry"


@dataclass
class RunDirProbeConfig:
    experiment_name: str
    run_dir: Path
    activations_path: Path
    output_dir: Path
    layers: list[int]
    modes: list[ProbeMode]
    eval_run_dir: Path
    alpha_sweep_size: int = 50
    demean_confounds: list[str] | None = None
    topics_json: Path | None = None
    standardize: bool = True
    n_jobs: int = 1
    eval_split_seed: int = 42
    # HOO settings — if hoo_grouping is set, runs HOO instead of standard training
    hoo_grouping: str | None = None  # "topic" | "dataset"
    hoo_hold_out_size: int = 1
    hoo_groups: list[str] | None = None  # specific groups, or None for all
    uniform_eval_run_dir: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> RunDirProbeConfig:
        with open(path) as f:
            data = yaml.safe_load(f)

        if "eval_run_dir" not in data:
            raise ValueError(f"eval_run_dir is required in config {path}")

        modes = [ProbeMode(m) for m in data.get("modes", ["ridge", "bradley_terry"])]
        topics_json = Path(data["topics_json"]) if "topics_json" in data else None

        # Accept train_run_dir as alias for run_dir (for heldout eval configs)
        run_dir_raw = data.get("run_dir") or data["train_run_dir"]

        optional = {}
        for key in (
            "alpha_sweep_size", "demean_confounds", "standardize",
            "n_jobs", "eval_split_seed", "hoo_grouping", "hoo_hold_out_size", "hoo_groups",
            "uniform_eval_run_dir",
        ):
            if key in data:
                optional[key] = data[key]

        if "uniform_eval_run_dir" in optional:
            optional["uniform_eval_run_dir"] = Path(optional["uniform_eval_run_dir"])

        return cls(
            experiment_name=data["experiment_name"],
            run_dir=Path(run_dir_raw),
            activations_path=Path(data["activations_path"]),
            output_dir=Path(data["output_dir"]),
            layers=data["layers"],
            modes=modes,
            topics_json=topics_json,
            eval_run_dir=Path(data["eval_run_dir"]),
            **optional,
        )


def train_ridge_heldout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    activations: np.ndarray,
    task_ids: np.ndarray,
    eval_scores: dict[str, float],
    eval_measurements: list,
    layer: int,
    standardize: bool = True,
    alpha_sweep_size: int = 50,
    eval_split_seed: int = 42,
) -> dict:
    """Train Ridge on X_train/y_train, sweep alpha and evaluate on heldout data.

    Returns dict with: weights, best_alpha, sweep_r, final_r, final_acc,
    n_train, n_sweep, n_final, n_final_pairs, alpha_sweep.
    """
    # Split eval into sweep and final halves
    rng = np.random.default_rng(eval_split_seed)
    eval_task_ids = sorted(eval_scores.keys())
    perm = rng.permutation(len(eval_task_ids))
    half = len(eval_task_ids) // 2
    sweep_ids = {eval_task_ids[i] for i in perm[:half]}
    final_ids = {eval_task_ids[i] for i in perm[half:]}
    sweep_scores = {tid: eval_scores[tid] for tid in sweep_ids}
    final_scores = {tid: eval_scores[tid] for tid in final_ids}

    sweep_indices, y_sweep = build_ridge_xy(task_ids, sweep_scores)
    final_indices, y_final = build_ridge_xy(task_ids, final_scores)

    # Build pairwise data filtered to final set
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    final_idx_set = {id_to_idx[tid] for tid in final_ids if tid in id_to_idx}
    final_bt_data = None
    if eval_measurements:
        all_bt_data = PairwiseActivationData.from_measurements(
            eval_measurements, task_ids, {layer: activations},
        )
        final_bt_data = all_bt_data.filter_by_indices(final_idx_set)

    X_sweep = activations[sweep_indices]

    if standardize:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_sweep_scaled = scaler.transform(X_sweep)
    else:
        X_train_scaled = X_train
        X_sweep_scaled = X_sweep
        scaler = None

    # Alpha sweep: train on train set, eval Pearson r on sweep half
    alphas = np.logspace(-1, 5, alpha_sweep_size)
    best_alpha = None
    best_sweep_r = -1.0
    sweep_results = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_sweep_scaled)
        if len(y_pred) >= 10:
            r, _ = pearsonr(y_sweep, y_pred)
            r = float(r)
        else:
            r = float("nan")
        sweep_results.append({"alpha": float(alpha), "sweep_r": r})
        if not np.isnan(r) and r > best_sweep_r:
            best_sweep_r = r
            best_alpha = float(alpha)

    # Final eval at best alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_scaled, y_train)

    if standardize:
        X_final_scaled = scaler.transform(activations[final_indices])
    else:
        X_final_scaled = activations[final_indices]
    y_pred_final = ridge.predict(X_final_scaled)

    final_r = None
    if len(y_pred_final) >= 10:
        r_val, _ = pearsonr(y_final, y_pred_final)
        final_r = float(r_val)

    # Pairwise accuracy on final half (predict in raw space)
    if standardize:
        coef_raw = ridge.coef_ / scaler.scale_
        intercept_raw = ridge.intercept_ - coef_raw @ scaler.mean_
    else:
        coef_raw = ridge.coef_
        intercept_raw = ridge.intercept_

    final_acc = None
    if final_bt_data is not None and len(final_bt_data.pairs) > 0:
        all_predicted = activations @ coef_raw + intercept_raw
        final_acc = pairwise_accuracy_from_scores(all_predicted, final_bt_data)

    weights = np.append(coef_raw, intercept_raw)

    return {
        "weights": weights,
        "best_alpha": best_alpha,
        "sweep_r": best_sweep_r,
        "final_r": final_r,
        "final_acc": final_acc,
        "n_train": len(y_train),
        "n_sweep": len(y_sweep),
        "n_final": len(y_final),
        "n_final_pairs": len(final_bt_data.pairs) if final_bt_data else 0,
        "alpha_sweep": sweep_results,
    }


def _train_ridge_probe_heldout(
    config: RunDirProbeConfig,
    layer: int,
    task_ids: np.ndarray,
    activations: np.ndarray,
    scores: dict[str, float],
    eval_scores: dict[str, float],
    eval_measurements: list,
    eval_split_seed: int,
) -> dict | None:
    """Train Ridge on train set, sweep alpha + evaluate on held-out eval set."""
    indices, y_train = build_ridge_xy(task_ids, scores)
    X_train = activations[indices]
    print(f"  Eval split: {len(eval_scores)} eval tasks")

    metrics = train_ridge_heldout(
        X_train, y_train, activations, task_ids,
        eval_scores, eval_measurements, layer,
        standardize=config.standardize,
        alpha_sweep_size=config.alpha_sweep_size,
        eval_split_seed=eval_split_seed,
    )

    print(f"  Best alpha: {metrics['best_alpha']:.4g} (sweep r={metrics['sweep_r']:.4f})")

    probe_id = f"ridge_L{layer:02d}"
    relative_path = save_probe(metrics["weights"], config.output_dir, probe_id)

    final_r_str = f"{metrics['final_r']:.4f}" if metrics['final_r'] is not None else "N/A"
    final_acc_str = f"{metrics['final_acc']:.4f}" if metrics['final_acc'] is not None else "N/A"
    print(f"  Final: r={final_r_str}, acc={final_acc_str}")

    return {
        "id": probe_id,
        "file": relative_path,
        "method": "ridge",
        "layer": layer,
        "standardize": config.standardize,
        "demean_confounds": config.demean_confounds,
        "best_alpha": metrics["best_alpha"],
        "sweep_r": metrics["sweep_r"],
        "final_r": metrics["final_r"],
        "final_acc": metrics["final_acc"],
        "n_train": metrics["n_train"],
        "n_sweep": metrics["n_sweep"],
        "n_final": metrics["n_final"],
        "n_final_pairs": metrics["n_final_pairs"],
        "alpha_sweep": metrics["alpha_sweep"],
    }


def _train_bt_probe(
    config: RunDirProbeConfig,
    data: PairwiseActivationData,
    layer: int,
) -> dict | None:
    """Train a single BT probe. Returns entry dict or None if no pairs."""
    if len(data.pairs) == 0:
        return None

    result = train_bt(data, layer, n_jobs=config.n_jobs)
    probe_id = f"bt_L{layer:02d}"
    relative_path = save_probe(result.weights, config.output_dir, probe_id)

    return {
        "id": probe_id,
        "file": relative_path,
        "method": "bradley_terry",
        "layer": layer,
        "train_accuracy": result.train_accuracy,
        "train_loss": result.train_loss,
        "cv_accuracy_mean": result.cv_accuracy_mean,
        "cv_accuracy_std": result.cv_accuracy_std,
        "best_l2_lambda": result.best_l2_lambda,
        "n_iterations": result.n_iterations,
        "n_pairs": len(data.pairs),
        "lambda_sweep": result.lambda_sweep,
    }


def _task_ids_from_measurements(measurements: list) -> set[str]:
    return {m.task_a.id for m in measurements} | {m.task_b.id for m in measurements}


def _compute_uniform_acc(
    uniform_bt_data: PairwiseActivationData | None,
    activations: np.ndarray,
    output_dir: Path,
    probe_id: str,
) -> float | None:
    if uniform_bt_data is None or len(uniform_bt_data.pairs) == 0:
        return None
    weights = load_probe(output_dir, probe_id)
    predicted = score_with_probe(weights, activations)
    return pairwise_accuracy_from_scores(predicted, uniform_bt_data)


def run_probes(config: RunDirProbeConfig) -> dict:
    """Train Ridge and Bradley-Terry probes from run directory data.

    Processes one layer at a time to limit memory usage.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_ridge = ProbeMode.RIDGE in config.modes
    run_bt = ProbeMode.BRADLEY_TERRY in config.modes
    mode_names = [m.value for m in config.modes]

    print(f"Training probes: {config.experiment_name}")
    print(f"Modes: {mode_names}")
    print(f"Run dir: {config.run_dir}")
    print(f"Eval run dir: {config.eval_run_dir}")
    print(f"Output: {config.output_dir}")

    # Load measurement data
    print("\nLoading measurement data...")
    scores = load_thurstonian_scores(config.run_dir) if run_ridge else {}
    measurements = load_pairwise_measurements(config.run_dir)
    if scores:
        print(f"  Loaded {len(scores)} task scores from Thurstonian fit")
    if measurements:
        print(f"  Loaded {len(measurements)} pairwise comparisons")

    # Load eval data (always — eval_run_dir is mandatory)
    eval_scores, eval_measurements = load_eval_data(
        config.eval_run_dir, set(scores.keys()),
        demean_confounds=config.demean_confounds,
        topics_json=config.topics_json,
    )

    # Load uniform eval data
    uniform_eval_measurements = load_pairwise_measurements(config.uniform_eval_run_dir) if config.uniform_eval_run_dir else []
    if uniform_eval_measurements:
        uniform_tids = _task_ids_from_measurements(uniform_eval_measurements)
        train_tids = set(scores.keys()) if scores else _task_ids_from_measurements(measurements)
        overlap = uniform_tids & train_tids
        if overlap:
            raise ValueError(
                f"Uniform eval has {len(overlap)} tasks that overlap with training data."
            )
        print(f"  Uniform eval: {len(uniform_eval_measurements)} measurements ({len(uniform_tids)} tasks)")

    # Optionally demean scores against metadata confounds
    metadata_stats = None
    if config.demean_confounds and scores:
        assert config.topics_json is not None, "topics_json required for demeaning"
        print(f"\nDemeaning scores against: {config.demean_confounds}")
        scores, metadata_stats = demean_scores(
            scores, config.topics_json, confounds=config.demean_confounds,
        )
        print(f"  Metadata R²={metadata_stats['metadata_r2']:.4f} "
              f"({metadata_stats['n_metadata_features']} features)")
        print(f"  {metadata_stats['n_tasks_demeaned']} tasks retained")

    task_id_filter = set(scores.keys()) if scores else None
    if task_id_filter is not None:
        task_id_filter = task_id_filter | set(eval_scores.keys())
    if uniform_eval_measurements:
        uniform_tids = _task_ids_from_measurements(uniform_eval_measurements)
        task_id_filter = (task_id_filter | uniform_tids) if task_id_filter else uniform_tids

    # Load one layer to get task_ids and count
    task_ids, first_layer_acts = load_activations(
        config.activations_path,
        task_id_filter=task_id_filter,
        layers=[config.layers[0]],
    )
    n_tasks = len(task_ids)
    del first_layer_acts
    gc.collect()

    manifest = {
        "experiment_name": config.experiment_name,
        "run_dir": str(config.run_dir),
        "activations_path": str(config.activations_path),
        "eval_run_dir": str(config.eval_run_dir),
        "eval_split_seed": config.eval_split_seed,
        "modes": mode_names,
        "standardize": config.standardize,
        "demean_confounds": config.demean_confounds,
        "created_at": datetime.now().isoformat(),
        "n_tasks_in_experiment": len(scores),
        "n_tasks_with_activations": n_tasks,
        "n_comparisons_in_experiment": len(measurements),
        "probes": [],
    }
    if config.uniform_eval_run_dir is not None:
        manifest["uniform_eval_run_dir"] = str(config.uniform_eval_run_dir)
    if metadata_stats is not None:
        manifest["metadata_r2"] = metadata_stats["metadata_r2"]
        manifest["metadata_features"] = metadata_stats["metadata_features"]
        manifest["n_tasks_demeaned"] = metadata_stats["n_tasks_demeaned"]
        manifest["n_tasks_dropped"] = metadata_stats["n_tasks_dropped"]

    # Process one layer at a time
    for layer in config.layers:
        print(f"\n--- Layer {layer} ---")
        task_ids, activations = load_activations(
            config.activations_path,
            task_id_filter=task_id_filter,
            layers=[layer],
        )

        # Build pairwise data for this layer (used by both Ridge and BT)
        bt_data = None
        if measurements:
            bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)

        # Build uniform eval pairwise data for this layer
        uniform_bt_data = None
        if uniform_eval_measurements:
            uniform_bt_data = PairwiseActivationData.from_measurements(
                uniform_eval_measurements, task_ids, activations,
            )

        if run_ridge:
            print(f"  Training Ridge probe...")
            ridge_entry = _train_ridge_probe_heldout(
                config, layer, task_ids, activations[layer], scores,
                eval_scores, eval_measurements,
                eval_split_seed=config.eval_split_seed,
            )
            if ridge_entry:
                uni_acc = _compute_uniform_acc(uniform_bt_data, activations[layer], config.output_dir, ridge_entry["id"])
                if uni_acc is not None:
                    ridge_entry["uniform_pairwise_acc"] = uni_acc
                manifest["probes"].append(ridge_entry)
                r_str = f"{ridge_entry['final_r']:.4f}" if ridge_entry['final_r'] is not None else "N/A"
                acc_str = f", acc={ridge_entry['final_acc']:.4f}" if ridge_entry['final_acc'] is not None else ""
                uni_str = f", uniform_acc={ridge_entry['uniform_pairwise_acc']:.4f}" if ridge_entry.get('uniform_pairwise_acc') is not None else ""
                print(f"  Ridge heldout: r={r_str}{acc_str}{uni_str}")

        if run_bt:
            print(f"  Training BT probe...")
            if bt_data is None:
                bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)
            bt_entry = _train_bt_probe(config, bt_data, layer)
            if bt_entry:
                uni_acc = _compute_uniform_acc(uniform_bt_data, activations[layer], config.output_dir, bt_entry["id"])
                if uni_acc is not None:
                    bt_entry["uniform_pairwise_acc"] = uni_acc
                manifest["probes"].append(bt_entry)
                uni_str = f", uniform_acc={bt_entry['uniform_pairwise_acc']:.4f}" if bt_entry.get('uniform_pairwise_acc') is not None else ""
                print(f"  BT cv_acc={bt_entry['cv_accuracy_mean']:.4f} "
                      f"(best l2={bt_entry['best_l2_lambda']:.4g}, {bt_entry['n_pairs']} pairs){uni_str}")

        del activations
        gc.collect()

    # Summary
    ridge_probes = [p for p in manifest["probes"] if p["method"] == "ridge"]
    bt_probes = [p for p in manifest["probes"] if p["method"] == "bradley_terry"]

    if ridge_probes:
        best = max(ridge_probes, key=lambda x: x["final_r"] or -1)
        r_str = f"{best['final_r']:.4f}" if best['final_r'] is not None else "N/A"
        uni_str = ""
        if any(p.get("uniform_pairwise_acc") is not None for p in ridge_probes):
            best_uni = max(ridge_probes, key=lambda x: x.get("uniform_pairwise_acc") or -1)
            uni_str = f", best uniform_acc={best_uni['uniform_pairwise_acc']:.4f} (L{best_uni['layer']})"
        print(f"\nBest Ridge: layer {best['layer']} (heldout r={r_str}){uni_str}")
    if bt_probes:
        best = max(bt_probes, key=lambda x: x["cv_accuracy_mean"])
        uni_str = ""
        if any(p.get("uniform_pairwise_acc") is not None for p in bt_probes):
            best_uni = max(bt_probes, key=lambda x: x.get("uniform_pairwise_acc") or -1)
            uni_str = f", best uniform_acc={best_uni['uniform_pairwise_acc']:.4f} (L{best_uni['layer']})"
        print(f"Best BT: layer {best['layer']} (cv_acc={best['cv_accuracy_mean']:.4f}, "
              f"l2={best['best_l2_lambda']:.4g}){uni_str}")

    save_manifest(manifest, config.output_dir)
    print(f"\nSaved manifest with {len(manifest['probes'])} probes")

    return manifest


def run_hoo(config: RunDirProbeConfig) -> dict:
    """Held-one-out training and evaluation by group (topic or dataset)."""
    assert config.hoo_grouping is not None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_ridge = ProbeMode.RIDGE in config.modes
    run_bt = ProbeMode.BRADLEY_TERRY in config.modes

    print(f"HOO Probes: {config.experiment_name}")
    print(f"Grouping: {config.hoo_grouping}, hold_out_size: {config.hoo_hold_out_size}")
    print(f"Modes: {[m.value for m in config.modes]}")
    print(f"Run dir: {config.run_dir}")
    print(f"Output: {config.output_dir}")

    # Load scores and measurements
    scores = load_thurstonian_scores(config.run_dir) if run_ridge else {}
    measurements = load_pairwise_measurements(config.run_dir)
    if scores:
        print(f"\nLoaded {len(scores)} task scores")
    if measurements:
        print(f"Loaded {len(measurements)} pairwise comparisons")

    # Load uniform eval data
    uniform_eval_measurements = load_pairwise_measurements(config.uniform_eval_run_dir) if config.uniform_eval_run_dir else []
    if uniform_eval_measurements:
        uniform_tids = _task_ids_from_measurements(uniform_eval_measurements)
        train_tids = set(scores.keys()) if scores else _task_ids_from_measurements(measurements)
        overlap = uniform_tids & train_tids
        if overlap:
            raise ValueError(
                f"Uniform eval has {len(overlap)} tasks that overlap with training data."
            )
        print(f"  Uniform eval: {len(uniform_eval_measurements)} measurements ({len(uniform_tids)} tasks)")

    # Collect all task IDs from scores and/or measurements
    all_task_ids = set(scores.keys())
    for m in measurements:
        all_task_ids.add(m.task_a.id)
        all_task_ids.add(m.task_b.id)
    if uniform_eval_measurements:
        all_task_ids |= _task_ids_from_measurements(uniform_eval_measurements)

    # Build task_id -> group mapping
    task_groups = build_task_groups(
        task_ids=all_task_ids,
        grouping=config.hoo_grouping,
        topics_json=config.topics_json,
    )
    # Keep only tasks that have group labels (and scores if running ridge)
    if run_ridge:
        scored_and_grouped = set(scores.keys()) & set(task_groups.keys())
        n_dropped = len(scores) - len(scored_and_grouped)
    else:
        scored_and_grouped = all_task_ids & set(task_groups.keys())
        n_dropped = len(all_task_ids) - len(scored_and_grouped)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} tasks without group metadata")

    all_groups = sorted(set(task_groups[tid] for tid in scored_and_grouped))
    if config.hoo_groups is not None:
        all_groups = [g for g in all_groups if g in config.hoo_groups]
    print(f"  Groups ({len(all_groups)}): {all_groups}")

    group_sizes = {}
    for g in all_groups:
        group_sizes[g] = sum(1 for tid in scored_and_grouped if task_groups[tid] == g)
    for g, n in sorted(group_sizes.items(), key=lambda x: -x[1]):
        print(f"    {g}: {n}")

    if config.hoo_hold_out_size >= len(all_groups):
        raise ValueError(
            f"hoo_hold_out_size ({config.hoo_hold_out_size}) must be < number of groups ({len(all_groups)})"
        )

    # Pre-load all activations (scored+grouped train tasks + eval tasks for alpha selection)
    eval_scores_for_alpha, _ = load_eval_data(
        config.eval_run_dir, set(scores.keys()),
        demean_confounds=config.demean_confounds,
        topics_json=config.topics_json,
    ) if run_ridge else ({}, [])
    activation_filter = scored_and_grouped | set(eval_scores_for_alpha.keys())
    if uniform_eval_measurements:
        activation_filter = activation_filter | _task_ids_from_measurements(uniform_eval_measurements)
    task_ids_arr, activations = load_activations(
        config.activations_path,
        task_id_filter=activation_filter,
        layers=config.layers,
    )
    print(f"  {len(task_ids_arr)} tasks with activations")

    # Build full pairwise data (used by both BT and Ridge for pairwise accuracy)
    bt_data = None
    if measurements:
        bt_data = PairwiseActivationData.from_measurements(measurements, task_ids_arr, activations)
        print(f"  {len(bt_data.pairs)} unique pairs ({bt_data.n_measurements} measurements)")

    # Build uniform eval pairwise data
    uniform_bt_data = None
    if uniform_eval_measurements:
        uniform_bt_data = PairwiseActivationData.from_measurements(
            uniform_eval_measurements, task_ids_arr, activations,
        )
        print(f"  Uniform eval: {len(uniform_bt_data.pairs)} unique pairs")

    folds = list(combinations(all_groups, config.hoo_hold_out_size))
    print(f"\n{len(folds)} folds\n")

    # Pre-select alpha using heldout eval set
    heldout_alpha: float | None = None
    if run_ridge:
        print("Selecting alpha from heldout eval set...")
        # Use first layer to sweep alpha (alpha transfers across layers)
        layer0 = config.layers[0]
        train_indices, y_train = hoo_ridge.build_ridge_xy(task_ids_arr, scores)
        X_train = activations[layer0][train_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        eval_indices, y_eval = hoo_ridge.build_ridge_xy(task_ids_arr, eval_scores_for_alpha)
        X_eval = activations[layer0][eval_indices]
        X_eval_scaled = scaler.transform(X_eval)

        alphas = np.logspace(-1, 5, config.alpha_sweep_size)
        best_r = -1.0
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_eval_scaled)
            r, _ = pearsonr(y_eval, y_pred)
            if r > best_r:
                best_r = float(r)
                heldout_alpha = float(alpha)
        print(f"  Best alpha from heldout: {heldout_alpha:.4g} (r={best_r:.4f})")

    # Method factory configs: (name, factory_fn, kwargs)
    methods_config = []
    if run_ridge:
        methods_config.append(("ridge", hoo_ridge.make_method, {
            "config": config,
            "task_ids": task_ids_arr,
            "activations": activations,
            "scores": scores,
            "task_groups": task_groups,
            "scored_and_grouped": scored_and_grouped,
            "bt_data": bt_data,
        }))
    if run_bt and bt_data is not None:
        methods_config.append(("bradley_terry", hoo_bt.make_method, {
            "bt_data": bt_data,
            "task_ids": task_ids_arr,
            "task_groups": task_groups,
        }))

    best_hps: dict[str, float | None] = {name: None for name, _, _ in methods_config}
    if heldout_alpha is not None:
        best_hps["ridge"] = heldout_alpha
    all_fold_results = []

    for fold_idx, held_out in enumerate(folds):
        held_out_set = set(held_out)
        train_groups = [g for g in all_groups if g not in held_out_set]

        # Split uniform eval into within-held-out pairs (cross-topic) and all pairs (test-set)
        fold_uniform_hoo = None
        if uniform_bt_data is not None and len(uniform_bt_data.pairs) > 0:
            _, fold_uniform_hoo = uniform_bt_data.split_by_groups(
                task_ids_arr, task_groups, held_out_set,
            )

        fold_metrics = {
            "fold_idx": fold_idx,
            "held_out_groups": list(held_out),
            "train_groups": train_groups,
            "layers": {},
        }

        for name, factory, kwargs in methods_config:
            method = factory(
                fold_idx=fold_idx,
                held_out_set=held_out_set,
                best_hp=best_hps[name],
                **kwargs,
            )
            if method is None:
                continue

            for layer in config.layers:
                weights, method.best_hp = method.train(layer, method.best_hp)
                probe_id = f"hoo_fold{fold_idx}_{method.name}_L{layer:02d}"
                save_probe(weights, config.output_dir, probe_id)
                metrics = method.evaluate(layer, weights)
                # Test-set uniform eval (all pairs)
                if uniform_bt_data is not None and len(uniform_bt_data.pairs) > 0:
                    predicted = score_with_probe(weights, activations[layer])
                    metrics["uniform_pairwise_acc"] = pairwise_accuracy_from_scores(
                        predicted, uniform_bt_data,
                    )
                # Cross-topic uniform eval (within held-out topic pairs only)
                if fold_uniform_hoo is not None and len(fold_uniform_hoo.pairs) > 0:
                    predicted = score_with_probe(weights, activations[layer])
                    metrics["uniform_hoo_acc"] = pairwise_accuracy_from_scores(
                        predicted, fold_uniform_hoo,
                    )
                    metrics["uniform_hoo_n_pairs"] = len(fold_uniform_hoo.pairs)
                metrics.update(method=method.name, probe_id=probe_id, layer=layer)
                fold_metrics["layers"][f"{method.name}_L{layer}"] = metrics

            best_hps[name] = method.best_hp

        all_fold_results.append(fold_metrics)

    # Build summary
    summary = {
        "experiment_name": config.experiment_name,
        "created_at": datetime.now().isoformat(),
        "grouping": config.hoo_grouping,
        "hold_out_size": config.hoo_hold_out_size,
        "all_groups": all_groups,
        "group_sizes": group_sizes,
        "n_folds": len(all_fold_results),
        "layers": config.layers,
        "folds": all_fold_results,
    }
    if config.uniform_eval_run_dir is not None:
        summary["uniform_eval_run_dir"] = str(config.uniform_eval_run_dir)

    # Aggregate across folds per layer
    def _collect(key: str, field: str) -> list:
        return [
            f["layers"][key][field]
            for f in all_fold_results
            if key in f["layers"] and f["layers"][key].get(field) is not None
        ]

    def _collect_with_n(key: str, field: str) -> tuple[list, list]:
        values, weights = [], []
        for f in all_fold_results:
            if key in f["layers"] and f["layers"][key].get(field) is not None:
                values.append(f["layers"][key][field])
                weights.append(f["layers"][key]["hoo_n_samples"])
        return values, weights

    layer_summary = {}
    for layer in config.layers:
        entry = {}
        if run_ridge:
            k = f"ridge_L{layer}"
            hoo_rs, hoo_ns = _collect_with_n(k, "hoo_r")
            hoo_accs = _collect(k, "hoo_acc")
            ridge_entry = {
                "mean_hoo_r": float(np.average(hoo_rs, weights=hoo_ns)) if hoo_rs else None,
                "std_hoo_r": float(np.std(hoo_rs)) if hoo_rs else None,
                "n_folds": len(hoo_rs),
            }
            if hoo_accs:
                ridge_entry["mean_hoo_acc"] = float(np.mean(hoo_accs))
                ridge_entry["std_hoo_acc"] = float(np.std(hoo_accs))
            uni_accs = _collect(k, "uniform_pairwise_acc")
            if uni_accs:
                ridge_entry["mean_uniform_acc"] = float(np.mean(uni_accs))
                ridge_entry["std_uniform_acc"] = float(np.std(uni_accs))
            uni_hoo_accs = _collect(k, "uniform_hoo_acc")
            if uni_hoo_accs:
                ridge_entry["mean_uniform_hoo_acc"] = float(np.mean(uni_hoo_accs))
                ridge_entry["std_uniform_hoo_acc"] = float(np.std(uni_hoo_accs))
            entry["ridge"] = ridge_entry
        if run_bt:
            k = f"bradley_terry_L{layer}"
            val_accs = _collect(k, "val_acc")
            hoo_accs = _collect(k, "hoo_acc")
            bt_uni_accs = _collect(k, "uniform_pairwise_acc")
            bt_summary = {
                "mean_val_acc": float(np.mean(val_accs)) if val_accs else None,
                "mean_hoo_acc": float(np.mean(hoo_accs)) if hoo_accs else None,
                "std_hoo_acc": float(np.std(hoo_accs)) if hoo_accs else None,
                "n_folds": len(hoo_accs),
            }
            if bt_uni_accs:
                bt_summary["mean_uniform_acc"] = float(np.mean(bt_uni_accs))
                bt_summary["std_uniform_acc"] = float(np.std(bt_uni_accs))
            bt_uni_hoo_accs = _collect(k, "uniform_hoo_acc")
            if bt_uni_hoo_accs:
                bt_summary["mean_uniform_hoo_acc"] = float(np.mean(bt_uni_hoo_accs))
                bt_summary["std_uniform_hoo_acc"] = float(np.std(bt_uni_hoo_accs))
            entry["bradley_terry"] = bt_summary
        layer_summary[layer] = entry
    summary["layer_summary"] = {str(k): v for k, v in layer_summary.items()}

    # Print summary
    print("\n" + "=" * 60)
    print("HOO Summary")
    print("=" * 60)
    for layer in config.layers:
        ls = layer_summary[layer]
        if "ridge" in ls and ls["ridge"]["mean_hoo_r"] is not None:
            r = ls["ridge"]
            acc_str = f", hoo_acc={r['mean_hoo_acc']:.4f}" if r.get("mean_hoo_acc") is not None else ""
            uni_str = f", uniform_acc={r['mean_uniform_acc']:.4f}" if r.get("mean_uniform_acc") is not None else ""
            uni_hoo_str = f", uniform_hoo_acc={r['mean_uniform_hoo_acc']:.4f}" if r.get("mean_uniform_hoo_acc") is not None else ""
            print(f"  Ridge L{layer}: hoo_r={r['mean_hoo_r']:.4f} +/- {r['std_hoo_r']:.4f}{acc_str}{uni_str}{uni_hoo_str} "
                  f"({r['n_folds']} folds)")
        if "bradley_terry" in ls and ls["bradley_terry"]["mean_hoo_acc"] is not None:
            b = ls["bradley_terry"]
            uni_str = f", uniform_acc={b['mean_uniform_acc']:.4f}" if b.get("mean_uniform_acc") is not None else ""
            uni_hoo_str = f", uniform_hoo_acc={b['mean_uniform_hoo_acc']:.4f}" if b.get("mean_uniform_hoo_acc") is not None else ""
            print(f"  BT    L{layer}: hoo_acc={b['mean_hoo_acc']:.4f} +/- {b['std_hoo_acc']:.4f}{uni_str}{uni_hoo_str} "
                  f"({b['n_folds']} folds)")

    # Save
    summary_path = config.output_dir / "hoo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")

    plot_hoo_summary(summary, config.output_dir)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes from run directory results")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = RunDirProbeConfig.from_yaml(args.config)

    if config.hoo_grouping is not None:
        run_hoo(config)
    else:
        run_probes(config)


if __name__ == "__main__":
    main()
