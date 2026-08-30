from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.core.evaluate import evaluate_probe_on_data
from src.probes.experiments.hoo_method import HooMethod
from src.probes.residualization import demean_scores

if TYPE_CHECKING:
    from src.probes.experiments.run_dir_probes import RunDirProbeConfig


def build_ridge_xy(
    task_ids: np.ndarray,
    scores: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map scores to activation indices, returning (indices, y)."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    valid_indices = []
    valid_scores = []
    for task_id, score in scores.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)
    indices = np.array(valid_indices)
    y = np.array(valid_scores)
    if not np.isfinite(y).all():
        bad_count = (~np.isfinite(y)).sum()
        raise ValueError(f"Found {bad_count} non-finite values in scores")
    return indices, y


def make_method(
    fold_idx: int,
    config: RunDirProbeConfig,
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float],
    task_groups: dict[str, str],
    scored_and_grouped: set[str],
    held_out_set: set[str],
    best_hp: float | None,
    bt_data: PairwiseActivationData | None = None,
) -> HooMethod | None:
    """Build a Ridge HooMethod for one fold, closing over train/eval data."""
    train_scores = {
        tid: scores[tid]
        for tid in scored_and_grouped
        if task_groups[tid] not in held_out_set
    }
    eval_scores = {
        tid: scores[tid]
        for tid in scored_and_grouped
        if task_groups[tid] in held_out_set
    }

    if len(train_scores) < 20 or len(eval_scores) < 10:
        print(f"Fold {fold_idx}: skip ridge (train={len(train_scores)}, eval={len(eval_scores)})")
        return None

    eval_label = ", ".join(sorted(held_out_set))
    print(f"Fold {fold_idx}: hold out [{eval_label}] "
          f"(train={len(train_scores)}, eval={len(eval_scores)})")

    # Split pairwise data for hoo_acc evaluation
    eval_bt_data = None
    if bt_data is not None:
        _, eval_bt_data = bt_data.split_by_groups(task_ids, task_groups, held_out_set)

    if config.demean_confounds:
        assert config.topics_json is not None
        train_scores, res_stats = demean_scores(
            train_scores, config.topics_json, confounds=config.demean_confounds,
        )
        print(f"  Demeaned train (R²={res_stats['metadata_r2']:.4f})")

    indices, y = build_ridge_xy(task_ids, train_scores)
    eval_task_ids_list = list(eval_scores.keys())
    eval_scores_arr = np.array([eval_scores[tid] for tid in eval_task_ids_list])

    def train(layer: int, hp: float | None) -> tuple[np.ndarray, float | None]:
        X = activations[layer][indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert hp is not None, "best_hp must be pre-selected from heldout eval set"
        probe = Ridge(alpha=hp)
        probe.fit(X_scaled, y)
        coef_raw = probe.coef_ / scaler.scale_
        intercept_raw = probe.intercept_ - coef_raw @ scaler.mean_
        weights = np.append(coef_raw, intercept_raw)
        return weights, hp

    def evaluate(layer: int, weights: np.ndarray) -> dict:
        eval_result = evaluate_probe_on_data(
            probe_weights=weights,
            activations=activations[layer],
            scores=eval_scores_arr,
            task_ids_data=task_ids,
            task_ids_scores=eval_task_ids_list,
            pairwise_data=eval_bt_data,
        )

        hoo_r = eval_result["pearson_r"]
        hoo_r_str = f"{hoo_r:.4f}" if hoo_r is not None else "N/A"
        hoo_acc = eval_result.get("pairwise_acc")
        hoo_acc_str = f", hoo_acc={hoo_acc:.4f}" if hoo_acc is not None else ""
        print(f"  Ridge L{layer}: hoo_r={hoo_r_str}{hoo_acc_str}")

        result = {
            "best_alpha": best_hp,
            "hoo_r2": eval_result["r2"],
            "hoo_r": hoo_r,
            "hoo_n_samples": eval_result["n_samples"],
            "n_train": len(train_scores),
            "n_eval": len(eval_scores),
            "demean_confounds": config.demean_confounds,
        }
        if hoo_acc is not None:
            result["hoo_acc"] = hoo_acc
        return result

    return HooMethod(name="ridge", train=train, evaluate=evaluate, best_hp=best_hp)
