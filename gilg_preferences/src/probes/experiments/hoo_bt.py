from __future__ import annotations

import numpy as np

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import train_bt, train_bt_fixed_lambda, weighted_accuracy
from src.probes.experiments.hoo_method import HooMethod


def make_method(
    fold_idx: int,
    bt_data: PairwiseActivationData,
    task_ids: np.ndarray,
    task_groups: dict[str, str],
    held_out_set: set[str],
    best_hp: float | None,
) -> HooMethod | None:
    """Build a BT HooMethod for one fold, closing over train/eval data."""
    train_bt_data, eval_bt_data = bt_data.split_by_groups(
        task_ids, task_groups, held_out_set,
    )
    n_train_pairs = len(train_bt_data.pairs)
    n_eval_pairs = len(eval_bt_data.pairs)

    if n_train_pairs < 50 or n_eval_pairs < 10:
        print(f"Fold {fold_idx}: skip BT (train={n_train_pairs}, eval={n_eval_pairs} pairs)")
        return None

    print(f"  BT: {n_train_pairs} train pairs, {n_eval_pairs} eval pairs")

    # Shared state: train stores (cv_accuracy, best_hp) for evaluate to read
    last_train_results: dict[int, tuple[float, float]] = {}

    def train(layer: int, hp: float | None) -> tuple[np.ndarray, float | None]:
        if hp is None:
            bt_result = train_bt(train_bt_data, layer)
            hp = bt_result.best_l2_lambda
            print(f"  BT lambda sweep: best_lambda={hp:.4g}")
        else:
            bt_result = train_bt_fixed_lambda(train_bt_data, layer, hp)
        last_train_results[layer] = (bt_result.cv_accuracy_mean, hp)
        return bt_result.weights, hp

    def evaluate(layer: int, weights: np.ndarray) -> dict:
        assert layer in last_train_results, "train() must be called before evaluate()"
        w = weights[:-1]  # strip trailing zero
        eval_acts = eval_bt_data.activations[layer]
        eval_idx_i = eval_bt_data.pairs[:, 0]
        eval_idx_j = eval_bt_data.pairs[:, 1]
        eval_wins_i = eval_bt_data.wins_i
        eval_wins_j = eval_bt_data.total - eval_bt_data.wins_i
        eval_total = float(np.sum(eval_wins_i + eval_wins_j))

        hoo_acc = weighted_accuracy(
            w, eval_acts, eval_idx_i, eval_idx_j,
            eval_wins_i, eval_wins_j, eval_total,
        )
        val_acc, hp = last_train_results[layer]

        print(f"  BT L{layer}: val_acc={val_acc:.4f}, hoo_acc={hoo_acc:.4f}")

        return {
            "val_acc": val_acc,
            "best_l2_lambda": hp,
            "hoo_acc": hoo_acc,
            "hoo_n_pairs": n_eval_pairs,
            "train_n_pairs": n_train_pairs,
        }

    return HooMethod(name="bradley_terry", train=train, evaluate=evaluate, best_hp=best_hp)
