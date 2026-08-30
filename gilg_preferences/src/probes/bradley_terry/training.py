from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .data import PairwiseActivationData


@dataclass
class BTResult:
    weights: np.ndarray  # [coef..., 0.0] — no intercept, trailing zero for format compat
    layer: int
    train_accuracy: float
    train_loss: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    best_l2_lambda: float
    n_iterations: int
    lambda_sweep: list[dict]


def _bt_loss_and_grad(
    w: np.ndarray,
    acts: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    wins_i: np.ndarray,
    wins_j: np.ndarray,
    total_weight: float,
    l2_lambda: float,
) -> tuple[float, np.ndarray]:
    """Weighted BT loss on aggregated pairs.

    For pair (i,j) with logit = score_i - score_j:
      loss = wins_i * log(1+exp(-logit)) + wins_j * log(1+exp(logit))
    """
    task_scores = acts @ w
    logits = task_scores[idx_i] - task_scores[idx_j]

    # log(1+exp(-x)) and log(1+exp(x))
    loss_ij = wins_i * np.logaddexp(0, -logits) + wins_j * np.logaddexp(0, logits)
    loss = np.sum(loss_ij) / total_weight + 0.5 * l2_lambda * np.sum(w ** 2)

    # Gradient: d/d(logit) = wins_i*(sigmoid-1) + wins_j*sigmoid = sigmoid*(wins_i+wins_j) - wins_i
    sigmoid = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    pair_grad = (sigmoid * (wins_i + wins_j) - wins_i) / total_weight

    n_tasks = len(acts)
    task_grad = (
        np.bincount(idx_i, weights=pair_grad, minlength=n_tasks)
        - np.bincount(idx_j, weights=pair_grad, minlength=n_tasks)
    )

    grad_w = acts.T @ task_grad + l2_lambda * w
    return loss, grad_w


def _fit_bt(
    acts: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    wins_i: np.ndarray,
    wins_j: np.ndarray,
    total_weight: float,
    l2_lambda: float,
    maxiter: int = 500,
) -> tuple[np.ndarray, int]:
    """Fit BT weights via L-BFGS-B. Returns (weights, n_iterations)."""
    w0 = np.zeros(acts.shape[1])
    n_iterations = 0

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal n_iterations
        n_iterations += 1
        return _bt_loss_and_grad(w, acts, idx_i, idx_j, wins_i, wins_j, total_weight, l2_lambda)

    result = minimize(objective, w0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter})
    return result.x, n_iterations


def weighted_accuracy(
    w: np.ndarray,
    acts: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    wins_i: np.ndarray,
    wins_j: np.ndarray,
    total_weight: float,
) -> float:
    """Weighted pairwise accuracy: fraction of individual comparisons correctly predicted."""
    task_scores = acts @ w
    logits = task_scores[idx_i] - task_scores[idx_j]
    # i wins when logit > 0, j wins when logit < 0
    correct = np.where(logits > 0, wins_i, wins_j)
    return float(np.sum(correct) / total_weight)


def _sweep_one_lambda(
    acts: np.ndarray,
    train_idx_i: np.ndarray,
    train_idx_j: np.ndarray,
    train_wins_i: np.ndarray,
    train_wins_j: np.ndarray,
    train_total: float,
    val_idx_i: np.ndarray,
    val_idx_j: np.ndarray,
    val_wins_i: np.ndarray,
    val_wins_j: np.ndarray,
    val_total: float,
    l2: float,
) -> dict:
    w, _ = _fit_bt(acts, train_idx_i, train_idx_j, train_wins_i, train_wins_j, train_total, l2)
    train_acc = weighted_accuracy(w, acts, train_idx_i, train_idx_j, train_wins_i, train_wins_j, train_total)
    val_acc = weighted_accuracy(w, acts, val_idx_i, val_idx_j, val_wins_i, val_wins_j, val_total)
    return {"l2_lambda": float(l2), "train_accuracy": train_acc, "val_accuracy": val_acc}


def train_bt(
    data: PairwiseActivationData,
    layer: int,
    lambdas: np.ndarray | None = None,
    val_fraction: float = 0.2,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> BTResult:
    if lambdas is None:
        lambdas = np.logspace(-1, 3, 5)
    if rng is None:
        rng = np.random.default_rng(0)

    acts = data.activations[layer]
    idx_i = data.pairs[:, 0]
    idx_j = data.pairs[:, 1]
    wins_i = data.wins_i
    wins_j = data.total - data.wins_i
    n_unique = len(data.pairs)

    # Single train/val split on unique pairs
    val_mask = rng.random(n_unique) < val_fraction
    train_mask = ~val_mask

    train_i, train_j = idx_i[train_mask], idx_j[train_mask]
    train_wi, train_wj = wins_i[train_mask], wins_j[train_mask]
    train_total = float(np.sum(train_wi + train_wj))

    val_i, val_j = idx_i[val_mask], idx_j[val_mask]
    val_wi, val_wj = wins_i[val_mask], wins_j[val_mask]
    val_total = float(np.sum(val_wi + val_wj))

    # Lambda sweep
    sweep_args = (acts, train_i, train_j, train_wi, train_wj, train_total,
                  val_i, val_j, val_wi, val_wj, val_total)
    if n_jobs == 1:
        lambda_sweep = [_sweep_one_lambda(*sweep_args, l2) for l2 in lambdas]
    else:
        lambda_sweep = Parallel(n_jobs=n_jobs)(
            delayed(_sweep_one_lambda)(*sweep_args, l2) for l2 in lambdas
        )
    for entry in lambda_sweep:
        print(f"    l2={entry['l2_lambda']:.4g}: train_acc={entry['train_accuracy']:.4f}, val_acc={entry['val_accuracy']:.4f}")

    best_entry = max(lambda_sweep, key=lambda x: x["val_accuracy"])
    best_l2 = best_entry["l2_lambda"]

    # Retrain on all data with best lambda
    all_total = float(np.sum(wins_i + wins_j))
    w, n_iterations = _fit_bt(acts, idx_i, idx_j, wins_i, wins_j, all_total, best_l2)
    train_acc = weighted_accuracy(w, acts, idx_i, idx_j, wins_i, wins_j, all_total)

    task_scores = acts @ w
    logits = task_scores[idx_i] - task_scores[idx_j]
    train_loss = float(np.sum(wins_i * np.logaddexp(0, -logits) + wins_j * np.logaddexp(0, logits)) / all_total)

    return BTResult(
        weights=np.append(w, 0.0),
        layer=layer,
        train_accuracy=train_acc,
        train_loss=train_loss,
        cv_accuracy_mean=best_entry["val_accuracy"],
        cv_accuracy_std=0.0,
        best_l2_lambda=best_l2,
        n_iterations=n_iterations,
        lambda_sweep=lambda_sweep,
    )


def pairwise_accuracy_from_scores(
    predicted_scores: np.ndarray,
    data: PairwiseActivationData,
) -> float:
    """Compute weighted pairwise accuracy of scalar scores on aggregated BT pairs.

    Args:
        predicted_scores: (n_tasks,) array aligned to the activation array indexing.
            Pairs in data.pairs index into this array.
        data: aggregated pairwise comparison data
    """
    scores_i = predicted_scores[data.pairs[:, 0]]
    scores_j = predicted_scores[data.pairs[:, 1]]
    wins_j = data.total - data.wins_i

    correct = np.where(scores_i > scores_j, data.wins_i, wins_j)
    return float(np.sum(correct) / data.n_measurements)


def train_bt_fixed_lambda(
    data: PairwiseActivationData,
    layer: int,
    l2_lambda: float,
) -> BTResult:
    """Train BT at a fixed lambda (no sweep)."""
    acts = data.activations[layer]
    idx_i = data.pairs[:, 0]
    idx_j = data.pairs[:, 1]
    wins_i = data.wins_i
    wins_j = data.total - data.wins_i
    all_total = float(np.sum(wins_i + wins_j))

    w, n_iterations = _fit_bt(acts, idx_i, idx_j, wins_i, wins_j, all_total, l2_lambda)
    train_acc = weighted_accuracy(w, acts, idx_i, idx_j, wins_i, wins_j, all_total)

    task_scores = acts @ w
    logits = task_scores[idx_i] - task_scores[idx_j]
    train_loss = float(np.sum(wins_i * np.logaddexp(0, -logits) + wins_j * np.logaddexp(0, logits)) / all_total)

    return BTResult(
        weights=np.append(w, 0.0),
        layer=layer,
        train_accuracy=train_acc,
        train_loss=train_loss,
        cv_accuracy_mean=train_acc,
        cv_accuracy_std=0.0,
        best_l2_lambda=l2_lambda,
        n_iterations=n_iterations,
        lambda_sweep=[],
    )


