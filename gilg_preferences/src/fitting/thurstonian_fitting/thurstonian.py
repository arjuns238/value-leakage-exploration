"""Thurstonian model for pairwise preference data.

Fits utility values μ and uncertainty σ for each item based on
pairwise comparison outcomes. Uses the model:

    U(i) ~ N(μ_i, σ_i²)
    P(i ≻ j) = Φ((μ_i - μ_j) / √(σ_i² + σ_j²))

where Φ is the standard normal CDF.

NOTE: Regularization on σ is not a current focus - using simple L2 on log(σ).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from collections import Counter, defaultdict

import autograd.numpy as anp
import numpy as np
import yaml
from autograd import grad
from autograd.scipy.stats import norm as anorm
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm

from src.task_data import Task
from src.types import BinaryPreferenceMeasurement

# Tuned defaults (see data_analysis/tuning.md)
DEFAULT_MU_BOUNDS = (-10.0, 10.0)
DEFAULT_LOG_SIGMA_BOUNDS = (-2.0, 2.0)
DEFAULT_SIGMA_INIT = 1.0
DEFAULT_LAMBDA_SIGMA = 1.0


class PairwiseData:
    """Pairwise comparison data stored in sparse COO format internally.

    Accepts a dense wins matrix for backwards compatibility, but stores
    only the nonzero entries as (row, col, count) arrays. The dense .wins
    matrix is reconstructed on demand.
    """

    def __init__(
        self,
        tasks: list[Task],
        wins: np.ndarray | None = None,
        *,
        _row: np.ndarray | None = None,
        _col: np.ndarray | None = None,
        _count: np.ndarray | None = None,
    ) -> None:
        self.tasks = tasks
        self._id_to_idx = {t.id: i for i, t in enumerate(tasks)}

        if _row is not None:
            self._row = _row.astype(np.int32)
            self._col = _col.astype(np.int32)
            self._count = _count.astype(np.int32)
        else:
            row, col = np.nonzero(wins)
            self._row = row.astype(np.int32)
            self._col = col.astype(np.int32)
            self._count = wins[row, col].astype(np.int32)
        self._n_comparisons = int(self._count.sum())

    @property
    def wins(self) -> np.ndarray:
        """Reconstruct dense wins matrix. O(n^2) memory — avoid for large n."""
        n = len(self.tasks)
        w = np.zeros((n, n), dtype=np.int32)
        w[self._row, self._col] = self._count
        return w

    def index_of(self, task: Task) -> int:
        return self._id_to_idx[task.id]

    def total_comparisons(self, task: Task) -> int:
        idx = self.index_of(task)
        as_winner = int(self._count[self._row == idx].sum())
        as_loser = int(self._count[self._col == idx].sum())
        return as_winner + as_loser

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_comparisons(self) -> int:
        return self._n_comparisons

    @classmethod
    def from_comparisons(
        cls,
        comparisons: list[BinaryPreferenceMeasurement],
        tasks: list[Task],
    ) -> PairwiseData:
        id_to_idx = {t.id: i for i, t in enumerate(tasks)}
        pair_counts: dict[tuple[int, int], int] = {}

        for c in comparisons:
            i, j = id_to_idx[c.task_a.id], id_to_idx[c.task_b.id]
            if c.choice == "a":
                key = (i, j)
            else:
                key = (j, i)
            pair_counts[key] = pair_counts.get(key, 0) + 1

        if pair_counts:
            keys = list(pair_counts.keys())
            row = np.array([k[0] for k in keys], dtype=np.int32)
            col = np.array([k[1] for k in keys], dtype=np.int32)
            count = np.array([pair_counts[k] for k in keys], dtype=np.int32)
        else:
            row = np.array([], dtype=np.int32)
            col = np.array([], dtype=np.int32)
            count = np.array([], dtype=np.int32)

        return cls(tasks, _row=row, _col=col, _count=count)


@dataclass
class OptimizationHistory:
    """Minimal optimization progress tracking."""

    loss: list[float] = field(default_factory=list)
    sigma_max: list[float] = field(default_factory=list)


@dataclass
class ThurstonianResult:
    """Thurstonian fitting result. Implements UtilityResult protocol."""

    tasks: list["Task"]
    mu: np.ndarray
    sigma: np.ndarray
    converged: bool
    neg_log_likelihood: float
    n_iterations: int
    n_function_evals: int
    termination_message: str
    gradient_norm: float | None
    history: OptimizationHistory
    n_comparisons: int = 0
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {t.id: i for i, t in enumerate(self.tasks)}

    @property
    def n_observations(self) -> int:
        return self.n_comparisons

    def utility(self, task: "Task") -> float:
        return float(self.mu[self._id_to_idx[task.id]])

    def uncertainty(self, task: "Task") -> float:
        return float(self.sigma[self._id_to_idx[task.id]])

    def preference_probability(self, task_a: "Task", task_b: "Task") -> float:
        """P(task_a ≻ task_b) under the fitted model."""
        i = self._id_to_idx[task_a.id]
        j = self._id_to_idx[task_b.id]
        return _preference_prob(self.mu[i], self.mu[j], self.sigma[i], self.sigma[j])

    def ranking(self) -> list["Task"]:
        """Tasks sorted by utility, highest first."""
        order = np.argsort(-self.mu)
        return [self.tasks[i] for i in order]

    def to_dict(self) -> dict:
        return {
            "task_ids": [t.id for t in self.tasks],
            "mu": {t.id: float(self.mu[i]) for i, t in enumerate(self.tasks)},
            "sigma": {t.id: float(self.sigma[i]) for i, t in enumerate(self.tasks)},
            "n_comparisons": self.n_comparisons,
            "converged": self.converged,
            "neg_log_likelihood": float(self.neg_log_likelihood),
        }

    def normalized_utility(self, task: "Task") -> float:
        """Average probability of being preferred over all other tasks (0.5 = average)."""
        probs = [
            self.preference_probability(task, other)
            for other in self.tasks
            if other.id != task.id
        ]
        if not probs:
            raise ValueError("Cannot compute normalized utility with fewer than 2 tasks")
        return sum(probs) / len(probs)


def _preference_prob(mu_i: float, mu_j: float, sigma_i: float, sigma_j: float) -> float:
    """P(i ≻ j) using the difference distribution."""
    scale = np.sqrt(sigma_i**2 + sigma_j**2)
    return float(norm.sf(0, loc=mu_i - mu_j, scale=scale))


def _neg_log_likelihood(
    params: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    count: np.ndarray,
    n: int,
) -> float:
    """Sparse NLL. params: [μ_1..μ_{n-1}, log(σ_0)..log(σ_{n-1})]. μ_0 fixed to 0."""
    mu = np.zeros(n)
    mu[1:] = params[: n - 1]
    sigma = np.exp(params[n - 1:])

    mu_diff = mu[row] - mu[col]
    scale = np.sqrt(sigma[row]**2 + sigma[col]**2)
    p = ndtr(mu_diff / scale)
    p = np.clip(p, 1e-10, 1 - 1e-10)

    return -float(np.sum(count * np.log(p)))


def _neg_log_likelihood_autograd(
    params: anp.ndarray,
    row: anp.ndarray,
    col: anp.ndarray,
    count: anp.ndarray,
    n: int,
    lambda_sigma: float = 0.0,
) -> float:
    """Autograd-compatible sparse NLL for computing analytical gradients."""
    mu = anp.concatenate([anp.zeros(1), params[: n - 1]])
    sigma = anp.exp(params[n - 1 :])

    mu_diff = mu[row] - mu[col]
    scale = anp.sqrt(sigma[row] ** 2 + sigma[col] ** 2)
    z = mu_diff / scale
    p = anorm.cdf(z)
    p = anp.clip(p, 1e-10, 1 - 1e-10)

    nll = -anp.sum(count * anp.log(p))
    if lambda_sigma > 0:
        log_sigma = params[n - 1 :]
        nll = nll + lambda_sigma * anp.sum(log_sigma**2)
    return nll


def _make_objective_and_grad(
    row: np.ndarray, col: np.ndarray, count: np.ndarray,
    n: int, lambda_sigma: float,
):
    """Create objective function and its gradient using autograd."""

    def objective(params):
        return _neg_log_likelihood_autograd(params, row, col, count, n, lambda_sigma)

    gradient_fn = grad(objective)

    def objective_and_grad(params):
        return float(objective(params)), np.asarray(gradient_fn(params))

    return objective_and_grad


def fit_thurstonian(
    data: PairwiseData,
    sigma_init: float = DEFAULT_SIGMA_INIT,
    max_iter: int = 1000,
    log_sigma_bounds: tuple[float, float] = DEFAULT_LOG_SIGMA_BOUNDS,
    mu_bounds: tuple[float, float] = DEFAULT_MU_BOUNDS,
    gradient_tol: float = 1.0,
    loss_tol: float = 1e-8,
    lambda_sigma: float = DEFAULT_LAMBDA_SIGMA,
) -> ThurstonianResult:
    n = data.n_tasks

    if n < 2:
        raise ValueError("Need at least 2 tasks to fit")

    mu_init = np.zeros(n - 1)
    log_sigma_init = np.full(n, np.log(sigma_init))
    params_init = np.concatenate([mu_init, log_sigma_init])

    bounds = [mu_bounds] * (n - 1) + [log_sigma_bounds] * n

    history = OptimizationHistory()
    row, col, count = data._row, data._col, data._count
    objective_and_grad = _make_objective_and_grad(row, col, count, n, lambda_sigma)

    def callback(params: np.ndarray) -> None:
        loss = _neg_log_likelihood(params, row, col, count, n)
        sigma_max = float(np.exp(params[n - 1 :]).max())
        history.loss.append(loss)
        history.sigma_max.append(sigma_max)

    result = minimize(
        objective_and_grad,
        params_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": max_iter,
            "maxfun": max_iter * 50,
            "gtol": gradient_tol,
            "ftol": loss_tol,
        },
    )

    mu = np.zeros(n)
    mu[1:] = result.x[: n - 1]
    sigma = np.exp(result.x[n - 1:])
    gradient_norm = float(np.linalg.norm(result.jac)) if result.jac is not None else None

    return ThurstonianResult(
        tasks=data.tasks,
        mu=mu,
        sigma=sigma,
        converged=result.success,
        neg_log_likelihood=result.fun,
        n_iterations=result.nit,
        n_function_evals=result.nfev,
        termination_message=result.message,
        gradient_norm=gradient_norm,
        history=history,
        n_comparisons=data.n_comparisons,
    )


def _config_hash(config: dict) -> str:
    """Compute short hash of config for filename uniqueness."""
    config_json = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.md5(config_json.encode())
    return hash_obj.hexdigest()[:8]


def save_thurstonian(
    result: ThurstonianResult,
    path: Path | str,
    fitting_method: str,
    config: dict | None = None,
) -> None:
    """Save Thurstonian results with config-based filename.

    The path should be the base path without hash. The config hash will be
    automatically appended to create unique filenames for different configs.
    """
    path = Path(path)

    # Add config hash to filename if config provided
    if config is not None:
        config_hash = _config_hash(config)
        stem = path.stem
        path = path.parent / f"{stem}_{config_hash}{path.suffix}"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "fitting_method": fitting_method,
        "converged": bool(result.converged),
        "neg_log_likelihood": float(result.neg_log_likelihood),
        "n_iterations": int(result.n_iterations),
        "n_function_evals": int(result.n_function_evals),
        "n_comparisons": int(result.n_comparisons),
        "termination_message": result.termination_message,
        "gradient_norm": float(result.gradient_norm) if result.gradient_norm is not None else None,
        "history": {
            "loss": [float(x) for x in result.history.loss],
            "sigma_max": [float(x) for x in result.history.sigma_max],
        },
    }

    if config is not None:
        data["config"] = config

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Save mu and sigma as CSV
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("task_id,mu,sigma\n")
        for task, mu, sigma in zip(result.tasks, result.mu, result.sigma):
            f.write(f"{task.id},{mu},{sigma}\n")


def load_thurstonian(path: Path | str, tasks: list["Task"]) -> ThurstonianResult:
    """Load Thurstonian results from YAML + CSV files."""
    path = Path(path)
    csv_path = path.with_suffix(".csv")

    with open(path) as f:
        data = yaml.safe_load(f)

    task_dict = {t.id: t for t in tasks}
    ordered_tasks = []
    mu_list = []
    sigma_list = []

    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            task_id, mu, sigma = line.strip().split(",")
            if task_id not in task_dict:
                raise ValueError(f"Task {task_id} not found in provided tasks")
            ordered_tasks.append(task_dict[task_id])
            mu_list.append(float(mu))
            sigma_list.append(float(sigma))

    history_data = data["history"]
    history = OptimizationHistory(
        loss=history_data["loss"],
        sigma_max=history_data["sigma_max"],
    )

    return ThurstonianResult(
        tasks=ordered_tasks,
        mu=np.array(mu_list),
        sigma=np.array(sigma_list),
        converged=data["converged"],
        neg_log_likelihood=data["neg_log_likelihood"],
        n_iterations=data["n_iterations"],
        n_function_evals=data["n_function_evals"],
        termination_message=data["termination_message"],
        gradient_norm=data["gradient_norm"],
        history=history,
        n_comparisons=data.get("n_comparisons", 0),
    )


def compute_pair_agreement(
    comparisons: list["BinaryPreferenceMeasurement"],
) -> float:
    """Compute mean agreement rate across pairs with multiple samples.

    For each pair (A, B), agreement = max(n_a_wins, n_b_wins) / total.
    Returns mean agreement across all pairs (1.0 = perfect consistency).
    """
    pair_outcomes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in comparisons:
        key = tuple(sorted([c.task_a.id, c.task_b.id]))
        winner = c.task_a.id if c.choice == "a" else c.task_b.id
        pair_outcomes[key].append(winner)

    agreements = []
    for outcomes in pair_outcomes.values():
        if len(outcomes) < 2:
            continue
        counts = Counter(outcomes)
        majority = max(counts.values())
        agreements.append(majority / len(outcomes))

    return float(np.mean(agreements)) if agreements else 1.0
