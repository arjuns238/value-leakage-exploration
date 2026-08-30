"""Protocol for utility fitting results.

Both ThurstonianResult and TrueSkillResult implement this interface,
allowing downstream code to work with either fitting method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from src.task_data import Task


@runtime_checkable
class UtilityResult(Protocol):
    """Common interface for utility fitting results.

    Both Thurstonian and TrueSkill results implement this interface.

    Properties:
        tasks: List of all tasks in the result
        n_observations: Number of observations used to fit (comparisons or rankings)

    Methods:
        utility(task) -> float: Point estimate of task utility (μ)
        uncertainty(task) -> float: Uncertainty in utility estimate (σ)
        ranking() -> list[Task]: Tasks sorted by utility, highest first
        preference_probability(task_a, task_b) -> float: P(a ≻ b)
        to_dict() -> dict: Serializable representation
    """

    @property
    def tasks(self) -> list["Task"]: ...

    @property
    def n_observations(self) -> int: ...

    def utility(self, task: "Task") -> float: ...

    def uncertainty(self, task: "Task") -> float: ...

    def ranking(self) -> list["Task"]: ...

    def preference_probability(self, task_a: "Task", task_b: "Task") -> float: ...

    def to_dict(self) -> dict: ...


def default_preference_probability(
    mu_a: float, mu_b: float, sigma_a: float, sigma_b: float
) -> float:
    """P(a ≻ b) assuming independent normal utilities.

    Under the Thurstonian model:
        U_a ~ N(mu_a, sigma_a²)
        U_b ~ N(mu_b, sigma_b²)
        U_a - U_b ~ N(mu_a - mu_b, sigma_a² + sigma_b²)
        P(a ≻ b) = P(U_a - U_b > 0) = Φ((mu_a - mu_b) / √(sigma_a² + sigma_b²))
    """
    scale = np.sqrt(sigma_a**2 + sigma_b**2)
    if scale < 1e-10:
        return 1.0 if mu_a > mu_b else (0.5 if mu_a == mu_b else 0.0)
    return float(norm.cdf((mu_a - mu_b) / scale))
