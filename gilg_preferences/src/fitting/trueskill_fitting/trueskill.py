from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import trueskill

from src.fitting.protocol import default_preference_probability

if TYPE_CHECKING:
    from src.task_data import Task
    from src.types import RankingMeasurement


@dataclass
class TrueSkillResult:
    """TrueSkill fitting result. Implements UtilityResult protocol."""

    tasks: list["Task"]
    n_observations: int
    _mu: dict[str, float] = field(repr=False)
    _sigma: dict[str, float] = field(repr=False)

    def utility(self, task: "Task") -> float:
        return self._mu[task.id]

    def uncertainty(self, task: "Task") -> float:
        return self._sigma[task.id]

    def ranking(self) -> list["Task"]:
        return sorted(self.tasks, key=lambda t: self._mu[t.id], reverse=True)

    def preference_probability(self, task_a: "Task", task_b: "Task") -> float:
        return default_preference_probability(
            self._mu[task_a.id], self._mu[task_b.id],
            self._sigma[task_a.id], self._sigma[task_b.id],
        )

    def to_dict(self) -> dict:
        return {
            "task_ids": [t.id for t in self.tasks],
            "mu": self._mu,
            "sigma": self._sigma,
            "n_observations": self.n_observations,
        }


def fit_trueskill_from_rankings(rankings: list["RankingMeasurement"]) -> TrueSkillResult:
    """Fit TrueSkill ratings treating each ranking as a multi-team match."""
    if not rankings:
        raise ValueError("No rankings provided")

    # Collect all unique tasks
    task_by_id: dict[str, "Task"] = {}
    for r in rankings:
        for t in r.tasks:
            task_by_id[t.id] = t

    ratings: dict[str, trueskill.Rating] = {
        tid: trueskill.Rating() for tid in task_by_id
    }

    for r in rankings:
        rating_groups = [
            {r.tasks[idx].id: ratings[r.tasks[idx].id]}
            for idx in r.ranking
        ]
        new_ratings = trueskill.rate(rating_groups)

        for group in new_ratings:
            for tid, rating in group.items():
                ratings[tid] = rating

    return TrueSkillResult(
        tasks=list(task_by_id.values()),
        n_observations=len(rankings),
        _mu={tid: r.mu for tid, r in ratings.items()},
        _sigma={tid: r.sigma for tid, r in ratings.items()},
    )
