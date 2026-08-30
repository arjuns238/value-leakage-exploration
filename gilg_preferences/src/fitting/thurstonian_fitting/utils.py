"""Utility functions for Thurstonian model analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from .thurstonian import _preference_prob


def simulate_pairwise_comparisons(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_comparisons_per_pair: int,
    rng: np.random.Generator,
    noise_rate: float = 0.0,
) -> np.ndarray:
    """Simulate pairwise comparison outcomes from known Thurstonian parameters.

    Args:
        mu: True utility means for each item
        sigma: True utility standard deviations for each item
        n_comparisons_per_pair: Number of comparisons per (i, j) pair
        rng: Random number generator
        noise_rate: Fraction of comparisons to flip randomly (0-1)

    Returns:
        wins: (n, n) matrix where wins[i, j] = times item i beat item j
    """
    n = len(mu)
    wins = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            p_i_beats_j = _preference_prob(mu[i], mu[j], sigma[i], sigma[j])
            for _ in range(n_comparisons_per_pair):
                if rng.random() < noise_rate:
                    i_wins = rng.random() < 0.5
                else:
                    i_wins = rng.random() < p_i_beats_j

                if i_wins:
                    wins[i, j] += 1
                else:
                    wins[j, i] += 1

    return wins


@dataclass
class SolutionSimilarity:
    """Pairwise similarity metrics between multiple fitted solutions."""

    mu_correlations: np.ndarray  # (n, n) Pearson correlations of μ vectors
    rank_correlations: np.ndarray  # (n, n) Spearman rank correlations
    labels: list[str]

    def mean_rank_correlation(self) -> float:
        """Mean off-diagonal rank correlation."""
        n = len(self.labels)
        if n < 2:
            return 1.0
        off_diag = self.rank_correlations[np.triu_indices(n, k=1)]
        return float(off_diag.mean())

    def min_rank_correlation(self) -> float:
        """Minimum pairwise rank correlation."""
        n = len(self.labels)
        if n < 2:
            return 1.0
        off_diag = self.rank_correlations[np.triu_indices(n, k=1)]
        return float(off_diag.min())


def compute_solution_similarity(
    mu_vectors: Sequence[np.ndarray],
    labels: list[str] | None = None,
) -> SolutionSimilarity:
    """Compute pairwise similarity between multiple μ solutions.

    Args:
        mu_vectors: List of μ arrays to compare
        labels: Optional labels for each solution

    Returns:
        SolutionSimilarity with correlation matrices
    """
    n = len(mu_vectors)
    if labels is None:
        labels = [f"S{i}" for i in range(n)]

    mu_correlations = np.zeros((n, n))
    rank_correlations = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mu_correlations[i, j] = np.corrcoef(mu_vectors[i], mu_vectors[j])[0, 1]
            rank_correlations[i, j] = spearmanr(mu_vectors[i], mu_vectors[j]).correlation

    return SolutionSimilarity(
        mu_correlations=mu_correlations,
        rank_correlations=rank_correlations,
        labels=labels,
    )


@dataclass
class AggregatedMetrics:
    """Summary statistics across multiple datasets or runs."""

    n_datasets: int
    mean: float
    std: float
    min: float
    max: float
    values: list[float]

    @classmethod
    def from_values(cls, values: Sequence[float], name: str = "") -> "AggregatedMetrics":
        arr = np.array(values)
        return cls(
            n_datasets=len(arr),
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
            values=list(values),
        )


def aggregate_rank_correlations(
    similarities: list[SolutionSimilarity],
) -> dict[str, AggregatedMetrics]:
    """Aggregate rank correlation statistics across multiple datasets.

    Returns dict mapping metric name to AggregatedMetrics.
    """
    means = [s.mean_rank_correlation() for s in similarities]
    mins = [s.min_rank_correlation() for s in similarities]

    return {
        "mean_rank_corr": AggregatedMetrics.from_values(means),
        "min_rank_corr": AggregatedMetrics.from_values(mins),
    }


def check_solution_stability(
    mu_vectors: Sequence[np.ndarray],
    threshold: float = 0.99,
) -> tuple[bool, float]:
    """Check if multiple solutions are essentially identical.

    Args:
        mu_vectors: List of μ arrays from different runs/configs
        threshold: Minimum rank correlation to consider stable

    Returns:
        (is_stable, min_correlation)
    """
    if len(mu_vectors) < 2:
        return True, 1.0

    similarity = compute_solution_similarity(mu_vectors)
    min_corr = similarity.min_rank_correlation()
    return min_corr >= threshold, min_corr
