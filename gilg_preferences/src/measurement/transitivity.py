from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class TransitivityResult:
    cycle_probability: float  # P(cycle) across all triads
    n_triads: int
    n_cycles: int
    sampled: bool = False

    @property
    def log_cycle_prob(self) -> float:
        if self.cycle_probability <= 0:
            return float("-inf")
        return float(np.log10(self.cycle_probability))

    @property
    def hard_cycle_rate(self) -> float:
        return self.n_cycles / self.n_triads if self.n_triads > 0 else 0.0


def measure_transitivity(
    wins: np.ndarray,
    max_triads: int = 100_000,
    seed: int = 42,
) -> TransitivityResult:
    """Measure transitivity from a wins matrix.

    wins[i,j] = number of times i beat j.

    For large matrices, uses sampling to estimate cycle probability.
    """
    n = wins.shape[0]

    if n < 3:
        return TransitivityResult(cycle_probability=0.0, n_triads=0, n_cycles=0)

    total_triads = n * (n - 1) * (n - 2) // 6

    # Preference probabilities: P(i > j)
    total = wins + wins.T
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(total > 0, wins / total, 0.5)

    # For hard cycles
    prefers = wins > wins.T

    if total_triads <= max_triads:
        # Exact calculation
        return _measure_exact(probs, prefers, n)
    else:
        # Sampling-based estimation
        return _measure_sampled(probs, prefers, n, max_triads, seed)


def _measure_exact(probs: np.ndarray, prefers: np.ndarray, n: int) -> TransitivityResult:
    """Exact transitivity calculation over all triads."""
    n_triads = 0
    cycle_prob_sum = 0.0
    n_cycles = 0

    for i, j, k in itertools.combinations(range(n), 3):
        n_triads += 1

        # Probabilistic cycle: P(i>j>k>i) + P(i>k>j>i)
        p_cycle_1 = probs[i, j] * probs[j, k] * probs[k, i]
        p_cycle_2 = probs[i, k] * probs[k, j] * probs[j, i]
        cycle_prob_sum += p_cycle_1 + p_cycle_2

        # Hard cycle check
        if (prefers[i, j] and prefers[j, k] and prefers[k, i]) or (
            prefers[i, k] and prefers[k, j] and prefers[j, i]
        ):
            n_cycles += 1

    avg_cycle_prob = cycle_prob_sum / n_triads if n_triads > 0 else 0.0

    return TransitivityResult(
        cycle_probability=avg_cycle_prob,
        n_triads=n_triads,
        n_cycles=n_cycles,
        sampled=False,
    )


def _measure_sampled(
    probs: np.ndarray,
    prefers: np.ndarray,
    n: int,
    n_samples: int,
    seed: int,
) -> TransitivityResult:
    """Sample-based transitivity estimation."""
    rng = np.random.default_rng(seed)

    cycle_prob_sum = 0.0
    n_cycles = 0

    for _ in range(n_samples):
        # Sample 3 distinct indices
        idx = rng.choice(n, size=3, replace=False)
        i, j, k = idx[0], idx[1], idx[2]

        # Probabilistic cycle
        p_cycle_1 = probs[i, j] * probs[j, k] * probs[k, i]
        p_cycle_2 = probs[i, k] * probs[k, j] * probs[j, i]
        cycle_prob_sum += p_cycle_1 + p_cycle_2

        # Hard cycle check
        if (prefers[i, j] and prefers[j, k] and prefers[k, i]) or (
            prefers[i, k] and prefers[k, j] and prefers[j, i]
        ):
            n_cycles += 1

    avg_cycle_prob = cycle_prob_sum / n_samples

    return TransitivityResult(
        cycle_probability=avg_cycle_prob,
        n_triads=n_samples,
        n_cycles=n_cycles,
        sampled=True,
    )
