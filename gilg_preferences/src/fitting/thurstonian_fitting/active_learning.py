"""Active learning for efficient Thurstonian utility estimation.

Implements the iterative pair selection algorithm from:
"Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"
(Mazeika et al., 2025), Appendix B.

The algorithm prioritizes pairs where:
1. Current utility estimates are close (ambiguous pairs)
2. Items have been compared fewer times (under-sampled)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from src.task_data import Task
from src.types import BinaryPreferenceMeasurement
from .thurstonian import PairwiseData, ThurstonianResult, fit_thurstonian


def _sorted_pair_key(a: Task, b: Task) -> tuple[str, str]:
    """Canonical string key for a pair of tasks. Only used in tests."""
    return tuple(sorted([a.id, b.id]))


@dataclass
class ActiveLearningState:
    """Tracks state across active learning iterations."""

    tasks: list[Task]
    comparisons: list[BinaryPreferenceMeasurement] = field(default_factory=list)
    sampled_pairs: set[tuple[int, int]] = field(default_factory=set)
    current_fit: ThurstonianResult | None = None
    previous_fit: ThurstonianResult | None = None
    iteration: int = 0

    _id_to_task: dict[str, Task] = field(init=False, repr=False)
    _id_to_idx: dict[str, int] = field(init=False, repr=False)
    _degrees: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_task = {t.id: t for t in self.tasks}
        self._id_to_idx = {t.id: i for i, t in enumerate(self.tasks)}
        self._degrees = np.zeros(len(self.tasks), dtype=np.int32)

    def _idx_pair(self, id_a: str, id_b: str) -> tuple[int, int]:
        i, j = self._id_to_idx[id_a], self._id_to_idx[id_b]
        return (i, j) if i < j else (j, i)

    def degree(self, task: Task) -> int:
        return int(self._degrees[self._id_to_idx[task.id]])

    def pair_degree_sum(self, a: Task, b: Task) -> int:
        return int(self._degrees[self._id_to_idx[a.id]] + self._degrees[self._id_to_idx[b.id]])

    def count_unsampled(self) -> int:
        n = len(self.tasks)
        return n * (n - 1) // 2 - len(self.sampled_pairs)

    def add_comparisons(self, comparisons: list[BinaryPreferenceMeasurement]) -> None:
        for c in comparisons:
            key = self._idx_pair(c.task_a.id, c.task_b.id)
            if key not in self.sampled_pairs:
                self.sampled_pairs.add(key)
                self._degrees[key[0]] += 1
                self._degrees[key[1]] += 1
        self.comparisons.extend(comparisons)

    def get_unsampled_pairs(self) -> list[tuple[Task, Task]]:
        """Get all pairs that haven't been sampled yet. O(n^2) — only for small n."""
        unsampled = []
        for a, b in combinations(self.tasks, 2):
            if self._idx_pair(a.id, b.id) not in self.sampled_pairs:
                unsampled.append((a, b))
        return unsampled

    def fit(self, **fit_kwargs) -> ThurstonianResult:
        data = PairwiseData.from_comparisons(self.comparisons, self.tasks)
        self.previous_fit = self.current_fit
        self.current_fit = fit_thurstonian(data, **fit_kwargs)
        return self.current_fit


def generate_d_regular_pairs(
    tasks: list[Task],
    d: int,
    rng: np.random.Generator,
) -> list[tuple[Task, Task]]:
    """Generate a random d-regular graph over tasks.

    Each task will be connected to exactly d other tasks (if possible).
    For n tasks with d edges each, we need n*d/2 unique edges.
    """
    n = len(tasks)
    if d >= n:
        return list(combinations(tasks, 2))

    degrees = {t.id: 0 for t in tasks}
    edges: set[tuple[str, str]] = set()
    task_ids = [t.id for t in tasks]

    max_attempts = n * d * 10
    attempts = 0

    while attempts < max_attempts:
        need_edges = [tid for tid in task_ids if degrees[tid] < d]
        if not need_edges:
            break

        rng.shuffle(need_edges)
        added = False
        for i, tid_a in enumerate(need_edges):
            for tid_b in need_edges[i + 1:]:
                key = tuple(sorted([tid_a, tid_b]))
                if key not in edges and degrees[tid_a] < d and degrees[tid_b] < d:
                    edges.add(key)
                    degrees[tid_a] += 1
                    degrees[tid_b] += 1
                    added = True
                    break
            if added:
                break
        attempts += 1

    id_to_task = {t.id: t for t in tasks}
    return [(id_to_task[a], id_to_task[b]) for a, b in edges]


def _sample_random_pairs(
    n: int,
    count: int,
    sampled: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample random unsampled index pairs via rejection sampling."""
    result: list[tuple[int, int]] = []
    seen = set(sampled)
    total_generated = 0
    max_generated = count * 80
    while len(result) < count and total_generated < max_generated:
        batch_size = min((count - len(result)) * 4, max_generated - total_generated)
        batch = rng.integers(0, n, size=(batch_size, 2))
        total_generated += batch_size
        for row in batch:
            i, j = int(row[0]), int(row[1])
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                result.append(key)
                if len(result) >= count:
                    break
    return result


def select_next_pairs(
    state: ActiveLearningState,
    batch_size: int,
    p_threshold: float = 0.3,
    q_threshold: float = 0.3,
    relaxation_factor: float = 1.5,
    rng: np.random.Generator | None = None,
) -> list[tuple[Task, Task]]:
    """Select the next batch of pairs to query.

    Prioritizes pairs where:
    1. |mu(a) - mu(b)| is small (ambiguous utility difference)
    2. degree(a) + degree(b) is small (under-sampled)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(state.tasks)
    n_unsampled = state.count_unsampled()

    if n_unsampled == 0:
        return []

    total_pairs = n * (n - 1) // 2
    if total_pairs <= 500_000:
        return _select_next_pairs_exact(state, batch_size, p_threshold, q_threshold, relaxation_factor, rng)

    if n_unsampled <= batch_size:
        return state.get_unsampled_pairs()

    # No fit yet — random sampling
    if state.current_fit is None:
        idx_pairs = _sample_random_pairs(n, batch_size, state.sampled_pairs, rng)
        return [(state.tasks[i], state.tasks[j]) for i, j in idx_pairs]

    # Scored selection via sampling
    return _select_next_pairs_sampled(state, batch_size, p_threshold, q_threshold, relaxation_factor, rng)


def _select_next_pairs_exact(
    state: ActiveLearningState,
    batch_size: int,
    p_threshold: float,
    q_threshold: float,
    relaxation_factor: float,
    rng: np.random.Generator,
) -> list[tuple[Task, Task]]:
    """Exact algorithm — enumerates all unsampled pairs. Fine for n^2 <= 500K."""
    unsampled = state.get_unsampled_pairs()

    if len(unsampled) <= batch_size:
        return unsampled

    if state.current_fit is None:
        rng.shuffle(unsampled)
        return unsampled[:batch_size]

    mu = state.current_fit.mu
    id_to_idx = state.current_fit._id_to_idx

    pair_scores = []
    for a, b in unsampled:
        mu_diff = abs(mu[id_to_idx[a.id]] - mu[id_to_idx[b.id]])
        degree_sum = state.pair_degree_sum(a, b)
        pair_scores.append((a, b, mu_diff, degree_sum))

    mu_diffs = np.array([s[2] for s in pair_scores])
    degree_sums = np.array([s[3] for s in pair_scores])

    current_p = p_threshold
    current_q = q_threshold

    while True:
        mu_cutoff = np.percentile(mu_diffs, current_p * 100)
        degree_cutoff = np.percentile(degree_sums, current_q * 100)

        candidates = [
            (a, b)
            for a, b, mu_d, deg_s in pair_scores
            if mu_d <= mu_cutoff and deg_s <= degree_cutoff
        ]

        if len(candidates) >= batch_size:
            rng.shuffle(candidates)
            return candidates[:batch_size]

        current_p = min(1.0, current_p * relaxation_factor)
        current_q = min(1.0, current_q * relaxation_factor)

        if current_p >= 1.0 and current_q >= 1.0:
            remaining = [
                (a, b) for a, b, _, _ in pair_scores if (a, b) not in candidates
            ]
            rng.shuffle(remaining)
            return candidates + remaining[: batch_size - len(candidates)]


def _select_next_pairs_sampled(
    state: ActiveLearningState,
    batch_size: int,
    p_threshold: float,
    q_threshold: float,
    relaxation_factor: float,
    rng: np.random.Generator,
) -> list[tuple[Task, Task]]:
    """Sampling-based pair selection for large n. Memory: O(chunk_size) not O(n^2).

    Draws random candidate chunks, computes scores vectorized, filters by
    percentile cutoffs on each chunk. Same logic as the exact path but
    never materializes all n^2 pairs.
    """
    n = len(state.tasks)
    mu = state.current_fit.mu
    fit_id_to_idx = state.current_fit._id_to_idx
    degrees = state._degrees
    # Remap mu from fit ordering to state task ordering
    remap = np.array([fit_id_to_idx[t.id] for t in state.tasks])
    mu_by_state_idx = mu[remap]

    current_p = p_threshold
    current_q = q_threshold
    collected: list[tuple[int, int]] = []
    collected_set: set[tuple[int, int]] = set()
    chunk_size = max(batch_size * 15, 75_000)

    while len(collected) < batch_size:
        # Sample a chunk of random pairs
        cand_is = rng.integers(0, n, size=chunk_size)
        cand_js = rng.integers(0, n, size=chunk_size)

        valid = cand_is != cand_js
        cand_is, cand_js = cand_is[valid], cand_js[valid]

        lo = np.minimum(cand_is, cand_js)
        hi = np.maximum(cand_is, cand_js)
        cand_is, cand_js = lo, hi

        # Score and filter by percentile cutoffs within this chunk
        mu_diffs = np.abs(mu_by_state_idx[cand_is] - mu_by_state_idx[cand_js])
        deg_sums = degrees[cand_is] + degrees[cand_js]

        mu_cutoff = np.percentile(mu_diffs, current_p * 100)
        degree_cutoff = np.percentile(deg_sums, current_q * 100)

        mask = (mu_diffs <= mu_cutoff) & (deg_sums <= degree_cutoff)
        filt_is = cand_is[mask]
        filt_js = cand_js[mask]

        for k in range(len(filt_is)):
            key = (int(filt_is[k]), int(filt_js[k]))
            if key not in state.sampled_pairs and key not in collected_set:
                collected_set.add(key)
                collected.append(key)
                if len(collected) >= batch_size:
                    break

        if len(collected) >= batch_size:
            break

        # Relax thresholds
        current_p = min(1.0, current_p * relaxation_factor)
        current_q = min(1.0, current_q * relaxation_factor)

        if current_p >= 1.0 and current_q >= 1.0:
            extra = _sample_random_pairs(
                n, batch_size - len(collected), state.sampled_pairs | collected_set, rng
            )
            collected.extend(extra)
            break

    return [(state.tasks[i], state.tasks[j]) for i, j in collected[:batch_size]]


def check_convergence(
    state: ActiveLearningState,
    threshold: float = 0.99,
) -> tuple[bool, float]:
    """Check if utility estimates have converged.

    Compares rank correlation between current and previous iteration.
    """
    if state.current_fit is None or state.previous_fit is None:
        return False, 0.0

    if state.current_fit.tasks != state.previous_fit.tasks:
        return False, 0.0

    corr = spearmanr(state.current_fit.mu, state.previous_fit.mu).correlation
    return corr >= threshold, float(corr)
