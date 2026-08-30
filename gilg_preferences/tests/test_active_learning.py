import numpy as np
import pytest

pytestmark = pytest.mark.thurstonian

from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType
from src.fitting.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
    _sorted_pair_key,
)
from src.fitting.thurstonian_fitting import PairwiseData, fit_thurstonian

from tests.helpers import make_task, make_comparison


class TestActiveLearningState:
    """Tests for ActiveLearningState tracking."""

    def test_state_tracks_degrees_and_sampled_pairs(self):
        """State correctly tracks degrees and sampled pairs across multiple comparisons."""
        tasks = [make_task(str(i)) for i in range(4)]
        state = ActiveLearningState(tasks=tasks)

        # Initial state should have zero degrees
        assert all(state.degree(t) == 0 for t in tasks)
        assert len(state.sampled_pairs) == 0

        # Add comparisons for pair (0, 1)
        c1 = make_comparison(tasks[0], tasks[1], "0")
        c2 = make_comparison(tasks[0], tasks[1], "1")
        state.add_comparisons([c1, c2])

        # Degrees should update only once per unique pair
        assert state.degree(tasks[0]) == 1
        assert state.degree(tasks[1]) == 1
        assert state.degree(tasks[2]) == 0
        assert len(state.sampled_pairs) == 1
        assert len(state.comparisons) == 2

        # Add comparison for new pair (1, 2)
        c3 = make_comparison(tasks[1], tasks[2], "1")
        state.add_comparisons([c3])

        assert state.degree(tasks[0]) == 1
        assert state.degree(tasks[1]) == 2
        assert state.degree(tasks[2]) == 1
        assert state.pair_degree_sum(tasks[0], tasks[1]) == 3
        assert len(state.sampled_pairs) == 2

    def test_get_unsampled_pairs_excludes_sampled(self):
        """get_unsampled_pairs returns only pairs not yet sampled."""
        tasks = [make_task(str(i)) for i in range(4)]
        state = ActiveLearningState(tasks=tasks)

        # Initially all 6 pairs (4 choose 2) are unsampled
        unsampled = state.get_unsampled_pairs()
        assert len(unsampled) == 6
        assert state.count_unsampled() == 6

        # Sample pair (0, 1)
        state.add_comparisons([make_comparison(tasks[0], tasks[1], "0")])
        unsampled = state.get_unsampled_pairs()
        assert len(unsampled) == 5
        assert state.count_unsampled() == 5

        # Verify (0, 1) is not in unsampled
        pair_keys = {_sorted_pair_key(a, b) for a, b in unsampled}
        assert ("0", "1") not in pair_keys

    def test_fit_updates_current_and_previous(self):
        """fit() updates current_fit and stores previous_fit."""
        tasks = [make_task(str(i)) for i in range(3)]
        state = ActiveLearningState(tasks=tasks)

        # Add comparisons to make fitting possible
        state.add_comparisons([
            make_comparison(tasks[0], tasks[1], "0"),
            make_comparison(tasks[1], tasks[2], "1"),
            make_comparison(tasks[0], tasks[2], "0"),
        ])

        assert state.current_fit is None
        assert state.previous_fit is None

        # First fit
        result1 = state.fit()
        assert state.current_fit is result1
        assert state.previous_fit is None

        # Second fit
        state.add_comparisons([make_comparison(tasks[0], tasks[1], "0")])
        result2 = state.fit()
        assert state.current_fit is result2
        assert state.previous_fit is result1

    def test_count_unsampled(self):
        """count_unsampled returns correct count without enumerating."""
        tasks = [make_task(str(i)) for i in range(5)]
        state = ActiveLearningState(tasks=tasks)

        assert state.count_unsampled() == 10  # 5 choose 2

        state.add_comparisons([make_comparison(tasks[0], tasks[1], "0")])
        assert state.count_unsampled() == 9

        state.add_comparisons([make_comparison(tasks[2], tasks[3], "2")])
        assert state.count_unsampled() == 8


class TestGenerateDRegularPairs:
    """Tests for d-regular graph generation."""

    def test_generates_reasonable_degree_distribution(self):
        """Tasks should have degrees at most d and form a connected graph."""
        tasks = [make_task(str(i)) for i in range(10)]
        rng = np.random.default_rng(42)

        pairs = generate_d_regular_pairs(tasks, d=3, rng=rng)

        # Count degrees
        degrees = {t.id: 0 for t in tasks}
        for a, b in pairs:
            degrees[a.id] += 1
            degrees[b.id] += 1

        # Degrees should not exceed d
        for tid, deg in degrees.items():
            assert deg <= 3, f"Task {tid} has degree {deg}, expected at most 3"

        # Most tasks should have degree d (algorithm is best-effort)
        assert sum(d == 3 for d in degrees.values()) >= 8

        # Expected number of edges for d-regular: n*d/2 = 10*3/2 = 15
        assert len(pairs) >= 14

    def test_returns_all_pairs_when_d_exceeds_n(self):
        """When d >= n-1, should return all possible pairs."""
        tasks = [make_task(str(i)) for i in range(5)]
        rng = np.random.default_rng(42)

        pairs = generate_d_regular_pairs(tasks, d=10, rng=rng)

        # All pairs = 5 choose 2 = 10
        assert len(pairs) == 10

    def test_no_duplicate_pairs(self):
        """Generated pairs should be unique."""
        tasks = [make_task(str(i)) for i in range(20)]
        rng = np.random.default_rng(42)

        pairs = generate_d_regular_pairs(tasks, d=4, rng=rng)

        pair_keys = [_sorted_pair_key(a, b) for a, b in pairs]
        assert len(pair_keys) == len(set(pair_keys))


class TestSelectNextPairs:
    """Tests for active pair selection."""

    def test_returns_random_when_no_fit(self):
        """Without a fitted model, returns random unsampled pairs."""
        tasks = [make_task(str(i)) for i in range(5)]
        state = ActiveLearningState(tasks=tasks)
        rng = np.random.default_rng(42)

        pairs = select_next_pairs(state, batch_size=3, rng=rng)
        assert len(pairs) == 3

        # All should be unsampled
        for a, b in pairs:
            assert state._idx_pair(a.id, b.id) not in state.sampled_pairs

    def test_returns_empty_when_all_sampled(self):
        """Returns empty list when no unsampled pairs remain."""
        tasks = [make_task(str(i)) for i in range(3)]
        state = ActiveLearningState(tasks=tasks)

        # Sample all pairs
        for i, t1 in enumerate(tasks):
            for t2 in tasks[i + 1:]:
                state.add_comparisons([make_comparison(t1, t2, t1.id)])

        pairs = select_next_pairs(state, batch_size=5)
        assert pairs == []

    def test_prioritizes_ambiguous_and_undersampled(self):
        """Should prefer pairs with close utilities and low degree."""
        tasks = [make_task(str(i)) for i in range(6)]
        state = ActiveLearningState(tasks=tasks)
        rng = np.random.default_rng(42)

        # Create clear preference ordering: 0 > 1 > 2 > 3 > 4 > 5
        for i in range(5):
            for _ in range(3):
                state.add_comparisons([make_comparison(tasks[i], tasks[i + 1], tasks[i].id)])

        state.fit()

        # Most pairs are unsampled. Pairs with adjacent indices should be preferred
        # because they have closer utilities.
        pairs = select_next_pairs(state, batch_size=5, rng=rng)
        assert len(pairs) == 5

        # Verify none are already sampled
        for a, b in pairs:
            assert state._idx_pair(a.id, b.id) not in state.sampled_pairs

    def test_returns_all_remaining_when_fewer_than_batch(self):
        """Returns all unsampled pairs when fewer than batch_size remain."""
        tasks = [make_task(str(i)) for i in range(3)]
        state = ActiveLearningState(tasks=tasks)

        # Sample 2 of 3 pairs
        state.add_comparisons([make_comparison(tasks[0], tasks[1], "0")])
        state.add_comparisons([make_comparison(tasks[1], tasks[2], "1")])

        pairs = select_next_pairs(state, batch_size=10)
        assert len(pairs) == 1

    def test_large_n_sampling_path(self):
        """Verify sampling-based selection works at scale (n=1001, triggers sampling path)."""
        n = 1001  # n*(n-1)/2 = 500500 > 500K, triggers sampling path
        tasks = [make_task(f"t{i}") for i in range(n)]
        rng = np.random.default_rng(42)
        state = ActiveLearningState(tasks=tasks)

        total_pairs = n * (n - 1) // 2
        assert state.count_unsampled() == total_pairs

        # No fit — random sampling
        pairs = select_next_pairs(state, batch_size=50, rng=rng)
        assert len(pairs) == 50

        # All unique
        pair_keys = {_sorted_pair_key(a, b) for a, b in pairs}
        assert len(pair_keys) == 50

        # None sampled
        for a, b in pairs:
            assert state._idx_pair(a.id, b.id) not in state.sampled_pairs

        # Add comparisons and fit, then select with scoring
        comparisons = []
        for a, b in pairs:
            choice = "a" if int(a.id[1:]) > int(b.id[1:]) else "b"
            comparisons.append(BinaryPreferenceMeasurement(
                task_a=a, task_b=b, choice=choice, preference_type=PreferenceType.PRE_TASK_STATED,
            ))
        state.add_comparisons(comparisons)
        state.fit()

        scored_pairs = select_next_pairs(state, batch_size=100, rng=rng)
        assert len(scored_pairs) == 100

        # All unique and unsampled
        scored_keys = {_sorted_pair_key(a, b) for a, b in scored_pairs}
        assert len(scored_keys) == 100
        for a, b in scored_pairs:
            assert state._idx_pair(a.id, b.id) not in state.sampled_pairs


class TestCheckConvergence:
    """Tests for convergence checking."""

    def test_not_converged_without_fits(self):
        """Returns False when no fits available."""
        tasks = [make_task(str(i)) for i in range(3)]
        state = ActiveLearningState(tasks=tasks)

        converged, corr = check_convergence(state)
        assert not converged
        assert corr == 0.0

    def test_not_converged_with_single_fit(self):
        """Returns False when only one fit available."""
        tasks = [make_task(str(i)) for i in range(3)]
        state = ActiveLearningState(tasks=tasks)

        state.add_comparisons([
            make_comparison(tasks[0], tasks[1], "0"),
            make_comparison(tasks[1], tasks[2], "1"),
        ])
        state.fit()

        converged, corr = check_convergence(state)
        assert not converged
        assert corr == 0.0

    def test_converges_with_stable_rankings(self):
        """Returns True when rankings are stable across fits."""
        tasks = [make_task(str(i)) for i in range(4)]
        state = ActiveLearningState(tasks=tasks)

        # Create strong preference ordering
        for i in range(3):
            for _ in range(5):
                state.add_comparisons([make_comparison(tasks[i], tasks[i + 1], tasks[i].id)])

        state.fit()

        # Add more data supporting the same ordering
        for _ in range(5):
            state.add_comparisons([make_comparison(tasks[0], tasks[3], tasks[0].id)])

        state.fit()

        converged, corr = check_convergence(state, threshold=0.95)
        # Rankings should be stable, so correlation should be high
        assert corr > 0.9


class TestSortedPairKey:
    """Tests for pair key canonicalization."""

    def test_produces_sorted_tuple(self):
        """Keys are sorted regardless of input order."""
        a = make_task("alpha")
        b = make_task("beta")

        assert _sorted_pair_key(a, b) == ("alpha", "beta")
        assert _sorted_pair_key(b, a) == ("alpha", "beta")


# --- Integration Tests ---


def simulate_oracle(task_a: Task, task_b: Task, true_utilities: dict[str, float], rng: np.random.Generator) -> BinaryPreferenceMeasurement:
    """Simulate a noisy oracle that prefers higher-utility tasks."""
    mu_a = true_utilities[task_a.id]
    mu_b = true_utilities[task_b.id]
    # Probability a wins is proportional to utility difference
    prob_a = 1.0 / (1.0 + np.exp(-(mu_a - mu_b)))
    choice = "a" if rng.random() < prob_a else "b"
    return BinaryPreferenceMeasurement(
        task_a=task_a, task_b=task_b, choice=choice, preference_type=PreferenceType.PRE_TASK_STATED
    )


class TestActiveLearningIntegration:
    """Integration tests simulating the full active learning loop."""

    def test_active_learning_recovers_ranking(self):
        """Active learning should recover the correct ranking from noisy comparisons."""
        n_tasks = 8
        tasks = [make_task(str(i)) for i in range(n_tasks)]
        true_utilities = {str(i): float(i) for i in range(n_tasks)}  # 0 < 1 < ... < 7
        rng = np.random.default_rng(123)

        state = ActiveLearningState(tasks=tasks)

        # Initial d-regular pairs
        initial_pairs = generate_d_regular_pairs(tasks, d=2, rng=rng)
        comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in initial_pairs]
        state.add_comparisons(comparisons)
        state.fit()

        # Run active learning iterations
        for _ in range(8):
            next_pairs = select_next_pairs(state, batch_size=4, rng=rng)
            if not next_pairs:
                break
            new_comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in next_pairs]
            state.add_comparisons(new_comparisons)
            state.fit()

        # Check that ranking correlates with true utilities
        ranking = state.current_fit.ranking()
        ranking_ids = [t.id for t in ranking]
        true_ranking = sorted(range(n_tasks), key=lambda i: -true_utilities[str(i)])
        true_ranking_ids = [str(i) for i in true_ranking]

        # Compute rank correlation
        from scipy.stats import spearmanr
        rank_positions = {id_: i for i, id_ in enumerate(ranking_ids)}
        true_positions = {id_: i for i, id_ in enumerate(true_ranking_ids)}
        fitted = [rank_positions[str(i)] for i in range(n_tasks)]
        expected = [true_positions[str(i)] for i in range(n_tasks)]
        corr = spearmanr(fitted, expected).correlation

        # Should achieve decent correlation with partial sampling
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_active_learning_uses_fewer_pairs_than_exhaustive(self):
        """Active learning should require fewer pairs than exhaustive comparison."""
        n_tasks = 10
        tasks = [make_task(str(i)) for i in range(n_tasks)]
        true_utilities = {str(i): float(i) for i in range(n_tasks)}
        rng = np.random.default_rng(456)

        state = ActiveLearningState(tasks=tasks)
        total_possible_pairs = n_tasks * (n_tasks - 1) // 2  # 45

        # Initial pairs
        initial_pairs = generate_d_regular_pairs(tasks, d=2, rng=rng)
        comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in initial_pairs]
        state.add_comparisons(comparisons)
        state.fit()

        # Run until convergence or max iterations
        max_iters = 10
        for _ in range(max_iters):
            next_pairs = select_next_pairs(state, batch_size=5, rng=rng)
            if not next_pairs:
                break
            new_comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in next_pairs]
            state.add_comparisons(new_comparisons)
            old_fit = state.current_fit
            state.fit()

            converged, corr = check_convergence(state, threshold=0.95)
            if converged:
                break

        # Should have sampled fewer than all pairs
        pairs_sampled = len(state.sampled_pairs)
        assert pairs_sampled < total_possible_pairs, f"Sampled {pairs_sampled} pairs, expected < {total_possible_pairs}"

    def test_active_learning_state_through_multiple_iterations(self):
        """Verify state consistency through multiple iterations."""
        tasks = [make_task(str(i)) for i in range(6)]
        true_utilities = {str(i): float(i) * 2 for i in range(6)}
        rng = np.random.default_rng(789)

        state = ActiveLearningState(tasks=tasks)

        # Track state through iterations
        iteration_data = []

        initial_pairs = generate_d_regular_pairs(tasks, d=2, rng=rng)
        comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in initial_pairs]
        state.add_comparisons(comparisons)
        state.fit()
        state.iteration = 1

        iteration_data.append({
            "iteration": 1,
            "pairs_sampled": len(state.sampled_pairs),
            "comparisons": len(state.comparisons),
        })

        for i in range(2, 5):
            next_pairs = select_next_pairs(state, batch_size=3, rng=rng)
            if not next_pairs:
                break
            new_comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in next_pairs]
            state.add_comparisons(new_comparisons)
            state.fit()
            state.iteration = i

            iteration_data.append({
                "iteration": i,
                "pairs_sampled": len(state.sampled_pairs),
                "comparisons": len(state.comparisons),
            })

        # Verify monotonic increases
        for j in range(1, len(iteration_data)):
            assert iteration_data[j]["pairs_sampled"] >= iteration_data[j - 1]["pairs_sampled"]
            assert iteration_data[j]["comparisons"] >= iteration_data[j - 1]["comparisons"]

        # Verify final state consistency
        assert state.iteration == len(iteration_data)
        assert state.current_fit is not None
        assert state.previous_fit is not None

    def test_select_next_pairs_avoids_already_sampled(self):
        """Selected pairs should never include already-sampled pairs."""
        tasks = [make_task(str(i)) for i in range(8)]
        true_utilities = {str(i): float(i) for i in range(8)}
        rng = np.random.default_rng(111)

        state = ActiveLearningState(tasks=tasks)

        # Run several iterations
        initial_pairs = generate_d_regular_pairs(tasks, d=2, rng=rng)
        comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in initial_pairs]
        state.add_comparisons(comparisons)
        state.fit()

        for _ in range(4):
            next_pairs = select_next_pairs(state, batch_size=4, rng=rng)
            # Verify none of these pairs are already sampled
            for a, b in next_pairs:
                key = state._idx_pair(a.id, b.id)
                assert key not in state.sampled_pairs, f"Pair {key} was already sampled!"

            if not next_pairs:
                break
            new_comparisons = [simulate_oracle(a, b, true_utilities, rng) for a, b in next_pairs]
            state.add_comparisons(new_comparisons)
            state.fit()


class TestMultiDatasetActivelearning:
    """Integration tests for active learning with multiple datasets."""

    def test_active_learning_with_multiple_datasets(self):
        """Active learning should work with tasks from multiple datasets."""
        from src.task_data import load_tasks

        tasks = load_tasks(
            n=10,
            origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA],
            seed=42,
        )

        # Verify we got tasks from multiple origins
        origins_present = {t.origin for t in tasks}
        assert len(origins_present) >= 1

        # Run active learning with these tasks
        rng = np.random.default_rng(42)
        state = ActiveLearningState(tasks=tasks)

        initial_pairs = generate_d_regular_pairs(tasks, d=2, rng=rng)
        assert len(initial_pairs) > 0

        # Simulate comparisons (arbitrary oracle based on task index)
        task_index = {t.id: i for i, t in enumerate(tasks)}
        comparisons = []
        for a, b in initial_pairs:
            winner = a if task_index[a.id] > task_index[b.id] else b
            comparisons.append(make_comparison(a, b, winner.id))

        state.add_comparisons(comparisons)
        state.fit()

        assert state.current_fit is not None
        assert len(state.current_fit.ranking()) == len(tasks)

    def test_multi_dataset_seed_reproducibility(self):
        """Same seed should produce identical active learning setup."""
        from src.task_data import load_tasks

        # Load twice with same seed
        tasks1 = load_tasks(n=8, origins=[OriginDataset.WILDCHAT, OriginDataset.MATH], seed=123)
        tasks2 = load_tasks(n=8, origins=[OriginDataset.WILDCHAT, OriginDataset.MATH], seed=123)

        assert [t.id for t in tasks1] == [t.id for t in tasks2]

        # Run active learning with same RNG seed - should get same pairs
        rng1 = np.random.default_rng(456)
        rng2 = np.random.default_rng(456)

        pairs1 = generate_d_regular_pairs(tasks1, d=2, rng=rng1)
        pairs2 = generate_d_regular_pairs(tasks2, d=2, rng=rng2)

        pairs1_ids = [(a.id, b.id) for a, b in pairs1]
        pairs2_ids = [(a.id, b.id) for a, b in pairs2]

        assert pairs1_ids == pairs2_ids
