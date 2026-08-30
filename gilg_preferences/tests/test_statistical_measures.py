import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.analysis

import yaml

from src.measurement.correlation import (
    safe_correlation,
    save_correlations_yaml,
    utility_vector_correlation,
    compute_pairwise_correlations,
    build_score_map,
    compute_per_task_std,
    compute_mean_std_across_tasks,
    scores_to_vector,
    build_win_rate_vector,
    win_rate_correlation,
)
from src.measurement.transitivity import measure_transitivity, TransitivityResult
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, TaskScore, PreferenceType
from src.fitting.thurstonian_fitting import ThurstonianResult, OptimizationHistory

from tests.helpers import make_task, make_score, make_measurement


class TestSafeCorrelation:
    """Tests for safe_correlation edge case handling."""

    def test_perfect_positive_correlation(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 4, 6, 8, 10])
        assert safe_correlation(a, b) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([10, 8, 6, 4, 2])
        assert safe_correlation(a, b) == pytest.approx(-1.0)

    def test_weak_correlation(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([3, 1, 4, 1, 5])
        r = safe_correlation(a, b)
        # This specific data has weak positive correlation ~0.35
        assert r == pytest.approx(0.354, abs=0.01)
        # Verify it's neither strongly positive nor negative
        assert -0.5 < r < 0.5

    def test_insufficient_data_single_element(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert safe_correlation(a, b) == 0.0

    def test_insufficient_data_empty(self):
        a = np.array([])
        b = np.array([])
        assert safe_correlation(a, b) == 0.0

    def test_zero_variance_in_first_array(self):
        a = np.array([5, 5, 5, 5, 5])
        b = np.array([1, 2, 3, 4, 5])
        assert safe_correlation(a, b) == 0.0

    def test_zero_variance_in_second_array(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([3, 3, 3, 3, 3])
        assert safe_correlation(a, b) == 0.0

    def test_spearman_correlation(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 4, 9, 16, 25])  # Monotonic but non-linear
        r = safe_correlation(a, b, method="spearman")
        assert r == pytest.approx(1.0)  # Perfect rank correlation

    def test_pearson_vs_spearman_difference(self):
        # Monotonic but non-linear relationship
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 4, 9, 16, 25])
        pearson = safe_correlation(a, b, method="pearson")
        spearman = safe_correlation(a, b, method="spearman")
        # Spearman should be 1.0, Pearson should be less
        assert spearman == pytest.approx(1.0)
        assert pearson < spearman


class TestSaveCorrelationsYaml:
    """Tests for saving correlations to YAML."""

    def test_saves_summary_and_pairwise(self):
        correlations = [
            {"template_a": "t1", "template_b": "t2", "r": 0.8},
            {"template_a": "t1", "template_b": "t3", "r": 0.6},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "correlations.yaml"
            save_correlations_yaml(correlations, summary_keys=["r"], path=path)

            with open(path) as f:
                data = yaml.safe_load(f)

            assert "summary" in data
            assert "pairwise" in data
            assert data["summary"]["n_pairs"] == 2
            assert data["summary"]["mean_r"] == pytest.approx(0.7)
            assert data["pairwise"] == correlations

    def test_creates_parent_directories(self):
        correlations = [{"x": 1}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "result.yaml"
            save_correlations_yaml(correlations, summary_keys=[], path=path)
            assert path.exists()

    def test_empty_correlations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.yaml"
            save_correlations_yaml([], summary_keys=["r"], path=path)

            with open(path) as f:
                data = yaml.safe_load(f)

            assert data["summary"]["n_pairs"] == 0
            assert data["summary"]["mean_r"] == 0.0


class TestMeasureTransitivity:
    """Tests for transitivity measurement."""

    def test_perfect_transitivity_no_cycles(self):
        # Clear dominance: 0 > 1 > 2 (no cycles)
        wins = np.array([
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0],
        ])
        result = measure_transitivity(wins)

        assert isinstance(result, TransitivityResult)
        assert result.n_triads == 1  # C(3,3) = 1
        assert result.n_cycles == 0
        assert result.cycle_probability == pytest.approx(0.0)

    def test_complete_cycle(self):
        # Rock-paper-scissors: 0 > 1 > 2 > 0
        wins = np.array([
            [0, 10, 0],
            [0, 0, 10],
            [10, 0, 0],
        ])
        result = measure_transitivity(wins)

        assert result.n_triads == 1
        assert result.n_cycles == 1
        # Deterministic cycle: P(0>1>2>0) = 1.0, P(0>2>1>0) = 0.0
        assert result.cycle_probability == pytest.approx(1.0)

    def test_probabilistic_preferences(self):
        # Probabilistic: 0 beats 1 60%, 1 beats 2 60%, 2 beats 0 60%
        wins = np.array([
            [0, 6, 4],
            [4, 0, 6],
            [6, 4, 0],
        ])
        result = measure_transitivity(wins)

        assert result.n_triads == 1
        # P(0>1>2>0) = 0.6^3 = 0.216, P(0>2>1>0) = 0.4^3 = 0.064
        # Total = 0.28
        assert result.cycle_probability == pytest.approx(0.28)

    def test_insufficient_items(self):
        # Need at least 3 items for triads
        wins = np.array([[0, 5], [5, 0]])
        result = measure_transitivity(wins)

        assert result.n_triads == 0
        assert result.n_cycles == 0
        assert result.cycle_probability == 0.0

    def test_larger_matrix(self):
        # 5 items = C(5,3) = 10 triads
        n = 5
        wins = np.zeros((n, n))
        # Create transitive order: i beats j for all i < j
        for i in range(n):
            for j in range(i + 1, n):
                wins[i, j] = 10
        result = measure_transitivity(wins)

        assert result.n_triads == 10
        assert result.n_cycles == 0

    def test_log_cycle_prob_zero(self):
        wins = np.array([
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0],
        ])
        result = measure_transitivity(wins)
        assert result.log_cycle_prob == float("-inf")

    def test_log_cycle_prob_positive(self):
        wins = np.array([
            [0, 10, 0],
            [0, 0, 10],
            [10, 0, 0],
        ])
        result = measure_transitivity(wins)
        # cycle_probability = 1.0, so log10(1.0) = 0.0
        assert result.log_cycle_prob == pytest.approx(0.0)


class TestRatingCorrelation:
    """End-to-end tests for rating-based correlation computation."""

    @pytest.fixture
    def tasks(self):
        return [make_task(f"task_{i}") for i in range(5)]

    def testbuild_score_map_averages_duplicates(self, tasks):
        scores = [
            make_score(tasks[0], 7.0),
            make_score(tasks[0], 9.0),  # Duplicate task
            make_score(tasks[1], 5.0),
        ]
        score_map = build_score_map(scores)

        assert score_map["task_0"] == pytest.approx(8.0)  # Average of 7 and 9
        assert score_map["task_1"] == pytest.approx(5.0)

    def test_compute_per_task_std(self, tasks):
        scores = [
            make_score(tasks[0], 6.0),
            make_score(tasks[0], 8.0),
            make_score(tasks[0], 10.0),
        ]
        std_map = compute_per_task_std(scores)

        # std of [6, 8, 10] = ~1.63
        assert std_map["task_0"] == pytest.approx(np.std([6, 8, 10]))

    def test_compute_mean_std_across_tasks(self, tasks):
        scores = [
            make_score(tasks[0], 5.0),
            make_score(tasks[0], 7.0),  # std = 1.0
            make_score(tasks[1], 3.0),
            make_score(tasks[1], 9.0),  # std = 3.0
        ]
        mean_std = compute_mean_std_across_tasks(scores)

        expected = (np.std([5, 7]) + np.std([3, 9])) / 2
        assert mean_std == pytest.approx(expected)

    def test_compute_pairwise_correlations_with_ratings(self, tasks):
        scores_by_template = {
            "template_1": [make_score(t, i) for i, t in enumerate(tasks)],
            "template_2": [make_score(t, i) for i, t in enumerate(tasks)],
            "template_3": [make_score(t, len(tasks) - i) for i, t in enumerate(tasks)],
        }

        # Convert to unified format using scores_to_vector
        results = {
            tid: scores_to_vector(scores, tasks)
            for tid, scores in scores_by_template.items()
        }

        correlations = compute_pairwise_correlations(results, min_overlap=2)

        # 3 templates = C(3,2) = 3 pairs
        assert len(correlations) == 3

        # Find t1 vs t2 (should be perfect)
        t1_t2 = next(c for c in correlations
                     if {c["template_a"], c["template_b"]} == {"template_1", "template_2"})
        assert t1_t2["correlation"] == pytest.approx(1.0)

        # Find t1 vs t3 (should be negative)
        t1_t3 = next(c for c in correlations
                     if {c["template_a"], c["template_b"]} == {"template_1", "template_3"})
        assert t1_t3["correlation"] == pytest.approx(-1.0)


class TestBinaryCorrelation:
    """End-to-end tests for binary preference correlation computation."""

    @pytest.fixture
    def tasks(self):
        return [make_task(f"task_{i}") for i in range(4)]

    def testbuild_win_rate_vector(self, tasks):
        # Task 0 always beats task 1, task 2 always beats task 3
        measurements = [
            make_measurement(tasks[0], tasks[1], "a"),
            make_measurement(tasks[0], tasks[1], "a"),
            make_measurement(tasks[2], tasks[3], "a"),
            make_measurement(tasks[2], tasks[3], "b"),  # 50% win rate
        ]
        rates = build_win_rate_vector(measurements, tasks)

        # n=4 tasks gives C(4,2)=6 pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        assert len(rates) == 6
        assert rates[0] == pytest.approx(1.0)   # (0,1): 0 wins 100%
        assert rates[1] == pytest.approx(0.5)   # (0,2): no data, default
        assert rates[2] == pytest.approx(0.5)   # (0,3): no data, default
        assert rates[3] == pytest.approx(0.5)   # (1,2): no data, default
        assert rates[4] == pytest.approx(0.5)   # (1,3): no data, default
        assert rates[5] == pytest.approx(0.5)   # (2,3): 2 wins 50%

    def test_build_win_rate_handles_reversed_pairs(self, tasks):
        # When task_b < task_a, the choice should be flipped internally
        measurements = [
            make_measurement(tasks[1], tasks[0], "b"),  # task_0 wins (reversed)
        ]
        rates = build_win_rate_vector(measurements, tasks)

        # Pair (0,1) should show task 0 winning
        assert rates[0] == pytest.approx(1.0)

    def test_win_rate_correlation_perfect(self, tasks):
        # Create varied win rates to ensure non-zero variance
        # Pattern: (0,1)=100%, (0,2)=75%, (0,3)=50%, (1,2)=25%, (1,3)=0%, (2,3)=100%
        measurements_a = [
            make_measurement(tasks[0], tasks[1], "a"),
            make_measurement(tasks[0], tasks[1], "a"),  # (0,1) = 100%
            make_measurement(tasks[0], tasks[2], "a"),
            make_measurement(tasks[0], tasks[2], "a"),
            make_measurement(tasks[0], tasks[2], "a"),
            make_measurement(tasks[0], tasks[2], "b"),  # (0,2) = 75%
            make_measurement(tasks[0], tasks[3], "a"),
            make_measurement(tasks[0], tasks[3], "b"),  # (0,3) = 50%
            make_measurement(tasks[1], tasks[2], "a"),
            make_measurement(tasks[1], tasks[2], "b"),
            make_measurement(tasks[1], tasks[2], "b"),
            make_measurement(tasks[1], tasks[2], "b"),  # (1,2) = 25%
            make_measurement(tasks[1], tasks[3], "b"),
            make_measurement(tasks[1], tasks[3], "b"),  # (1,3) = 0%
            make_measurement(tasks[2], tasks[3], "a"),
            make_measurement(tasks[2], tasks[3], "a"),  # (2,3) = 100%
        ]
        # Identical pattern
        measurements_b = list(measurements_a)

        r = win_rate_correlation(measurements_a, measurements_b, tasks)
        assert r == pytest.approx(1.0)

    def test_win_rate_correlation_opposite(self, tasks):
        # Create varied win rates with variance
        measurements_a = [
            make_measurement(tasks[0], tasks[1], "a"),
            make_measurement(tasks[0], tasks[1], "a"),  # (0,1) = 100%
            make_measurement(tasks[0], tasks[2], "a"),
            make_measurement(tasks[0], tasks[2], "b"),  # (0,2) = 50%
            make_measurement(tasks[0], tasks[3], "b"),
            make_measurement(tasks[0], tasks[3], "b"),  # (0,3) = 0%
        ]
        # Opposite choices
        measurements_b = [
            make_measurement(tasks[0], tasks[1], "b"),
            make_measurement(tasks[0], tasks[1], "b"),  # (0,1) = 0%
            make_measurement(tasks[0], tasks[2], "b"),
            make_measurement(tasks[0], tasks[2], "a"),  # (0,2) = 50%
            make_measurement(tasks[0], tasks[3], "a"),
            make_measurement(tasks[0], tasks[3], "a"),  # (0,3) = 100%
        ]

        r = win_rate_correlation(measurements_a, measurements_b, tasks)
        assert r == pytest.approx(-1.0)

    def test_utility_vector_correlation_same_order(self, tasks):
        # Same utility ordering
        task_ids = [t.id for t in tasks]
        mu_a = np.array([1.0, 2.0, 3.0, 4.0])
        mu_b = np.array([2.0, 4.0, 6.0, 8.0])  # Same ordering, different scale

        r = utility_vector_correlation(mu_a, task_ids, mu_b, task_ids, min_overlap=2)
        assert r == pytest.approx(1.0)

    def test_utility_vector_correlation_reorders_tasks(self, tasks):
        # Tasks in different order but same underlying utilities
        ids_a = [t.id for t in tasks]
        ids_b = [tasks[3].id, tasks[2].id, tasks[1].id, tasks[0].id]

        mu_a = np.array([1.0, 2.0, 3.0, 4.0])
        mu_b = np.array([4.0, 3.0, 2.0, 1.0])  # Reversed order matches

        r = utility_vector_correlation(mu_a, ids_a, mu_b, ids_b, min_overlap=2)
        assert r == pytest.approx(1.0)

    def test_utility_vector_correlation_mismatched_tasks_returns_nan(self, tasks):
        ids_a = [t.id for t in tasks]
        ids_b = [f"other_{i}" for i in range(4)]

        mu_a = np.array([1.0, 2.0, 3.0, 4.0])
        mu_b = np.array([1.0, 2.0, 3.0, 4.0])

        r = utility_vector_correlation(mu_a, ids_a, mu_b, ids_b, min_overlap=2)
        assert np.isnan(r)

    def test_compute_pairwise_correlations_with_utilities(self, tasks):
        # Test unified function with utility vectors
        results = {
            "template_1": (np.array([1.0, 2.0, 3.0, 4.0]), [t.id for t in tasks]),
            "template_2": (np.array([1.0, 2.0, 3.0, 4.0]), [t.id for t in tasks]),
        }

        correlations = compute_pairwise_correlations(results, min_overlap=2)

        assert len(correlations) == 1
        assert correlations[0]["template_a"] == "template_1"
        assert correlations[0]["template_b"] == "template_2"
        # Identical data should give perfect correlation
        assert correlations[0]["correlation"] == pytest.approx(1.0)
