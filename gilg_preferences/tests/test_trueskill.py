"""Tests for TrueSkill fitting, sampling, and convergence."""

from itertools import combinations

import numpy as np
import pytest
from scipy.stats import spearmanr
import trueskill

from src.task_data import Task, OriginDataset
from src.types import RankingMeasurement, PreferenceType
from src.fitting.trueskill_fitting import fit_trueskill_from_rankings, sample_ranking_groups


pytestmark = [pytest.mark.measurement, pytest.mark.ranking]

DEFAULT_SIGMA = trueskill.Rating().sigma


@pytest.fixture
def sample_tasks():
    return [
        Task(prompt=f"Task {i}: Do something interesting", origin=OriginDataset.WILDCHAT, id=f"task_{i}", metadata={})
        for i in range(20)
    ]


class TestTrueSkillConvergence:
    """Test TrueSkill fitting converges with enough rankings."""

    def test_trueskill_ordering_stability(self, sample_tasks):
        """With consistent rankings, TrueSkill should produce stable ordering."""
        tasks = sample_tasks[:5]

        measurements = []
        for _ in range(20):
            measurements.append(RankingMeasurement(
                tasks=tasks,
                ranking=[0, 1, 2, 3, 4],
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)

        final_ranking = result.ranking()
        assert [t.id for t in final_ranking] == ["task_0", "task_1", "task_2", "task_3", "task_4"]

        for task in result.tasks:
            assert result.uncertainty(task) < DEFAULT_SIGMA

    def test_trueskill_handles_ties(self, sample_tasks):
        """TrueSkill should handle conflicting rankings gracefully."""
        tasks = sample_tasks[:3]

        measurements = [
            RankingMeasurement(tasks=tasks, ranking=[0, 1, 2], preference_type=PreferenceType.PRE_TASK_RANKING),
            RankingMeasurement(tasks=tasks, ranking=[2, 1, 0], preference_type=PreferenceType.PRE_TASK_RANKING),
            RankingMeasurement(tasks=tasks, ranking=[1, 0, 2], preference_type=PreferenceType.PRE_TASK_RANKING),
        ]

        result = fit_trueskill_from_rankings(measurements)

        assert result.n_observations == 3
        assert len(result.ranking()) == 3
        for task in result.tasks:
            assert result.uncertainty(task) > 3.0


class TestSamplingCoverage:
    """Test that sampling produces balanced task coverage."""

    def test_balanced_coverage(self, sample_tasks):
        """Each task should appear roughly equally across groups."""
        rng = np.random.default_rng(42)
        groups = sample_ranking_groups(sample_tasks, n_tasks_per_group=5, n_groups=40, rng=rng)

        counts = {}
        for group in groups:
            for t in group:
                counts[t.id] = counts.get(t.id, 0) + 1

        assert len(counts) == 20

        # Expected: 40 groups × 5 tasks / 20 unique = 10 appearances each
        values = list(counts.values())
        assert min(values) >= 5   # At least half of expected
        assert max(values) <= 15  # At most 1.5× expected

    def test_deterministic_with_seed(self, sample_tasks):
        """Same seed should produce same groups."""
        groups1 = sample_ranking_groups(sample_tasks, 5, 10, np.random.default_rng(42))
        groups2 = sample_ranking_groups(sample_tasks, 5, 10, np.random.default_rng(42))

        for g1, g2 in zip(groups1, groups2):
            assert [t.id for t in g1] == [t.id for t in g2]


class TestTrueSkillRevealedCorrelation:
    """E2E test: TrueSkill rankings should correlate with pairwise revealed preferences."""

    def test_trueskill_predicts_pairwise_choices(self, sample_tasks):
        """TrueSkill utilities should predict pairwise comparison outcomes."""
        tasks = sample_tasks[:10]
        true_utilities = {t.id: 10 - i for i, t in enumerate(tasks)}

        rng = np.random.default_rng(42)
        task_groups = sample_ranking_groups(tasks, n_tasks_per_group=5, n_groups=20, rng=rng)

        measurements = []
        for group in task_groups:
            noise = rng.normal(0, 0.5, len(group))
            scores = [true_utilities[t.id] + n for t, n in zip(group, noise)]
            ranking = sorted(range(len(group)), key=lambda i: scores[i], reverse=True)
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)

        fitted_utilities = [result.utility(t) for t in tasks]
        true_util_list = [true_utilities[t.id] for t in tasks]
        correlation, p_value = spearmanr(fitted_utilities, true_util_list)

        assert correlation > 0.8, f"Correlation {correlation:.3f} too low"
        assert p_value < 0.01, f"p-value {p_value:.4f} too high"

    def test_trueskill_agrees_with_pairwise_comparisons(self, sample_tasks):
        """TrueSkill ranking should agree with majority of pairwise comparisons."""
        tasks = sample_tasks[:8]
        true_ranking = {t.id: 8 - i for i, t in enumerate(tasks)}

        rng = np.random.default_rng(123)
        task_groups = sample_ranking_groups(tasks, n_tasks_per_group=4, n_groups=30, rng=rng)

        measurements = []
        for group in task_groups:
            if rng.random() < 0.9:
                ranking = sorted(range(len(group)), key=lambda i: true_ranking[group[i].id], reverse=True)
            else:
                ranking = list(rng.permutation(len(group)))
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)

        all_pairs = list(combinations(tasks, 2))
        agreements = 0
        for task_a, task_b in all_pairs:
            gt_winner = task_a.id if true_ranking[task_a.id] > true_ranking[task_b.id] else task_b.id
            ts_winner = task_a.id if result.utility(task_a) > result.utility(task_b) else task_b.id
            if gt_winner == ts_winner:
                agreements += 1

        agreement_rate = agreements / len(all_pairs)
        assert agreement_rate > 0.85, f"Agreement rate {agreement_rate:.2%} too low"
