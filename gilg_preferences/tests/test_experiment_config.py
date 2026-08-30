"""Tests for experiment config."""

import pytest

pytestmark = [pytest.mark.runners, pytest.mark.thurstonian]

from pathlib import Path

import numpy as np

from src.measurement.runners.config import ExperimentConfig
from src.task_data.consistency import load_consistency_index
from src.measurement.runners.utils.experiment_utils import (
    shuffle_pair_order,
    apply_pair_order,
    flip_pairs,
)
from src.task_data import OriginDataset, Task
from src.types import BinaryPreferenceMeasurement, PreferenceType
from src.fitting.thurstonian_fitting import PairwiseData, fit_thurstonian


def make_tasks(n: int) -> list[Task]:
    return [
        Task(id=f"task_{i}", prompt=f"Task {i}", origin=OriginDataset.SYNTHETIC, metadata={})
        for i in range(n)
    ]


class TestPairOrderSeedValidation:

    def test_cannot_set_both_pair_order_seed_and_reverse(self):
        """Setting both pair_order_seed and include_reverse_order raises error."""
        with pytest.raises(ValueError, match="Cannot set pair_order_seed when include_reverse_order=True"):
            ExperimentConfig(
                preference_mode="pre_task_revealed",
                model="llama-3.1-8b",
                n_tasks=5,
                task_origins=["wildchat"],
                templates=Path("dummy.yaml"),
                pair_order_seed=42,
                include_reverse_order=True,
            )

    def test_pair_order_seed_without_reverse_is_valid(self):
        """pair_order_seed alone is valid."""
        config = ExperimentConfig(
            preference_mode="pre_task_revealed",
            model="llama-3.1-8b",
            n_tasks=5,
            task_origins=["wildchat"],
            templates=Path("dummy.yaml"),
            pair_order_seed=42,
            include_reverse_order=False,
        )
        assert config.pair_order_seed == 42

    def test_reverse_without_pair_order_seed_is_valid(self):
        """include_reverse_order alone is valid."""
        config = ExperimentConfig(
            preference_mode="pre_task_revealed",
            model="llama-3.1-8b",
            n_tasks=5,
            task_origins=["wildchat"],
            templates=Path("dummy.yaml"),
            pair_order_seed=None,
            include_reverse_order=True,
        )
        assert config.include_reverse_order is True

    def test_neither_option_set_defaults_to_shuffle(self):
        """Neither option set defaults to shuffling with seed 0."""
        config = ExperimentConfig(
            preference_mode="pre_task_revealed",
            model="llama-3.1-8b",
            n_tasks=5,
            task_origins=["wildchat"],
            templates=Path("dummy.yaml"),
        )
        assert config.pair_order_seed == 0  # Defaults to shuffle with seed 0
        assert config.include_reverse_order is False


class TestShufflePairOrder:

    def test_shuffle_is_deterministic(self):
        """Same seed produces same shuffle."""
        tasks = make_tasks(4)
        pairs = [(tasks[0], tasks[1]), (tasks[0], tasks[2]), (tasks[1], tasks[2])]

        result1 = shuffle_pair_order(pairs, seed=42)
        result2 = shuffle_pair_order(pairs, seed=42)

        assert [(a.id, b.id) for a, b in result1] == [(a.id, b.id) for a, b in result2]

    def test_different_seeds_produce_different_orders(self):
        """Different seeds produce different shuffles (with high probability)."""
        tasks = make_tasks(10)
        pairs = [(tasks[i], tasks[j]) for i in range(10) for j in range(i + 1, 10)]

        result1 = shuffle_pair_order(pairs, seed=42)
        result2 = shuffle_pair_order(pairs, seed=123)

        # With 45 pairs and 50% flip probability, it's extremely unlikely to get same result
        assert [(a.id, b.id) for a, b in result1] != [(a.id, b.id) for a, b in result2]

    def test_shuffle_flips_approximately_half(self):
        """Shuffle flips roughly half the pairs."""
        tasks = make_tasks(20)
        pairs = [(tasks[i], tasks[j]) for i in range(20) for j in range(i + 1, 20)]

        result = shuffle_pair_order(pairs, seed=42)

        flipped = sum(1 for (a1, b1), (a2, b2) in zip(pairs, result) if a1.id != a2.id)
        # With 190 pairs, expect ~95 flipped, allow wide margin
        assert 50 < flipped < 140


class TestApplyPairOrder:

    def test_canonical_with_reverse_order_returns_unchanged(self):
        """Canonical order with include_reverse_order=True returns pairs unchanged."""
        tasks = make_tasks(3)
        pairs = [(tasks[0], tasks[1]), (tasks[0], tasks[2])]

        result = apply_pair_order(pairs, order="canonical", pair_order_seed=None, include_reverse_order=True)
        assert [(a.id, b.id) for a, b in result] == [(a.id, b.id) for a, b in pairs]

    def test_reversed_flips_all(self):
        """Reversed order flips all pairs."""
        tasks = make_tasks(3)
        pairs = [(tasks[0], tasks[1]), (tasks[0], tasks[2])]

        result = apply_pair_order(pairs, order="reversed", pair_order_seed=None, include_reverse_order=True)
        assert [(a.id, b.id) for a, b in result] == [(b.id, a.id) for a, b in pairs]

    def test_shuffles_when_not_using_reverse_order(self):
        """When include_reverse_order=False, pairs are shuffled."""
        tasks = make_tasks(10)
        pairs = [(tasks[i], tasks[j]) for i in range(10) for j in range(i + 1, 10)]

        result = apply_pair_order(pairs, order="canonical", pair_order_seed=42, include_reverse_order=False)

        # Result should have some pairs flipped (not all same as original)
        original_ids = [(a.id, b.id) for a, b in pairs]
        result_ids = [(a.id, b.id) for a, b in result]
        assert result_ids != original_ids  # Should be shuffled


class TestPairOrderInMeasurements:
    """Verify that shuffled pair order is correctly reflected in measurements."""

    @pytest.mark.asyncio
    async def test_measurement_records_correct_winner_after_shuffle(self):
        """When pairs are shuffled, measurements still record the correct winner."""
        from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
        from src.measurement.elicitation.response_format import RegexChoiceFormat
        from src.measurement.elicitation.prompt_templates.builders import PreTaskRevealedPromptBuilder
        from src.measurement.elicitation.prompt_templates.template import PromptTemplate

        tasks = make_tasks(2)
        task_x, task_y = tasks[0], tasks[1]

        # Create a simple template
        template = PromptTemplate(
            template="Compare:\nTask A: {task_a}\nTask B: {task_b}\n{format_instruction}",
            name="test",
            required_placeholders=frozenset(["task_a", "task_b", "format_instruction"]),
            tags=frozenset(),
        )
        response_format = RegexChoiceFormat("Task A", "Task B")
        measurer = RevealedPreferenceMeasurer()
        builder = PreTaskRevealedPromptBuilder(measurer, response_format, template)

        # Original order: (task_x, task_y) - task_x is "Task A"
        prompt_original = builder.build(task_x, task_y)
        assert prompt_original.tasks[0].id == "task_0"  # task_x is Task A
        assert prompt_original.tasks[1].id == "task_1"  # task_y is Task B

        # Shuffled order: (task_y, task_x) - task_y is now "Task A"
        prompt_shuffled = builder.build(task_y, task_x)
        assert prompt_shuffled.tasks[0].id == "task_1"  # task_y is Task A
        assert prompt_shuffled.tasks[1].id == "task_0"  # task_x is Task B

        # Model responds "Task A" in both cases
        response_text = "Task A"

        # In original order: "Task A" means task_x wins
        result_original = await measurer.parse(response_text, prompt_original)
        assert result_original.result.task_a.id == "task_0"
        assert result_original.result.choice == "a"
        # Winner is task_x (task_0)

        # In shuffled order: "Task A" means task_y wins
        result_shuffled = await measurer.parse(response_text, prompt_shuffled)
        assert result_shuffled.result.task_a.id == "task_1"
        assert result_shuffled.result.choice == "a"
        # Winner is task_y (task_1)

    def test_thurstonian_aggregates_shuffled_measurements_correctly(self):
        """Thurstonian model correctly aggregates measurements from shuffled pairs."""
        tasks = make_tasks(3)

        # Simulate measurements where task_0 always wins
        # Some pairs are in canonical order, some are shuffled
        comparisons = [
            # Canonical: (task_0, task_1), task_0 shown as A, model says A
            BinaryPreferenceMeasurement(
                task_a=tasks[0], task_b=tasks[1], choice="a",
                preference_type=PreferenceType.PRE_TASK_REVEALED
            ),
            # Shuffled: (task_1, task_0), task_1 shown as A, model says B (meaning task_0 wins)
            BinaryPreferenceMeasurement(
                task_a=tasks[1], task_b=tasks[0], choice="b",
                preference_type=PreferenceType.PRE_TASK_REVEALED
            ),
            # Canonical: (task_0, task_2), task_0 shown as A, model says A
            BinaryPreferenceMeasurement(
                task_a=tasks[0], task_b=tasks[2], choice="a",
                preference_type=PreferenceType.PRE_TASK_REVEALED
            ),
            # Shuffled: (task_2, task_0), task_2 shown as A, model says B (meaning task_0 wins)
            BinaryPreferenceMeasurement(
                task_a=tasks[2], task_b=tasks[0], choice="b",
                preference_type=PreferenceType.PRE_TASK_REVEALED
            ),
            # task_1 vs task_2: task_1 wins
            BinaryPreferenceMeasurement(
                task_a=tasks[1], task_b=tasks[2], choice="a",
                preference_type=PreferenceType.PRE_TASK_REVEALED
            ),
        ]

        data = PairwiseData.from_comparisons(comparisons, tasks)

        # task_0 should have 4 wins (2 pairs x 2 measurements each showing task_0 winning)
        # Actually: task_0 beats task_1 twice, task_0 beats task_2 twice = 4 wins
        idx_0 = next(i for i, t in enumerate(data.tasks) if t.id == "task_0")
        assert data.wins[idx_0, :].sum() == 4

        # task_1 has 1 win (over task_2)
        idx_1 = next(i for i, t in enumerate(data.tasks) if t.id == "task_1")
        assert data.wins[idx_1, :].sum() == 1

        # task_2 has 0 wins
        idx_2 = next(i for i, t in enumerate(data.tasks) if t.id == "task_2")
        assert data.wins[idx_2, :].sum() == 0


class TestShuffledTasksInThurstonian:
    """Verify Thurstonian fitting uses task IDs correctly regardless of order."""

    def _make_comparisons(
        self, tasks: list[Task], winner_indices: list[tuple[int, int]]
    ) -> list[BinaryPreferenceMeasurement]:
        """Create comparisons where first index always wins."""
        comparisons = []
        for i, j in winner_indices:
            comparisons.append(BinaryPreferenceMeasurement(
                task_a=tasks[i],
                task_b=tasks[j],
                choice="a",
                preference_type=PreferenceType.PRE_TASK_REVEALED,
            ))
        return comparisons

    def test_thurstonian_uses_task_ids_not_positions(self):
        """Fitting with shuffled task list gives same ranking by task ID.

        Note: Absolute utilities differ because the model fixes the first task's
        utility to 0 for identification. But relative utilities (and thus rankings)
        should be identical.
        """
        tasks = make_tasks(4)

        # Task 0 beats everyone, task 1 beats 2 and 3, task 2 beats 3
        # Expected ranking: 0 > 1 > 2 > 3
        winner_pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 3),
        ]
        # Repeat each comparison to get stable estimates
        winner_pairs = winner_pairs * 5
        comparisons = self._make_comparisons(tasks, winner_pairs)

        # Fit with original order
        data_original = PairwiseData.from_comparisons(comparisons, tasks)
        result_original = fit_thurstonian(data_original, max_iter=2000)

        # Shuffle tasks (reverse order)
        tasks_reversed = list(reversed(tasks))
        data_shuffled = PairwiseData.from_comparisons(comparisons, tasks_reversed)
        result_shuffled = fit_thurstonian(data_shuffled, max_iter=2000)

        # Ranking should be the same regardless of task order
        ranking_original = [t.id for t in result_original.ranking()]
        ranking_shuffled = [t.id for t in result_shuffled.ranking()]
        assert ranking_original == ranking_shuffled

        # Pairwise preference probabilities should match (within optimization tolerance)
        for i, task_i in enumerate(tasks):
            for task_j in tasks[i + 1:]:
                prob_original = result_original.preference_probability(task_i, task_j)
                prob_shuffled = result_shuffled.preference_probability(task_i, task_j)
                assert np.isclose(prob_original, prob_shuffled, atol=0.02), (
                    f"P({task_i.id} > {task_j.id}): original={prob_original:.3f}, shuffled={prob_shuffled:.3f}"
                )

    def test_pairwise_data_uses_task_ids(self):
        """PairwiseData.from_comparisons maps by task ID."""
        tasks = make_tasks(3)
        comparisons = self._make_comparisons(tasks, [(0, 1), (0, 2), (1, 2)])

        # Original order
        data_original = PairwiseData.from_comparisons(comparisons, tasks)

        # Reversed order
        tasks_reversed = list(reversed(tasks))
        data_reversed = PairwiseData.from_comparisons(comparisons, tasks_reversed)

        # Win counts should be consistent: task_0 has 2 wins, task_1 has 1 win, task_2 has 0 wins
        # In original order [0,1,2]: wins[0,1]=1, wins[0,2]=1, wins[1,2]=1
        # In reversed order [2,1,0]: wins[2,1]=1, wins[2,0]=1, wins[1,0]=1

        # Get win counts per task
        def get_wins_for_task(data: PairwiseData, task_id: str) -> int:
            idx = next(i for i, t in enumerate(data.tasks) if t.id == task_id)
            return int(data.wins[idx, :].sum())

        for task in tasks:
            wins_original = get_wins_for_task(data_original, task.id)
            wins_reversed = get_wins_for_task(data_reversed, task.id)
            assert wins_original == wins_reversed, (
                f"Task {task.id}: wins differ (original={wins_original}, reversed={wins_reversed})"
            )


class TestConsistencyFilterConfig:

    def test_config_accepts_consistency_filter_fields(self):
        config = ExperimentConfig(
            preference_mode="post_task_stated",
            model="gemma-2-27b",
            n_tasks=100,
            consistency_filter_model="gemma2",
            consistency_keep_ratio=0.7,
        )
        assert config.consistency_filter_model == "gemma2"
        assert config.consistency_keep_ratio == 0.7

    def test_config_defaults_no_consistency_filter(self):
        config = ExperimentConfig(
            preference_mode="post_task_stated",
            model="gemma-2-27b",
            n_tasks=100,
        )
        assert config.consistency_filter_model is None
        assert config.consistency_keep_ratio == 0.7  # default

    def test_available_consistency_models(self):
        """Verify we have consistency indices for expected models."""
        for model in ["gemma2", "qwen_think", "claude_haiku"]:
            index = load_consistency_index(model)
            assert len(index.scores) > 0
            assert len(index.percentiles) > 0
