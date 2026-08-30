"""End-to-end tests for ranking measurement workflows.

Tests the full pipeline: sampling -> measurement -> TrueSkill fitting.
Uses mock clients to avoid API calls.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from src.task_data import Task, OriginDataset
from src.types import RankingMeasurement, PreferenceType
from src.measurement.elicitation.response_format import get_ranking_response_format
from src.measurement.elicitation.measurer import RankingMeasurer
from src.measurement.elicitation.prompt_templates import PreTaskRankingPromptBuilder, PostTaskRankingPromptBuilder, PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS
from src.fitting.trueskill_fitting import TrueSkillResult, fit_trueskill_from_rankings, sample_ranking_groups
from src.measurement.storage import RankingCache
from src.models.openai_compatible import BatchResult


pytestmark = [pytest.mark.measurement, pytest.mark.ranking]


@pytest.fixture
def sample_tasks():
    return [
        Task(prompt=f"Task {i}: Do something interesting", origin=OriginDataset.WILDCHAT, id=f"task_{i}", metadata={})
        for i in range(20)
    ]


@pytest.fixture
def task_lookup(sample_tasks):
    return {t.id: t for t in sample_tasks}


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.canonical_model_name = "test-model"
    client.model_name = "test-model"
    return client


@pytest.fixture
def ranking_template():
    return PromptTemplate(
        name="test_ranking_template",
        template=(
            "Rank these tasks from most to least preferred.\n\n"
            "{format_instruction}\n\n"
            "Task A: {task_a}\n"
            "Task B: {task_b}\n"
            "Task C: {task_c}\n"
            "Task D: {task_d}\n"
            "Task E: {task_e}"
        ),
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_ranking"],
    )


class TestPreTaskRankingE2E:
    """End-to-end test of pre-task ranking workflow."""

    @pytest.mark.asyncio
    async def test_full_ranking_pipeline(self, sample_tasks, task_lookup, temp_cache_dir, mock_client, ranking_template):
        """Full pipeline: sample groups -> measure rankings -> fit TrueSkill."""
        rng = np.random.default_rng(42)
        task_groups = sample_ranking_groups(sample_tasks, n_tasks_per_group=5, n_groups=10, rng=rng)

        assert len(task_groups) == 10
        for group in task_groups:
            assert len(group) == 5
            assert len(set(t.id for t in group)) == 5

        def make_ranking_response(group):
            sorted_indices = sorted(range(5), key=lambda i: group[i].id)
            labels = [chr(65 + i) for i in sorted_indices]
            return " > ".join(labels)

        responses = [[BatchResult(response=make_ranking_response(g), error=None)] for g in task_groups]
        mock_client.generate_batch_async = AsyncMock(side_effect=responses)

        task_labels = ("A", "B", "C", "D", "E")
        response_format = get_ranking_response_format(task_labels, "regex")
        builder = PreTaskRankingPromptBuilder(
            measurer=RankingMeasurer(),
            response_format=response_format,
            template=ranking_template,
        )

        measurements = []
        for group in task_groups:
            prompt = builder.build(group)
            result = await mock_client.generate_batch_async([{"messages": prompt.messages}])
            response_text = result[0].response
            ranking = await response_format.parse(response_text)
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        assert len(measurements) == 10

        result = fit_trueskill_from_rankings(measurements)

        assert isinstance(result, TrueSkillResult)
        assert result.n_observations == 10
        assert len(result.tasks) == 20

        task_0 = next(t for t in result.tasks if t.id == "task_0")
        task_19 = next(t for t in result.tasks if t.id == "task_19")
        assert result.utility(task_0) > result.utility(task_19)

        final_ranking = result.ranking()
        assert len(final_ranking) == 20
        assert any(t.id == "task_0" for t in final_ranking[:5])

    @pytest.mark.asyncio
    async def test_ranking_with_cache(self, sample_tasks, task_lookup, temp_cache_dir, mock_client, ranking_template):
        """Test caching behavior: second run should use cached results."""
        with patch.object(RankingCache, "CACHE_DIR", temp_cache_dir):
            cache = RankingCache("test-model")

            rng = np.random.default_rng(42)
            task_groups = sample_ranking_groups(sample_tasks[:10], n_tasks_per_group=5, n_groups=4, rng=rng)

            measurements = []
            for group in task_groups:
                ranking = list(range(5))
                measurements.append(RankingMeasurement(
                    tasks=group,
                    ranking=ranking,
                    preference_type=PreferenceType.PRE_TASK_RANKING,
                ))

            cache.add(measurements, "test_template", "regex", seed=42)

            existing = cache.get_measured_groups("test_template", "regex", seed=42)
            assert len(existing) == 4

            loaded = cache.get_all_measurements("test_template", "regex", seed=42, task_lookup=task_lookup)
            assert len(loaded) == 4

            result = fit_trueskill_from_rankings(loaded)
            assert result.n_observations == 4


class TestPostTaskRankingE2E:
    """End-to-end test of post-task ranking with completions."""

    @pytest.mark.asyncio
    async def test_post_task_ranking_pipeline(self, sample_tasks, mock_client):
        """Full post-task ranking: tasks + completions -> ranking -> fit."""
        tasks = sample_tasks[:10]
        completions = {t.id: f"Completion for {t.id}" for t in tasks}

        template = PromptTemplate(
            name="test_post_ranking",
            template=(
                "You completed these tasks. Rank them by how much you enjoyed them.\n\n"
                "{format_instruction}\n\n"
                "Task A: {task_a}\nYour response: {completion_a}\n\n"
                "Task B: {task_b}\nYour response: {completion_b}\n\n"
                "Task C: {task_c}\nYour response: {completion_c}\n\n"
                "Task D: {task_d}\nYour response: {completion_d}\n\n"
                "Task E: {task_e}\nYour response: {completion_e}"
            ),
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["post_task_ranking"],
        )

        task_labels = ("A", "B", "C", "D", "E")
        response_format = get_ranking_response_format(task_labels, "regex")
        builder = PostTaskRankingPromptBuilder(
            measurer=RankingMeasurer(),
            response_format=response_format,
            template=template,
        )

        rng = np.random.default_rng(42)
        task_groups = sample_ranking_groups(tasks, n_tasks_per_group=5, n_groups=4, rng=rng)

        responses = [[BatchResult(response="A > B > C > D > E", error=None)] for _ in task_groups]
        mock_client.generate_batch_async = AsyncMock(side_effect=responses)

        measurements = []
        for group in task_groups:
            group_completions = [completions[t.id] for t in group]
            prompt = builder.build(group, group_completions)

            prompt_text = str(prompt.messages)
            assert "Completion for" in prompt_text

            result = await mock_client.generate_batch_async([{"messages": prompt.messages}])
            ranking = await response_format.parse(result[0].response)
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.POST_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)
        assert result.n_observations == 4
