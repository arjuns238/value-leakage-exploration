"""End-to-end tests for post-task measurement pipeline.

Tests the full workflow: completion generation -> storage -> measurement.
Uses a mock client to avoid API calls.

Run with:
    pytest tests/test_post_task_pipeline.py -v
"""

import pytest

pytestmark = [pytest.mark.measurement, pytest.mark.cache]

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.task_data import Task, OriginDataset
from src.measurement.storage.completions import (
    CompletionStore,
    TaskCompletion,
    generate_completions,
    load_completions,
    completions_exist,
)
from src.measurement.storage import save_yaml, load_yaml
from src.measurement.elicitation import (
    measure_post_task_stated,
    measure_post_task_revealed,
    StatedScoreMeasurer,
    RevealedPreferenceMeasurer,
    RATING_FORMATS,
    CHOICE_FORMATS,
)
from src.measurement.elicitation.prompt_templates import (
    PostTaskStatedPromptBuilder,
    PostTaskRevealedPromptBuilder,
    PromptTemplate,
    TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.types import TaskScore, BinaryPreferenceMeasurement, PreferenceType
from src.models.openai_compatible import BatchResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tasks():
    return [
        Task(prompt="What is 2 + 2?", origin=OriginDataset.MATH, id="math_1", metadata={}),
        Task(prompt="Write a haiku about rain.", origin=OriginDataset.WILDCHAT, id="creative_1", metadata={}),
        Task(prompt="Explain gravity.", origin=OriginDataset.WILDCHAT, id="science_1", metadata={}),
    ]


@pytest.fixture
def task_lookup(sample_tasks):
    return {t.id: t for t in sample_tasks}


@pytest.fixture
def temp_results_dir():
    """Temporary directory for test results."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_client():
    """Mock client that returns predictable completions and ratings."""
    client = MagicMock()
    client.canonical_model_name = "test-model"
    client.model_name = "test-model"
    return client


@pytest.fixture
def sample_template():
    """A simple template for testing cache operations."""
    from src.measurement.elicitation.prompt_templates.template import PromptTemplate
    return PromptTemplate(
        name="test_template",
        template="Rate this: {task}",
        required_placeholders=frozenset({"task"}),
        tags=frozenset({"test_tag:value"}),
    )


# =============================================================================
# CompletionStore Tests
# =============================================================================


class TestCompletionStore:
    """Test completion storage and retrieval."""

    def test_store_and_load_completions(self, sample_tasks, task_lookup, temp_results_dir, mock_client):
        """Should store completions and load them back."""
        with patch("src.measurement.storage.completions.COMPLETIONS_DIR", temp_results_dir):
            store = CompletionStore(client=mock_client, seed=42)

            completions = [
                TaskCompletion(task=sample_tasks[0], completion="The answer is 4."),
                TaskCompletion(task=sample_tasks[1], completion="Rain falls gently down\nPuddles form on the wet ground\nNature's soft refrain"),
            ]

            store.save(completions, config={"model": "test-model", "seed": 42})

            assert store.exists()
            assert store.get_existing_task_ids() == {"math_1", "creative_1"}

            loaded = store.load(task_lookup)
            assert len(loaded) == 2
            assert loaded[0].task.id == "math_1"
            assert loaded[0].completion == "The answer is 4."

    def test_incremental_save(self, sample_tasks, task_lookup, temp_results_dir, mock_client):
        """Should append new completions without duplicating existing ones."""
        with patch("src.measurement.storage.completions.COMPLETIONS_DIR", temp_results_dir):
            store = CompletionStore(client=mock_client, seed=42)

            # First save
            completions1 = [TaskCompletion(task=sample_tasks[0], completion="Answer 1")]
            store.save(completions1, config={"model": "test-model", "seed": 42})

            # Second save with overlap
            completions2 = [
                TaskCompletion(task=sample_tasks[0], completion="Should be ignored"),
                TaskCompletion(task=sample_tasks[1], completion="Answer 2"),
            ]
            store.save(completions2, config={"model": "test-model", "seed": 42})

            loaded = store.load(task_lookup)
            assert len(loaded) == 2
            # Original should be kept, not replaced
            assert loaded[0].completion == "Answer 1"
            assert loaded[1].completion == "Answer 2"

    def test_load_without_task_lookup(self, sample_tasks, temp_results_dir, mock_client):
        """Should reconstruct Task objects from stored data when no lookup provided."""
        with patch("src.measurement.storage.completions.COMPLETIONS_DIR", temp_results_dir):
            store = CompletionStore(client=mock_client, seed=42)

            completions = [TaskCompletion(task=sample_tasks[0], completion="Test")]
            store.save(completions, config={"model": "test-model", "seed": 42})

            loaded = store.load(task_lookup=None)
            assert len(loaded) == 1
            assert loaded[0].task.id == "math_1"
            assert loaded[0].task.prompt == "What is 2 + 2?"
            assert loaded[0].task.origin == OriginDataset.MATH


class TestGenerateCompletions:
    """Test completion generation with mock client."""

    def test_generates_completions_for_all_tasks(self, sample_tasks, mock_client):
        """Should generate completions for all tasks and return successful ones."""
        mock_client.generate_batch.return_value = [
            BatchResult(response="Answer 1", error=None),
            BatchResult(response="Answer 2", error=None),
            BatchResult(response="Answer 3", error=None),
        ]

        completions = generate_completions(
            client=mock_client,
            tasks=sample_tasks,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(completions) == 3
        assert completions[0].task.id == "math_1"
        assert completions[0].completion == "Answer 1"

    def test_handles_failed_requests(self, sample_tasks, mock_client):
        """Should skip failed requests and return only successful completions."""
        mock_client.generate_batch.return_value = [
            BatchResult(response="Answer 1", error=None),
            BatchResult(response=None, error=Exception("API error")),
            BatchResult(response="Answer 3", error=None),
        ]

        completions = generate_completions(
            client=mock_client,
            tasks=sample_tasks,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(completions) == 2
        assert completions[0].task.id == "math_1"
        assert completions[1].task.id == "science_1"


# =============================================================================
# Post-Task Stated Measurement Tests
# =============================================================================


class TestPostTaskStatedMeasurement:
    """Test post-task stated preference measurement."""

    def test_measures_post_task_ratings(self, sample_tasks, mock_client, post_task_stated_template_fixture):
        """Should measure ratings after task completion."""
        # Mock returns valid rating responses - one per call to generate_batch_async
        responses = [
            [BatchResult(response="I rate this task 7 out of 10.", error=None)],
            [BatchResult(response="Rating: 8", error=None)],
        ]
        mock_client.generate_batch_async = AsyncMock(side_effect=responses)

        data = [
            (sample_tasks[0], "The answer is 4."),
            (sample_tasks[1], "Rain haiku here."),
        ]

        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 10),
            template=post_task_stated_template_fixture,
        )

        batch = measure_post_task_stated(
            client=mock_client,
            data=data,
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(batch.successes) == 2
        assert isinstance(batch.successes[0], TaskScore)
        assert batch.successes[0].score == 7.0
        assert batch.successes[1].score == 8.0
        assert batch.successes[0].preference_type == PreferenceType.POST_TASK_STATED

    def test_prompt_structure_is_multi_turn(self, sample_tasks, post_task_stated_template_fixture):
        """Post-task prompt should have 3 messages: user, assistant, user."""
        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 10),
            template=post_task_stated_template_fixture,
        )

        prompt = builder.build(sample_tasks[0], "The answer is 4.")

        assert len(prompt.messages) == 3
        assert prompt.messages[0]["role"] == "user"
        assert prompt.messages[1]["role"] == "assistant"
        assert prompt.messages[2]["role"] == "user"
        assert sample_tasks[0].prompt in prompt.messages[0]["content"]
        assert "The answer is 4." in prompt.messages[1]["content"]


# =============================================================================
# Post-Task Revealed Measurement Tests
# =============================================================================


class TestPostTaskRevealedMeasurement:
    """Test post-task revealed preference measurement."""

    def test_measures_post_task_binary_choice(self, sample_tasks, mock_client, post_task_revealed_template_fixture):
        """Should measure binary preference after completing both tasks."""
        mock_client.generate_batch_async = AsyncMock(return_value=[
            BatchResult(response="I preferred Task A.", error=None),
        ])

        data = [
            (sample_tasks[0], sample_tasks[1], "Answer to math", "Answer to creative"),
        ]

        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS["regex"]("Task A", "Task B"),
            template=post_task_revealed_template_fixture,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=data,
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(batch.successes) == 1
        assert isinstance(batch.successes[0], BinaryPreferenceMeasurement)
        assert batch.successes[0].choice == "a"
        assert batch.successes[0].preference_type == PreferenceType.POST_TASK_REVEALED

    def test_prompt_structure_is_five_turn(self, sample_tasks, post_task_revealed_template_fixture):
        """Post-task revealed prompt should have 5 messages."""
        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS["regex"]("Task A", "Task B"),
            template=post_task_revealed_template_fixture,
        )

        prompt = builder.build(
            sample_tasks[0], sample_tasks[1],
            "Completion A", "Completion B"
        )

        assert len(prompt.messages) == 5
        assert prompt.messages[0]["role"] == "user"
        assert prompt.messages[1]["role"] == "assistant"
        assert prompt.messages[2]["role"] == "user"
        assert prompt.messages[3]["role"] == "assistant"
        assert prompt.messages[4]["role"] == "user"


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestFullPostTaskPipeline:
    """End-to-end tests for the complete post-task measurement workflow."""

    def test_completion_to_stated_measurement_pipeline(
        self, sample_tasks, task_lookup, temp_results_dir, mock_client, post_task_stated_template_fixture
    ):
        """Full pipeline: generate completions -> store -> load -> measure stated."""
        with patch("src.measurement.storage.completions.COMPLETIONS_DIR", temp_results_dir):
            # Step 1: Generate completions
            mock_client.generate_batch.return_value = [
                BatchResult(response="4", error=None),
                BatchResult(response="Haiku here", error=None),
                BatchResult(response="Gravity pulls", error=None),
            ]

            completions = generate_completions(
                client=mock_client, tasks=sample_tasks,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            # Step 2: Store completions
            store = CompletionStore(client=mock_client, seed=42)
            store.save(completions, config={"model": "test-model", "seed": 42})

            # Step 3: Load completions
            loaded = store.load(task_lookup)
            assert len(loaded) == 3

            # Step 4: Measure stated preferences
            mock_client.generate_batch_async = AsyncMock(side_effect=[
                [BatchResult(response="Rating: 7", error=None)],
                [BatchResult(response="I give it an 8", error=None)],
                [BatchResult(response="9 out of 10", error=None)],
            ])

            data = [(tc.task, tc.completion) for tc in loaded]
            builder = PostTaskStatedPromptBuilder(
                measurer=StatedScoreMeasurer(),
                response_format=RATING_FORMATS["regex"](1, 10),
                template=post_task_stated_template_fixture,
            )

            batch = measure_post_task_stated(
                client=mock_client, data=data, builder=builder,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            assert len(batch.successes) == 3
            scores = [s.score for s in batch.successes]
            assert scores == [7.0, 8.0, 9.0]

    def test_completion_to_revealed_measurement_pipeline(
        self, sample_tasks, task_lookup, temp_results_dir, mock_client, post_task_revealed_template_fixture
    ):
        """Full pipeline: generate completions -> store -> load -> measure revealed."""
        with patch("src.measurement.storage.completions.COMPLETIONS_DIR", temp_results_dir):
            # Step 1: Generate completions
            mock_client.generate_batch.return_value = [
                BatchResult(response="Completion A", error=None),
                BatchResult(response="Completion B", error=None),
                BatchResult(response="Completion C", error=None),
            ]

            completions = generate_completions(
                client=mock_client, tasks=sample_tasks,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            # Step 2: Store completions
            store = CompletionStore(client=mock_client, seed=42)
            store.save(completions, config={"model": "test-model", "seed": 42})

            # Step 3: Load and build completion lookup
            loaded = store.load(task_lookup)
            completion_lookup = {tc.task.id: tc.completion for tc in loaded}

            # Step 4: Generate pairs and measure revealed preferences
            pairs = [
                (sample_tasks[0], sample_tasks[1]),
                (sample_tasks[0], sample_tasks[2]),
                (sample_tasks[1], sample_tasks[2]),
            ]

            mock_client.generate_batch_async = AsyncMock(side_effect=[
                [BatchResult(response="Task A was better", error=None)],
                [BatchResult(response="I prefer Task B", error=None)],
                [BatchResult(response="Task A", error=None)],
            ])

            data = [
                (a, b, completion_lookup[a.id], completion_lookup[b.id])
                for a, b in pairs
            ]

            builder = PostTaskRevealedPromptBuilder(
                measurer=RevealedPreferenceMeasurer(),
                response_format=CHOICE_FORMATS["regex"]("Task A", "Task B"),
                template=post_task_revealed_template_fixture,
            )

            batch = measure_post_task_revealed(
                client=mock_client, data=data, builder=builder,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            assert len(batch.successes) == 3
            choices = [m.choice for m in batch.successes]
            assert choices == ["a", "b", "a"]

    def test_multiple_seeds_stored_separately(
        self, sample_tasks, task_lookup, temp_results_dir, mock_client
    ):
        """Different seeds should create separate storage directories."""
        with patch("src.measurement.storage.completions.COMPLETIONS_DIR", temp_results_dir):
            mock_client.generate_batch.return_value = [
                BatchResult(response="Seed 0 completion", error=None),
            ]

            # Store with seed 0
            store0 = CompletionStore(client=mock_client, seed=0)
            store0.save(
                [TaskCompletion(task=sample_tasks[0], completion="Seed 0")],
                config={"seed": 0}
            )

            # Store with seed 1
            store1 = CompletionStore(client=mock_client, seed=1)
            store1.save(
                [TaskCompletion(task=sample_tasks[0], completion="Seed 1")],
                config={"seed": 1}
            )

            # Verify separate storage
            assert store0.store_dir != store1.store_dir
            assert store0.load()[0].completion == "Seed 0"
            assert store1.load()[0].completion == "Seed 1"


# =============================================================================
# Response Format Tests
# =============================================================================


# =============================================================================
# Post-Task Active Learning Tests
# =============================================================================


class TestPostRevealedCacheActiveLearning:
    """Tests for PostRevealedCache methods used in active learning."""

    def test_get_measurements_returns_empty_when_no_file(self, temp_results_dir, sample_template):
        """get_measurements should return empty list when no cache exists."""
        import uuid
        from src.measurement.storage.post_task import PostRevealedCache

        with patch("src.measurement.storage.post_task.POST_REVEALED_DIR", temp_results_dir):
            # Use unique model name to avoid cache pollution from other tests
            cache = PostRevealedCache(
                model_name=f"test-model-{uuid.uuid4().hex[:8]}",
                template=sample_template,
                response_format="regex",
                order="canonical",
                completion_seed=0,
                rating_seed=42,
            )
            result = cache.get_measurements()
            assert result == []

    def test_get_measurements_filters_by_task_ids(self, sample_tasks, temp_results_dir, sample_template):
        """get_measurements should filter by task_ids when provided."""
        import uuid
        from src.measurement.storage.post_task import PostRevealedCache

        with patch("src.measurement.storage.post_task.POST_REVEALED_DIR", temp_results_dir):
            cache = PostRevealedCache(
                model_name=f"test-model-{uuid.uuid4().hex[:8]}",
                template=sample_template,
                response_format="regex",
                order="canonical",
                completion_seed=0,
                rating_seed=42,
            )

            # Create measurements
            measurements = [
                BinaryPreferenceMeasurement(
                    task_a=sample_tasks[0], task_b=sample_tasks[1],
                    choice="a", preference_type=PreferenceType.POST_TASK_REVEALED,
                ),
                BinaryPreferenceMeasurement(
                    task_a=sample_tasks[1], task_b=sample_tasks[2],
                    choice="b", preference_type=PreferenceType.POST_TASK_REVEALED,
                ),
            ]
            cache.append(measurements)

            # Get all
            all_data = cache.get_measurements()
            assert len(all_data) == 2

            # Filter to subset
            filtered = cache.get_measurements(task_ids={"math_1", "creative_1"})
            assert len(filtered) == 1
            assert filtered[0]["task_a"] == "math_1"

    @pytest.mark.asyncio
    async def test_get_or_measure_async_uses_cache(self, sample_tasks, task_lookup, temp_results_dir, sample_template):
        """get_or_measure_async should return cached measurements without calling API."""
        import uuid
        from src.measurement.storage.post_task import PostRevealedCache

        with patch("src.measurement.storage.post_task.POST_REVEALED_DIR", temp_results_dir):
            cache = PostRevealedCache(
                model_name=f"test-model-{uuid.uuid4().hex[:8]}",
                template=sample_template,
                response_format="regex",
                order="canonical",
                completion_seed=0,
                rating_seed=42,
            )

            # Pre-populate cache
            measurements = [
                BinaryPreferenceMeasurement(
                    task_a=sample_tasks[0], task_b=sample_tasks[1],
                    choice="a", preference_type=PreferenceType.POST_TASK_REVEALED,
                ),
            ]
            cache.append(measurements)

            # Setup
            completion_lookup = {
                "math_1": "Completion A",
                "creative_1": "Completion B",
            }
            pairs = [(sample_tasks[0], sample_tasks[1])]

            # Mock measure_fn that should NOT be called
            measure_fn = AsyncMock()

            result, stats = await cache.get_or_measure_async(
                pairs, completion_lookup, measure_fn, task_lookup
            )

            assert stats.cache_hits == 1
            assert stats.api_successes == 0
            measure_fn.assert_not_called()
            assert len(result) == 1
            assert result[0].choice == "a"

    @pytest.mark.asyncio
    async def test_get_or_measure_async_calls_api_for_misses(
        self, sample_tasks, task_lookup, temp_results_dir, sample_template
    ):
        """get_or_measure_async should call measure_fn for uncached pairs."""
        import uuid
        from src.measurement.storage.post_task import PostRevealedCache
        from src.types import MeasurementBatch

        with patch("src.measurement.storage.post_task.POST_REVEALED_DIR", temp_results_dir):
            cache = PostRevealedCache(
                model_name=f"test-model-{uuid.uuid4().hex[:8]}",
                template=sample_template,
                response_format="regex",
                order="canonical",
                completion_seed=0,
                rating_seed=42,
            )

            completion_lookup = {
                "math_1": "Completion A",
                "creative_1": "Completion B",
            }
            pairs = [(sample_tasks[0], sample_tasks[1])]

            # Mock measure_fn returns a measurement
            mock_measurement = BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="b", preference_type=PreferenceType.POST_TASK_REVEALED,
            )
            measure_fn = AsyncMock(return_value=MeasurementBatch(
                successes=[mock_measurement], failures=[]
            ))

            result, stats = await cache.get_or_measure_async(
                pairs, completion_lookup, measure_fn, task_lookup
            )

            assert stats.cache_hits == 0
            assert stats.api_successes == 1
            measure_fn.assert_called_once()
            assert len(result) == 1
            assert result[0].choice == "b"

            # Verify it was cached
            cached = cache.get_measurements()
            assert len(cached) == 1


class TestPostTaskActiveLearningE2E:
    """End-to-end test of post-task active learning workflow."""

    @pytest.mark.asyncio
    async def test_active_learning_loop_with_caching(
        self, sample_tasks, task_lookup, temp_results_dir, sample_template
    ):
        """Test the full active learning loop with caching behavior."""
        from src.measurement.storage.post_task import PostRevealedCache
        from src.measurement.storage.cache import reconstruct_measurements
        from src.fitting.thurstonian_fitting.active_learning import (
            ActiveLearningState,
            generate_d_regular_pairs,
        )
        from src.types import MeasurementBatch

        import uuid
        unique_model = f"test-model-e2e-{uuid.uuid4().hex[:8]}"

        with patch("src.measurement.storage.post_task.POST_REVEALED_DIR", temp_results_dir):
            cache = PostRevealedCache(
                model_name=unique_model,
                template=sample_template,
                response_format="regex",
                order="canonical",
                completion_seed=0,
                rating_seed=42,
            )

            completion_lookup = {t.id: f"Completion for {t.id}" for t in sample_tasks}
            state = ActiveLearningState(tasks=sample_tasks)

            # Generate initial pairs
            rng = __import__("numpy").random.default_rng(42)
            pairs = generate_d_regular_pairs(sample_tasks, d=2, rng=rng)

            # Simulate measure_fn that returns deterministic choices
            call_count = [0]
            async def mock_measure_fn(data):
                results = []
                for task_a, task_b, comp_a, comp_b in data:
                    call_count[0] += 1
                    choice = "a" if task_a.id < task_b.id else "b"
                    results.append(BinaryPreferenceMeasurement(
                        task_a=task_a, task_b=task_b,
                        choice=choice, preference_type=PreferenceType.POST_TASK_REVEALED,
                    ))
                return MeasurementBatch(successes=results, failures=[])

            # First iteration - all should be API calls
            result, stats = await cache.get_or_measure_async(
                pairs, completion_lookup, mock_measure_fn, task_lookup
            )

            assert stats.cache_hits == 0
            assert stats.api_successes == len(pairs)
            assert len(result) == len(pairs)

            state.add_comparisons(result)

            # Second iteration - same pairs should all be cache hits
            call_count[0] = 0
            result2, stats2 = await cache.get_or_measure_async(
                pairs, completion_lookup, mock_measure_fn, task_lookup
            )

            assert stats2.cache_hits == len(pairs)
            assert stats2.api_successes == 0
            assert call_count[0] == 0  # measure_fn not called
            assert len(result2) == len(pairs)

            # Verify state tracks measurements correctly
            assert len(state.comparisons) == len(pairs)
            assert len(state.sampled_pairs) == len(pairs)


class TestPostTaskResponseFormats:
    """Test different response formats work with post-task measurements."""

    @pytest.mark.parametrize("format_name", ["regex", "xml"])
    def test_stated_rating_formats(self, sample_tasks, mock_client, format_name):
        """Should parse ratings from different formats."""
        responses = {
            "regex": "I rate this 7 out of 10",
            "xml": "<rating>7</rating>",
        }

        mock_client.generate_batch_async = AsyncMock(return_value=[
            BatchResult(response=responses[format_name], error=None),
        ])

        # Use a simple template without scale placeholders
        simple_template = PromptTemplate(
            name="test_template",
            template="Rate that task.\n{format_instruction}",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["post_task_stated"],
        )

        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS[format_name](scale_min=1, scale_max=10),
            template=simple_template,
        )

        batch = measure_post_task_stated(
            client=mock_client,
            data=[(sample_tasks[0], "Test completion")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        assert batch.successes[0].score == 7.0

    @pytest.mark.parametrize("format_name", ["regex", "xml"])
    def test_revealed_choice_formats(self, sample_tasks, mock_client, format_name, post_task_revealed_template_fixture):
        """Should parse choices from different formats."""
        responses = {
            "regex": "I prefer Task A",
            "xml": "<choice>Task A</choice>",
        }

        mock_client.generate_batch_async = AsyncMock(return_value=[
            BatchResult(response=responses[format_name], error=None),
        ])

        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS[format_name]("Task A", "Task B"),
            template=post_task_revealed_template_fixture,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=[(sample_tasks[0], sample_tasks[1], "Comp A", "Comp B")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        assert batch.successes[0].choice == "a"


class TestReasoningMode:
    """Test reasoning mode for revealed preference measurements."""

    def test_reasoning_mode_extracts_choice_and_stores_raw_response(
        self, sample_tasks, mock_client, post_task_revealed_template_fixture
    ):
        """Reasoning mode should extract choice from XML and store full response."""
        from src.measurement.elicitation.response_format import get_revealed_response_format

        # Model explains reasoning first, then gives choice in XML tags
        verbose_response = (
            "Writing poetry allows for creative expression while the math task felt mechanical. "
            "<choice>Task A</choice>"
        )

        mock_client.generate_batch_async = AsyncMock(return_value=[
            BatchResult(response=verbose_response, error=None),
        ])

        response_format = get_revealed_response_format("Task A", "Task B", "xml", reasoning_mode=True)
        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=response_format,
            template=post_task_revealed_template_fixture,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=[(sample_tasks[0], sample_tasks[1], "Comp A", "Comp B")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        measurement = batch.successes[0]
        assert measurement.choice == "a"
        assert measurement.raw_response == verbose_response
        assert "creative expression" in measurement.raw_response

    def test_reasoning_mode_handles_both_tasks_mentioned_in_reasoning(
        self, sample_tasks, mock_client, post_task_revealed_template_fixture
    ):
        """Should extract correct choice even when reasoning mentions both tasks."""
        from src.measurement.elicitation.response_format import get_revealed_response_format

        # Reasoning mentions both tasks, then choice is Task B at the end
        verbose_response = (
            "While Task A was interesting, Task B challenged my analytical skills more. "
            "<choice>Task B</choice>"
        )

        mock_client.generate_batch_async = AsyncMock(return_value=[
            BatchResult(response=verbose_response, error=None),
        ])

        response_format = get_revealed_response_format("Task A", "Task B", "xml", reasoning_mode=True)
        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=response_format,
            template=post_task_revealed_template_fixture,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=[(sample_tasks[0], sample_tasks[1], "Comp A", "Comp B")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        # Should correctly extract Task B despite Task A being mentioned multiple times
        assert batch.successes[0].choice == "b"
        assert batch.successes[0].raw_response is not None

    def test_reasoning_mode_only_allowed_with_xml_format(self):
        """Should raise error when reasoning_mode used with non-xml format."""
        from src.measurement.elicitation.response_format import get_revealed_response_format

        with pytest.raises(ValueError, match="reasoning_mode requires xml format"):
            get_revealed_response_format("Task A", "Task B", "regex", reasoning_mode=True)

        with pytest.raises(ValueError, match="reasoning_mode requires xml format"):
            get_revealed_response_format("Task A", "Task B", "tool_use", reasoning_mode=True)

    def test_non_reasoning_mode_does_not_store_raw_response(
        self, sample_tasks, mock_client, post_task_revealed_template_fixture
    ):
        """Without reasoning mode, raw_response should be None."""
        mock_client.generate_batch_async = AsyncMock(return_value=[
            BatchResult(response="<choice>Task A</choice>", error=None),
        ])

        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS["xml"]("Task A", "Task B"),
            template=post_task_revealed_template_fixture,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=[(sample_tasks[0], sample_tasks[1], "Comp A", "Comp B")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        assert batch.successes[0].choice == "a"
        assert batch.successes[0].raw_response is None

    def test_reasoning_mode_format_instruction(self):
        """Format instruction should ask for reasoning then choice."""
        from src.measurement.elicitation.response_format import get_revealed_response_format

        fmt = get_revealed_response_format("Task A", "Task B", "xml", reasoning_mode=True)
        instruction = fmt.format_instruction()

        assert "<choice>" in instruction
        assert "reasoning" in instruction.lower()
        # Reasoning instruction should come before choice instruction
        reasoning_pos = instruction.lower().find("reasoning")
        choice_pos = instruction.find("<choice>")
        assert reasoning_pos < choice_pos
