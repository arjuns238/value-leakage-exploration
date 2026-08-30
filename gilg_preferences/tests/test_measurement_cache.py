from __future__ import annotations

from pathlib import Path
import uuid

import pytest

pytestmark = pytest.mark.cache

from src.measurement.storage.cache import MeasurementCache, reconstruct_measurements
from src.measurement.storage.unified_cache import RevealedCache
from src.measurement.elicitation.prompt_templates import PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


class MockClient:
    """Mock that matches inference client interface for testing."""

    def __init__(self, model_name: str = "llama-3.1-8b"):
        self.model_name = model_name
        self.canonical_model_name = model_name


@pytest.fixture
def sample_template() -> PromptTemplate:
    return PromptTemplate(
        template="Choose: {task_a} or {task_b}? {format_instruction}",
        name="binary_choice_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        tags=frozenset(["phrasing:1", "task_a_label:A", "task_b_label:B"]),
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(prompt="Write a poem", origin=OriginDataset.WILDCHAT, id="task_1", metadata={}),
        Task(prompt="Solve an equation", origin=OriginDataset.WILDCHAT, id="task_2", metadata={}),
        Task(prompt="Explain gravity", origin=OriginDataset.WILDCHAT, id="task_3", metadata={}),
    ]


@pytest.fixture
def sample_measurements(sample_tasks: list[Task]) -> list[BinaryPreferenceMeasurement]:
    return [
        BinaryPreferenceMeasurement(
            task_a=sample_tasks[0],
            task_b=sample_tasks[1],
            choice="a",
            preference_type=PreferenceType.PRE_TASK_STATED,
        ),
        BinaryPreferenceMeasurement(
            task_a=sample_tasks[1],
            task_b=sample_tasks[2],
            choice="b",
            preference_type=PreferenceType.PRE_TASK_STATED,
        ),
    ]


@pytest.fixture
def unique_model() -> MockClient:
    """Create a model with unique name to isolate cache between tests."""
    return MockClient(f"test-model-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def isolated_cache_dir(tmp_path: Path, monkeypatch) -> Path:
    """Set cache directory to temp path for test isolation."""
    cache_dir = tmp_path / "cache" / "revealed"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(RevealedCache, "CACHE_DIR", cache_dir)
    return cache_dir


class TestMeasurementCacheUnit:
    """Unit tests for MeasurementCache - tests individual methods."""

    def test_empty_cache_returns_empty_set(
        self, tmp_path: Path, sample_template, unique_model, isolated_cache_dir
    ):
        """Test that empty cache returns empty set for existing pairs."""
        cache = MeasurementCache(sample_template, unique_model)

        # Empty cache returns empty set
        existing = cache.get_existing_pairs()
        assert existing == set()

    def test_append_saves_to_unified_cache(
        self, tmp_path: Path, sample_template, sample_measurements, unique_model, isolated_cache_dir
    ):
        """Test append saves measurements to unified cache."""
        cache = MeasurementCache(sample_template, unique_model)

        cache.append(sample_measurements)

        # Verify measurements are saved
        measurements = cache.get_measurements()
        assert len(measurements) == 2
        assert measurements[0] == {"task_a": "task_1", "task_b": "task_2", "choice": "a"}
        assert measurements[1] == {"task_a": "task_2", "task_b": "task_3", "choice": "b"}

    def test_append_empty_list_does_nothing(
        self, tmp_path: Path, sample_template, unique_model, isolated_cache_dir
    ):
        """Appending empty list should not save anything."""
        cache = MeasurementCache(sample_template, unique_model)

        cache.append([])

        assert cache.get_existing_pairs() == set()
        assert cache.get_measurements() == []

    def test_get_measurements_filters_by_task_ids(
        self, tmp_path: Path, sample_template, sample_measurements, unique_model, isolated_cache_dir
    ):
        """Test filtering measurements by task IDs."""
        cache = MeasurementCache(sample_template, unique_model)
        cache.append(sample_measurements)

        # Get all measurements
        all_data = cache.get_measurements()
        assert len(all_data) == 2

        # Filter to only pairs involving task_1 and task_2
        filtered = cache.get_measurements(task_ids={"task_1", "task_2"})
        assert len(filtered) == 1
        assert filtered[0]["task_a"] == "task_1"
        assert filtered[0]["task_b"] == "task_2"

        # Filter with task set that excludes all pairs
        empty_result = cache.get_measurements(task_ids={"task_1"})
        assert len(empty_result) == 0


class TestReconstructMeasurements:
    """Tests for the reconstruct_measurements helper function."""

    def test_reconstructs_measurements_from_raw_dicts(self, sample_tasks):
        """Test that raw dicts are reconstructed into proper measurement objects."""
        tasks_dict = {t.id: t for t in sample_tasks}
        raw = [
            {"task_a": "task_1", "task_b": "task_2", "choice": "a"},
            {"task_a": "task_2", "task_b": "task_3", "choice": "b"},
        ]

        measurements = reconstruct_measurements(raw, tasks_dict)

        assert len(measurements) == 2
        assert measurements[0].task_a.id == "task_1"
        assert measurements[0].task_b.id == "task_2"
        assert measurements[0].choice == "a"
        assert measurements[0].preference_type == PreferenceType.PRE_TASK_STATED

        assert measurements[1].task_a.id == "task_2"
        assert measurements[1].choice == "b"

    def test_uses_provided_preference_type(self, sample_tasks):
        """Test that the provided preference_type is used."""
        tasks_dict = {t.id: t for t in sample_tasks}
        raw = [{"task_a": "task_1", "task_b": "task_2", "choice": "a"}]

        measurements = reconstruct_measurements(
            raw, tasks_dict, preference_type=PreferenceType.POST_TASK_STATED
        )

        assert measurements[0].preference_type == PreferenceType.POST_TASK_STATED


class TestMeasurementCacheIntegration:
    """Integration tests - test the full workflow as would be used in practice."""

    def test_incremental_append_accumulates_and_tracks_pairs(
        self, tmp_path: Path, sample_template, sample_tasks, unique_model, isolated_cache_dir
    ):
        """Test that multiple appends accumulate measurements and get_existing_pairs tracks them."""
        cache = MeasurementCache(sample_template, unique_model)

        # First batch
        batch1 = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(batch1)

        assert cache.get_existing_pairs() == {("task_1", "task_2")}
        assert len(cache.get_measurements()) == 1

        # Second batch - different pair
        batch2 = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[2],
                choice="b", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(batch2)

        # Both pairs now tracked
        assert cache.get_existing_pairs() == {("task_1", "task_2"), ("task_2", "task_3")}
        assert len(cache.get_measurements()) == 2

        # Third batch - reversed order of first pair (should be distinct)
        batch3 = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[0],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(batch3)

        # Now have 3 pairs, (1,2) and (2,1) are distinct
        existing = cache.get_existing_pairs()
        assert ("task_1", "task_2") in existing
        assert ("task_2", "task_1") in existing
        assert len(existing) == 3
        assert len(cache.get_measurements()) == 3

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_template, sample_tasks, unique_model, isolated_cache_dir
    ):
        """Test that measurements persist when creating a new cache instance."""
        # First session - append some measurements
        cache1 = MeasurementCache(sample_template, unique_model)
        measurements = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[2],
                choice="b", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache1.append(measurements)

        # Second session - new instance, same parameters
        cache2 = MeasurementCache(sample_template, unique_model)

        # Should see the same data
        assert cache2.get_existing_pairs() == {("task_1", "task_2"), ("task_1", "task_3")}
        data = cache2.get_measurements()
        assert len(data) == 2
        assert data[0]["choice"] == "a"
        assert data[1]["choice"] == "b"

    def test_roundtrip_with_reconstruct(
        self, tmp_path: Path, sample_template, sample_tasks, unique_model, isolated_cache_dir
    ):
        """Test full round-trip: append -> persist -> load -> reconstruct."""
        cache = MeasurementCache(sample_template, unique_model)

        original = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[2],
                choice="b", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(original)

        # Load raw and reconstruct
        raw = cache.get_measurements()
        tasks_dict = {t.id: t for t in sample_tasks}
        reconstructed = reconstruct_measurements(raw, tasks_dict)

        # Verify structural equivalence
        assert len(reconstructed) == 2
        assert reconstructed[0].task_a.prompt == sample_tasks[0].prompt
        assert reconstructed[0].task_b.prompt == sample_tasks[1].prompt
        assert reconstructed[0].choice == "a"
        assert reconstructed[1].task_a.prompt == sample_tasks[1].prompt
        assert reconstructed[1].choice == "b"

    def test_different_template_model_pairs_isolated(
        self, tmp_path: Path, sample_tasks, isolated_cache_dir
    ):
        """Test that different template/model combinations use separate storage."""
        template1 = PromptTemplate(
            template="Choose: {task_a} or {task_b}? {format_instruction}",
            name="template_one",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        template2 = PromptTemplate(
            template="Pick: {task_a} or {task_b}? {format_instruction}",
            name="template_two",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        model1 = MockClient(f"test-model-{uuid.uuid4().hex[:8]}")
        model2 = MockClient(f"test-model-{uuid.uuid4().hex[:8]}")

        cache1 = MeasurementCache(template1, model1)
        cache2 = MeasurementCache(template2, model1)
        cache3 = MeasurementCache(template1, model2)

        # Write to cache1 only
        cache1.append([
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ])

        # Others should be empty
        assert len(cache1.get_measurements()) == 1
        assert len(cache2.get_measurements()) == 0
        assert len(cache3.get_measurements()) == 0
