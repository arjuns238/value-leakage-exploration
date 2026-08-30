"""Tests for active learning false convergence detection and checkpointing."""

import pytest

from src.measurement.storage.cache import MeasurementStats
from src.types import FailureCategory, MeasurementFailure
from src.fitting.thurstonian_fitting.active_learning_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    checkpoint_exists,
)


pytestmark = pytest.mark.thurstonian


class TestApiSideFailureCount:

    def test_counts_api_side_failures(self):
        failures = [
            MeasurementFailure(task_ids=["a", "b"], category=FailureCategory.API_ERROR, raw_response=None, error_message="402"),
            MeasurementFailure(task_ids=["c", "d"], category=FailureCategory.TIMEOUT, raw_response=None, error_message="timeout"),
            MeasurementFailure(task_ids=["e", "f"], category=FailureCategory.RATE_LIMIT, raw_response=None, error_message="429"),
            MeasurementFailure(task_ids=["g", "h"], category=FailureCategory.CONTENT_FILTER, raw_response=None, error_message="filtered"),
        ]
        stats = MeasurementStats(api_failures=4, failures=failures)
        assert stats.api_side_failure_count == 4

    def test_excludes_non_api_failures(self):
        failures = [
            MeasurementFailure(task_ids=["a", "b"], category=FailureCategory.PARSE_ERROR, raw_response="bad", error_message="parse error"),
            MeasurementFailure(task_ids=["c", "d"], category=FailureCategory.REFUSAL_NO_PREFERENCES, raw_response="I don't", error_message="refusal"),
            MeasurementFailure(task_ids=["e", "f"], category=FailureCategory.OTHER, raw_response=None, error_message="other"),
        ]
        stats = MeasurementStats(api_failures=3, failures=failures)
        assert stats.api_side_failure_count == 0

    def test_mixed_failure_categories(self):
        failures = [
            MeasurementFailure(task_ids=["a"], category=FailureCategory.API_ERROR, raw_response=None, error_message="402"),
            MeasurementFailure(task_ids=["b"], category=FailureCategory.PARSE_ERROR, raw_response="bad", error_message="parse"),
            MeasurementFailure(task_ids=["c"], category=FailureCategory.TIMEOUT, raw_response=None, error_message="timeout"),
            MeasurementFailure(task_ids=["d"], category=FailureCategory.REFUSAL_CONTENT_POLICY, raw_response="no", error_message="refusal"),
        ]
        stats = MeasurementStats(api_failures=4, failures=failures)
        assert stats.api_side_failure_count == 2

    def test_empty_failures(self):
        stats = MeasurementStats()
        assert stats.api_side_failure_count == 0


class TestCheckpointRoundTrip:

    def test_save_and_load(self, tmp_path):
        comparisons = [
            {"task_a": "t1", "task_b": "t2", "choice": "a"},
            {"task_a": "t3", "task_b": "t4", "choice": "b"},
        ]
        rank_correlations = [0.85, 0.92, 0.97]

        save_checkpoint(
            tmp_path,
            iteration=3,
            comparisons_dicts=comparisons,
            rank_correlations=rank_correlations,
        )

        assert checkpoint_exists(tmp_path)

        loaded = load_checkpoint(tmp_path)
        assert loaded["iteration"] == 3
        assert loaded["comparisons"] == comparisons
        assert loaded["rank_correlations"] == rank_correlations

    def test_checkpoint_not_exists(self, tmp_path):
        assert not checkpoint_exists(tmp_path)

    def test_overwrite_checkpoint(self, tmp_path):
        save_checkpoint(tmp_path, iteration=1, comparisons_dicts=[], rank_correlations=[])
        save_checkpoint(tmp_path, iteration=5, comparisons_dicts=[{"task_a": "a", "task_b": "b", "choice": "a"}], rank_correlations=[0.99])

        loaded = load_checkpoint(tmp_path)
        assert loaded["iteration"] == 5
        assert len(loaded["comparisons"]) == 1
