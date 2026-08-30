"""Tests for aggregate_choice_rates."""

import pytest

from src.measurement.elicitation.measure import aggregate_choice_rates
from src.task_data import Task, OriginDataset
from src.types import (
    BinaryPreferenceMeasurement,
    FailureCategory,
    MeasurementBatch,
    MeasurementFailure,
    PreferenceType,
)

_T = PreferenceType.PRE_TASK_REVEALED


def _task(id: str) -> Task:
    return Task(prompt=f"Do {id}", origin=OriginDataset.WILDCHAT, id=id, metadata={})


def _measurement(choice: str) -> BinaryPreferenceMeasurement:
    return BinaryPreferenceMeasurement(
        task_a=_task("a"), task_b=_task("b"), choice=choice, preference_type=_T
    )


def _failure() -> MeasurementFailure:
    return MeasurementFailure(
        task_ids=["a", "b"], category=FailureCategory.PARSE_ERROR,
        raw_response="gibberish", error_message="could not parse",
    )


class TestAggregateChoiceRates:
    def test_all_choices(self):
        batch = MeasurementBatch(
            successes=[_measurement("a"), _measurement("b"), _measurement("a")],
            failures=[],
        )
        result = aggregate_choice_rates(batch)
        assert result == {"rate": 1.0, "n_parsed": 3, "n_failed": 0, "n_refusal": 0}

    def test_with_refusals(self):
        batch = MeasurementBatch(
            successes=[_measurement("a"), _measurement("refusal"), _measurement("b")],
            failures=[],
        )
        result = aggregate_choice_rates(batch)
        assert result["n_parsed"] == 3
        assert result["n_refusal"] == 1
        assert result["rate"] == pytest.approx(2 / 3)

    def test_with_failures(self):
        batch = MeasurementBatch(
            successes=[_measurement("a")],
            failures=[_failure(), _failure()],
        )
        result = aggregate_choice_rates(batch)
        assert result == {"rate": 1.0, "n_parsed": 1, "n_failed": 2, "n_refusal": 0}

    def test_empty_batch(self):
        batch = MeasurementBatch(successes=[], failures=[])
        result = aggregate_choice_rates(batch)
        assert result == {"rate": 0.0, "n_parsed": 0, "n_failed": 0, "n_refusal": 0}

    def test_all_refusals(self):
        batch = MeasurementBatch(
            successes=[_measurement("refusal"), _measurement("refusal")],
            failures=[],
        )
        result = aggregate_choice_rates(batch)
        assert result == {"rate": 0.0, "n_parsed": 2, "n_failed": 0, "n_refusal": 2}

    def test_mixed(self):
        batch = MeasurementBatch(
            successes=[_measurement("a"), _measurement("refusal"), _measurement("b"), _measurement("a")],
            failures=[_failure()],
        )
        result = aggregate_choice_rates(batch)
        assert result["rate"] == pytest.approx(3 / 4)
        assert result["n_parsed"] == 4
        assert result["n_failed"] == 1
        assert result["n_refusal"] == 1
