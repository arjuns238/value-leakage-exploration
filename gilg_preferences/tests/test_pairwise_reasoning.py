"""Tests that pairwise-measurement clients default `enable_reasoning` from the
registry's `reasoning_mode` field."""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation import (
    CompletionChoiceFormat,
    RevealedPreferenceMeasurer,
    measure_pre_task_revealed,
)
from src.measurement.elicitation.prompt_templates import (
    PreTaskRevealedPromptBuilder,
    PromptTemplate,
    TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.models import BatchResult, OpenRouterClient, get_client
from src.models.registry import should_capture_reasoning
from src.task_data import OriginDataset, Task


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_builder() -> PreTaskRevealedPromptBuilder:
    template = PromptTemplate(
        template=(
            "You will be given two tasks. Choose one and complete it.\n"
            "{format_instruction}\n\n"
            "Task A:\n{task_a}\n\n"
            "Task B:\n{task_b}"
        ),
        name="pre_task_revealed_completion_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
    )
    return PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=template,
    )


def _math_pair() -> tuple[Task, Task]:
    return (
        Task(
            prompt=(
                "Compute the sum 1/(1*2) + 1/(2*3) + ... + 1/(9*10). "
                "Express the answer as a common fraction."
            ),
            origin=OriginDataset.MATH,
            id="t_math",
            metadata={},
        ),
        Task(
            prompt="Write a short haiku about autumn leaves.",
            origin=OriginDataset.WILDCHAT,
            id="t_haiku",
            metadata={},
        ),
    )


# ---------------------------------------------------------------------------
# Unit test: clients resolve enable_reasoning from the model registry by default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,expected_enabled",
    [
        ("qwen3.5-122b", True),
        ("qwen3.5-122b-nothink", False),
        ("qwen3-32b", True),
        ("qwen3-32b-nothink", False),
        ("gemma-3-27b", False),
    ],
)
def test_default_reasoning_resolves_from_registry(model_name: str, expected_enabled: bool):
    client = OpenRouterClient(model_name=model_name)
    resolved = should_capture_reasoning(client.canonical_model_name)
    assert resolved is expected_enabled
    assert client._get_extra_body(enable_reasoning=resolved)["reasoning"]["enabled"] is expected_enabled


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.mark.api
def test_pairwise_measurement_returns_reasoning_for_reasoning_model():
    """A live pairwise measurement on a reasoning-capable model must surface
    non-empty reasoning content from the API. Wraps the client's
    `generate_batch_async` to peek at the BatchResult before measure.py drops it."""
    client = get_client(model_name="qwen3-32b", max_new_tokens=2048, reasoning_effort="low")
    builder = _make_builder()
    task_a, task_b = _math_pair()

    captured: list[BatchResult] = []
    original = client.generate_batch_async

    async def spy(requests, semaphore, *, enable_reasoning=False, async_client=None):
        results = await original(
            requests, semaphore, enable_reasoning=enable_reasoning, async_client=async_client
        )
        captured.extend(results)
        return results

    client.generate_batch_async = spy  # type: ignore[assignment]

    batch = measure_pre_task_revealed(client, [(task_a, task_b)], builder, max_concurrent=1)

    assert len(captured) == 1, "Expected exactly one API call"
    assert captured[0].ok, f"API call failed: {captured[0].error_details()}"
    reasoning = captured[0].reasoning
    assert reasoning is not None and len(reasoning.strip()) > 0, (
        f"Expected reasoning content from qwen3-32b with low effort. "
        f"Reasoning: {reasoning!r}"
    )
    # Sanity: measurement also succeeded
    assert len(batch.successes) + len(batch.failures) == 1


@pytest.mark.api
def test_pairwise_preferences_differ_between_thinking_and_nothink():
    """A model run with reasoning ON should disagree on at least one pair vs the
    same model run with reasoning OFF. Uses 6 math-vs-other pairs at temperature
    0 to maximize reproducibility while still allowing the regimes to diverge."""
    extra_pairs = [
        (
            Task(
                prompt="Find all integer solutions to x^2 + y^2 = 25.",
                origin=OriginDataset.MATH,
                id=f"m{i}",
                metadata={},
            ),
            Task(
                prompt=f"Write a friendly two-sentence email greeting (variant {i}).",
                origin=OriginDataset.WILDCHAT,
                id=f"w{i}",
                metadata={},
            ),
        )
        for i in range(5)
    ]
    pairs = [_math_pair(), *extra_pairs]
    builder = _make_builder()

    thinking_client = get_client(
        model_name="qwen3-32b", max_new_tokens=2048, reasoning_effort="low"
    )
    nothink_client = get_client(model_name="qwen3-32b-nothink", max_new_tokens=512)

    thinking_batch = measure_pre_task_revealed(
        thinking_client, pairs, builder, max_concurrent=3, seed=0
    )
    nothink_batch = measure_pre_task_revealed(
        nothink_client, pairs, builder, max_concurrent=3, seed=0
    )

    # Index by (task_a.id, task_b.id) so we compare apples to apples.
    def by_pair(batch):
        return {(m.task_a.id, m.task_b.id): m.choice for m in batch.successes}

    thinking_choices = by_pair(thinking_batch)
    nothink_choices = by_pair(nothink_batch)

    overlap = set(thinking_choices) & set(nothink_choices)
    assert len(overlap) >= 4, f"Too few overlapping successful pairs: {len(overlap)}"

    diffs = [k for k in overlap if thinking_choices[k] != nothink_choices[k]]
    assert len(diffs) >= 1, (
        "Expected at least one pair where thinking and no-think disagree. "
        f"thinking={thinking_choices}, nothink={nothink_choices}"
    )
    print(
        f"\nDisagreements: {len(diffs)}/{len(overlap)}.\n"
        f"thinking={thinking_choices}\nnothink={nothink_choices}"
    )
