"""Measurement functions for open-ended responses with semantic valence scoring.

Collects open-ended responses from models and scores them using semantic valence analysis.
"""

from __future__ import annotations

import asyncio
from typing import Callable, TYPE_CHECKING

from src.models import OpenAICompatibleClient
from src.task_data import Task
from src.types import (
    FailureCategory,
    MeasurementBatch,
    MeasurementFailure,
    OpenEndedResponse,
    PreferencePrompt,
)
from src.measurement.elicitation.measure import _build_request, _categorize_error

if TYPE_CHECKING:
    from src.measurement.elicitation.prompt_templates.builders import PromptBuilder


async def _generate_and_parse_open_ended(
    client: OpenAICompatibleClient,
    prompt: PreferencePrompt,
    temperature: float,
    seed: int | None,
    semaphore: asyncio.Semaphore,
) -> tuple[OpenEndedResponse | None, MeasurementFailure | None]:
    """Generate response and parse it with semantic valence scoring."""
    request = _build_request(prompt, temperature, seed)

    # Generate
    results = await client.generate_batch_async([request], semaphore)
    response = results[0]

    if not response.ok:
        return None, MeasurementFailure(
            task_ids=[t.id for t in prompt.tasks],
            category=FailureCategory.API_ERROR,
            raw_response=None,
            error_message=f"Request failed: {response.error_details()}",
        )

    response_text = response.unwrap()

    # Parse the response and score valence
    try:
        parsed = await prompt.measurer.parse(response_text, prompt)
        if isinstance(parsed.result, OpenEndedResponse):
            return parsed.result, None
        return None, MeasurementFailure(
            task_ids=[t.id for t in prompt.tasks],
            category=FailureCategory.PARSE_ERROR,
            raw_response=response_text,
            error_message=f"Unexpected result type: {type(parsed.result)}",
        )
    except Exception as e:
        return None, MeasurementFailure(
            task_ids=[t.id for t in prompt.tasks],
            category=_categorize_error(str(e), True),
            raw_response=response_text,
            error_message=str(e),
        )


async def _measure_open_ended_async(
    client: OpenAICompatibleClient,
    prompts: list[PreferencePrompt],
    semaphore: asyncio.Semaphore,
    temperature: float,
    seed: int | None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[OpenEndedResponse]:
    """Generate and parse open-ended responses concurrently."""

    async def process_with_callback(prompt: PreferencePrompt) -> tuple[OpenEndedResponse | None, MeasurementFailure | None]:
        result = await _generate_and_parse_open_ended(client, prompt, temperature, seed, semaphore)
        if on_complete:
            on_complete()
        return result

    results = await asyncio.gather(*[process_with_callback(p) for p in prompts])

    successes: list[OpenEndedResponse] = []
    failures: list[MeasurementFailure] = []
    for success, failure in results:
        if success is not None:
            successes.append(success)
        if failure is not None:
            failures.append(failure)

    return MeasurementBatch(successes=successes, failures=failures)


# Public API - Async

async def measure_open_ended_stated_async(
    client: OpenAICompatibleClient,
    data: list[tuple[Task, str]],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[OpenEndedResponse]:
    """Measure open-ended responses after task completion.

    Args:
        client: LLM client for generation
        data: List of (task, completion_text) pairs
        builder: OpenEndedPromptBuilder instance
        semaphore: Semaphore for rate limiting
        temperature: Generation temperature
        seed: Random seed for reproducibility
        on_complete: Callback after each measurement completes

    Returns:
        MeasurementBatch[OpenEndedResponse] with successes and failures
    """
    prompts = [builder.build(task, completion) for task, completion in data]
    return await _measure_open_ended_async(client, prompts, semaphore, temperature, seed, on_complete)
