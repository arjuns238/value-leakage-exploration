from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar, Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.models import GenerateRequest, OpenAICompatibleClient
from src.models.openai_compatible import REQUEST_TIMEOUT
from src.models.registry import adjust_timeout_for_reasoning
from src.task_data import Task
from src.types import (
    BinaryPreferenceMeasurement,
    FailureCategory,
    MeasurementBatch,
    MeasurementFailure,
    PreferencePrompt,
    RankingMeasurement,
    TaskScore,
)

if TYPE_CHECKING:
    from src.measurement.elicitation.prompt_templates.builders import PostTaskRevealedPromptBuilder, PromptBuilder, PreTaskRankingPromptBuilder

T = TypeVar("T", TaskScore, BinaryPreferenceMeasurement)


def _build_request(prompt: PreferencePrompt, temperature: float, seed: int | None, timeout: float | None = None) -> GenerateRequest:
    return GenerateRequest(
        messages=prompt.messages,
        temperature=temperature,
        tools=prompt.response_format.tools,
        seed=seed,
        timeout=timeout,
        task_prompts=[t.prompt for t in prompt.tasks],
    )


def _categorize_error(error_msg: str, has_response: bool) -> FailureCategory:
    """Categorize an error message into a failure category."""
    error_lower = error_msg.lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        return FailureCategory.TIMEOUT
    if "rate" in error_lower and "limit" in error_lower:
        return FailureCategory.RATE_LIMIT
    if "expected tool call" in error_lower or "tool" in error_lower:
        return FailureCategory.TOOL_USE_FAILURE
    if "content" in error_lower and "filter" in error_lower:
        return FailureCategory.CONTENT_FILTER
    if "request failed" in error_lower:
        return FailureCategory.API_ERROR
    if has_response:
        return FailureCategory.PARSE_ERROR
    return FailureCategory.OTHER


def _make_failure(
    prompt: PreferencePrompt,
    error_message: str,
    raw_response: str | None = None,
    category: FailureCategory | None = None,
) -> MeasurementFailure:
    """Create a structured failure from a prompt and error."""
    if category is None:
        category = _categorize_error(error_message, raw_response is not None)
    return MeasurementFailure(
        task_ids=[t.id for t in prompt.tasks],
        category=category,
        raw_response=raw_response,
        error_message=error_message,
    )


async def _generate_and_parse_one(
    client: OpenAICompatibleClient,
    prompt: PreferencePrompt,
    temperature: float,
    seed: int | None,
    semaphore: asyncio.Semaphore,
    result_type: type[T],
    timeout: float | None = None,
    async_client: "AsyncOpenAI | None" = None,
) -> tuple[T | None, MeasurementFailure | None]:
    """Generate a response and parse it immediately. Returns (success, failure)."""
    request = _build_request(prompt, temperature, seed, timeout)

    # Generate
    results = await client.generate_batch_async([request], semaphore, async_client=async_client)
    response = results[0]

    if not response.ok:
        return None, _make_failure(prompt, f"Request failed: {response.error_details()}")

    response_text = response.unwrap()

    # Parse the response (refusal detection is handled by the parser itself)
    try:
        parsed = await prompt.measurer.parse(response_text, prompt)
        if isinstance(parsed.result, result_type):
            return parsed.result, None
        return None, _make_failure(
            prompt,
            f"Unexpected result type: {type(parsed.result)}",
            raw_response=response_text,
        )
    except Exception as e:
        return None, _make_failure(prompt, str(e), raw_response=response_text)


async def _measure_async(
    client: OpenAICompatibleClient,
    prompts: list[PreferencePrompt],
    semaphore: asyncio.Semaphore,
    temperature: float,
    seed: int | None,
    result_type: type[T],
    on_complete: Callable[[], None] | None = None,
    async_client: "AsyncOpenAI | None" = None,
) -> MeasurementBatch[T]:
    """Generate and parse concurrently - parsing starts as soon as each response arrives."""
    timeout = adjust_timeout_for_reasoning(client.canonical_model_name, REQUEST_TIMEOUT)

    async def process_with_callback(prompt: PreferencePrompt) -> tuple[T | None, MeasurementFailure | None]:
        result = await _generate_and_parse_one(client, prompt, temperature, seed, semaphore, result_type, timeout, async_client=async_client)
        if on_complete:
            on_complete()
        return result

    results = await asyncio.gather(*[process_with_callback(p) for p in prompts])

    successes: list[T] = []
    failures: list[MeasurementFailure] = []
    for success, failure in results:
        if success is not None:
            successes.append(success)
        if failure is not None:
            failures.append(failure)

    return MeasurementBatch(successes=successes, failures=failures)


def _measure_sync(
    client: OpenAICompatibleClient,
    prompts: list[PreferencePrompt],
    max_concurrent: int,
    temperature: float,
    seed: int | None,
    result_type: type[T],
) -> MeasurementBatch[T]:
    semaphore = asyncio.Semaphore(max_concurrent)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Requests", total=len(prompts))
        return asyncio.run(_measure_async(
            client, prompts, semaphore, temperature, seed, result_type,
            on_complete=lambda: progress.update(task, advance=1),
        ))


# Public API - Async versions

async def measure_pre_task_revealed_async(
    client: OpenAICompatibleClient,
    pairs: list[tuple[Task, Task]],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
    async_client: "AsyncOpenAI | None" = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b) for a, b in pairs]
    return await _measure_async(client, prompts, semaphore, temperature, seed, BinaryPreferenceMeasurement, on_complete, async_client=async_client)


async def measure_pre_task_stated_async(
    client: OpenAICompatibleClient,
    tasks: list[Task],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task) for task in tasks]
    return await _measure_async(client, prompts, semaphore, temperature, seed, TaskScore, on_complete)


async def measure_post_task_stated_async(
    client: OpenAICompatibleClient,
    data: list[tuple[Task, str]],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task, completion) for task, completion in data]
    return await _measure_async(client, prompts, semaphore, temperature, seed, TaskScore, on_complete)


async def measure_post_task_revealed_async(
    client: OpenAICompatibleClient,
    data: list[tuple[Task, Task, str, str]],
    builder: PostTaskRevealedPromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b, ca, cb) for a, b, ca, cb in data]
    return await _measure_async(client, prompts, semaphore, temperature, seed, BinaryPreferenceMeasurement, on_complete)


# Public API - Sync versions (with progress bar)

def measure_pre_task_revealed(
    client: OpenAICompatibleClient,
    pairs: list[tuple[Task, Task]],
    builder: PromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b) for a, b in pairs]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, BinaryPreferenceMeasurement)


def measure_pre_task_stated(
    client: OpenAICompatibleClient,
    tasks: list[Task],
    builder: PromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task) for task in tasks]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, TaskScore)


def measure_post_task_stated(
    client: OpenAICompatibleClient,
    data: list[tuple[Task, str]],
    builder: PromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task, completion) for task, completion in data]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, TaskScore)


def measure_post_task_revealed(
    client: OpenAICompatibleClient,
    data: list[tuple[Task, Task, str, str]],
    builder: PostTaskRevealedPromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b, ca, cb) for a, b, ca, cb in data]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, BinaryPreferenceMeasurement)


# Public API - Ranking measurements

async def measure_pre_task_ranking_async(
    client: OpenAICompatibleClient,
    task_groups: list[list[Task]],
    builder: "PreTaskRankingPromptBuilder",
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[RankingMeasurement]:
    prompts = [builder.build(tasks) for tasks in task_groups]
    return await _measure_async(client, prompts, semaphore, temperature, seed, RankingMeasurement, on_complete)


# Public API - Aggregation

def aggregate_choice_rates(
    batch: MeasurementBatch[BinaryPreferenceMeasurement],
) -> dict[str, float | int]:
    """Aggregate a measurement batch into choice rates.

    Returns {"rate": float, "n_parsed": int, "n_failed": int, "n_refusal": int}.
    rate = (n choices that are "a" or "b") / n_parsed, or 0.0 if n_parsed == 0.
    """
    n_refusal = sum(1 for m in batch.successes if m.choice == "refusal")
    n_chose = len(batch.successes) - n_refusal
    n_parsed = len(batch.successes)
    n_failed = len(batch.failures)
    rate = n_chose / n_parsed if n_parsed > 0 else 0.0
    return {"rate": rate, "n_parsed": n_parsed, "n_failed": n_failed, "n_refusal": n_refusal}
