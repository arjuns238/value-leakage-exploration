"""Storage for pre-task stated measurements using unified StatedCache."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

from src.models import OpenAICompatibleClient
from src.measurement.storage.unified_cache import StatedCache, template_config_from_template
from src.measurement.storage.cache import MeasurementStats
from src.measurement.elicitation.prompt_templates.template import PromptTemplate
from src.types import TaskScore, PreferenceType

if TYPE_CHECKING:
    from src.task_data import Task
    from src.types import MeasurementBatch


PRE_TASK_STATED_DIR = Path("results/pre_task_stated")


def save_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    scores: list[TaskScore],
    response_format: str,
    seed: int,
    config: dict | None = None,
) -> None:
    """Save stated preference scores to unified cache."""
    cache = StatedCache(client.canonical_model_name)
    template_config = template_config_from_template(template)

    for s in scores:
        cache.add(
            template_config=template_config,
            response_format=response_format,
            rating_seed=seed,
            task_id=s.task.id,
            sample={"score": s.score},
        )

    cache.save()


def load_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> list[dict]:
    """Load stated preference scores from unified cache.

    Returns list of {task_id, score} dicts.
    """
    cache = StatedCache(client.canonical_model_name)
    template_config = template_config_from_template(template)

    # Get all task IDs for this configuration
    task_ids = cache.get_task_ids(
        template_config=template_config,
        response_format=response_format,
        rating_seed=seed,
    )

    results = []
    for task_id in task_ids:
        samples = cache.get(
            template_config=template_config,
            response_format=response_format,
            rating_seed=seed,
            task_id=task_id,
        )
        for sample in samples:
            results.append({"task_id": task_id, "score": sample["score"]})

    return results


def stated_exist(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> bool:
    """Check if stated measurements exist for this configuration."""
    cache = StatedCache(client.canonical_model_name)
    template_config = template_config_from_template(template)

    task_ids = cache.get_task_ids(
        template_config=template_config,
        response_format=response_format,
        rating_seed=seed,
    )

    return len(task_ids) > 0


class PreTaskStatedCache:
    """Cache for pre-task stated measurements using unified StatedCache."""

    def __init__(
        self,
        template: PromptTemplate,
        client: OpenAICompatibleClient,
        response_format: str,
        rating_seed: int,
    ):
        self._cache = StatedCache(client.canonical_model_name)
        self._template_config = template_config_from_template(template)
        self._response_format = response_format
        self._rating_seed = rating_seed

    def get_existing_task_ids(self) -> set[str]:
        """Get task IDs that have been measured."""
        return self._cache.get_task_ids(
            template_config=self._template_config,
            response_format=self._response_format,
            rating_seed=self._rating_seed,
        )

    def save(self, scores: list[TaskScore]) -> None:
        """Save scores to cache."""
        for s in scores:
            self._cache.add(
                template_config=self._template_config,
                response_format=self._response_format,
                rating_seed=self._rating_seed,
                task_id=s.task.id,
                sample={"score": s.score},
            )
        self._cache.save()

    def _partition_tasks(
        self,
        tasks: list[Task],
    ) -> tuple[list[TaskScore], list[Task]]:
        """Split tasks into cached hits and tasks needing API calls."""
        existing_ids = self.get_existing_task_ids()

        # Build lookup of cached scores by task_id
        cached_scores: dict[str, list[float]] = {}
        for task_id in existing_ids:
            samples = self._cache.get(
                template_config=self._template_config,
                response_format=self._response_format,
                rating_seed=self._rating_seed,
                task_id=task_id,
            )
            cached_scores[task_id] = [s["score"] for s in samples]

        cached_hits: list[TaskScore] = []
        to_query: list[Task] = []

        for task in tasks:
            if task.id in cached_scores and cached_scores[task.id]:
                score = cached_scores[task.id].pop()
                cached_hits.append(TaskScore(
                    task=task,
                    score=score,
                    preference_type=PreferenceType.PRE_TASK_STATED,
                ))
            else:
                to_query.append(task)

        return cached_hits, to_query

    async def get_or_measure_async(
        self,
        tasks: list["Task"],
        measure_fn: Callable[[list["Task"]], Awaitable["MeasurementBatch"]],
    ) -> tuple[list[TaskScore], MeasurementStats]:
        """Check cache for each task, call measure_fn for misses, return combined."""
        if not tasks:
            return [], MeasurementStats()

        cached_hits, to_query = self._partition_tasks(tasks)
        stats = MeasurementStats(cache_hits=len(cached_hits))

        if to_query:
            fresh_batch = await measure_fn(to_query)
            stats.api_successes = len(fresh_batch.successes)
            stats.api_failures = len(fresh_batch.failures)
            stats.failures = fresh_batch.failures
            self.save(fresh_batch.successes)
            return cached_hits + fresh_batch.successes, stats

        return cached_hits, stats
