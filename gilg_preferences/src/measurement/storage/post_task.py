"""Storage for post-task measurements using unified caches."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

from src.measurement.storage.unified_cache import StatedCache, RevealedCache, template_config_from_template
from src.measurement.storage.cache import MeasurementStats, reconstruct_measurements
from src.types import PreferenceType

if TYPE_CHECKING:
    from src.measurement.elicitation.prompt_templates.template import PromptTemplate
    from src.task_data import Task
    from src.types import BinaryPreferenceMeasurement, MeasurementBatch, TaskScore


POST_STATED_DIR = Path("results/post_task_stated")
POST_REVEALED_DIR = Path("results/post_task_revealed")


class PostStatedCache:
    """Cache for post-task stated measurements using unified StatedCache."""

    def __init__(
        self,
        model_name: str,
        template: "PromptTemplate",
        response_format: str,
        completion_seed: int,
        rating_seed: int,
        system_prompt: str | None = None,
        completion_model: str | None = None,
    ):
        self._cache = StatedCache(model_name)
        self._template_config = template_config_from_template(template)
        self._response_format = response_format
        self._completion_seed = completion_seed
        self._rating_seed = rating_seed
        self._system_prompt = system_prompt
        self._completion_model = completion_model

    def get_existing_task_ids(self) -> set[str]:
        """Get task IDs that have been measured."""
        return self._cache.get_task_ids(
            template_config=self._template_config,
            response_format=self._response_format,
            rating_seed=self._rating_seed,
            completion_seed=self._completion_seed,
            system_prompt=self._system_prompt,
            completion_model=self._completion_model,
        )

    def save(self, scores: list["TaskScore"]) -> None:
        """Save scores to cache."""
        for s in scores:
            self._cache.add(
                template_config=self._template_config,
                response_format=self._response_format,
                rating_seed=self._rating_seed,
                task_id=s.task.id,
                sample={"score": s.score},
                completion_seed=self._completion_seed,
                system_prompt=self._system_prompt,
                completion_model=self._completion_model,
            )
        self._cache.save()

    def _partition_tasks(
        self,
        tasks: list["Task"],
    ) -> tuple[list["TaskScore"], list["Task"]]:
        """Split tasks into cached hits and tasks needing API calls."""
        from src.types import TaskScore

        existing_ids = self.get_existing_task_ids()

        # Build lookup of cached scores by task_id
        cached_scores: dict[str, list[float]] = {}
        for task_id in existing_ids:
            samples = self._cache.get(
                template_config=self._template_config,
                response_format=self._response_format,
                rating_seed=self._rating_seed,
                task_id=task_id,
                completion_seed=self._completion_seed,
                system_prompt=self._system_prompt,
                completion_model=self._completion_model,
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
                    preference_type=PreferenceType.POST_TASK_STATED,
                ))
            else:
                to_query.append(task)

        return cached_hits, to_query

    async def get_or_measure_async(
        self,
        tasks: list["Task"],
        completion_lookup: dict[str, str],
        measure_fn: Callable[[list[tuple["Task", str]]], Awaitable["MeasurementBatch"]],
    ) -> tuple[list["TaskScore"], MeasurementStats]:
        """Check cache for each task, call measure_fn for misses, return combined."""
        from src.types import TaskScore

        if not tasks:
            return [], MeasurementStats()

        cached_hits, to_query = self._partition_tasks(tasks)
        stats = MeasurementStats(cache_hits=len(cached_hits))

        if to_query:
            data = [(task, completion_lookup[task.id]) for task in to_query]
            fresh_batch = await measure_fn(data)
            stats.api_successes = len(fresh_batch.successes)
            stats.api_failures = len(fresh_batch.failures)
            stats.failures = fresh_batch.failures
            self.save(fresh_batch.successes)
            return cached_hits + fresh_batch.successes, stats

        return cached_hits, stats


class PostRevealedCache:
    """Cache for post-task revealed measurements using unified RevealedCache."""

    def __init__(
        self,
        model_name: str,
        template: PromptTemplate,
        response_format: str,
        order: str,
        completion_seed: int,
        rating_seed: int,
    ):
        self._cache = RevealedCache(model_name)
        self._template_config = template_config_from_template(template)
        self._response_format = response_format
        self._order = order
        self._completion_seed = completion_seed
        self._rating_seed = rating_seed

    def get_existing_pairs(self) -> set[tuple[str, str]]:
        return self._cache.get_pairs(
            template_config=self._template_config,
            response_format=self._response_format,
            order=self._order,
            rating_seed=self._rating_seed,
            completion_seed=self._completion_seed,
        )

    def append(self, measurements: list[BinaryPreferenceMeasurement]) -> None:
        if not measurements:
            return

        for m in measurements:
            self._cache.add(
                template_config=self._template_config,
                response_format=self._response_format,
                order=self._order,
                rating_seed=self._rating_seed,
                task_a_id=m.task_a.id,
                task_b_id=m.task_b.id,
                sample={"choice": m.choice},
                completion_seed=self._completion_seed,
            )
        self._cache.save()

    def get_measurements(
        self,
        task_ids: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Load measurements, optionally filtered to pairs where both tasks in task_ids."""
        return self._cache.get_measurements(
            template_config=self._template_config,
            response_format=self._response_format,
            order=self._order,
            rating_seed=self._rating_seed,
            task_ids=task_ids,
            completion_seed=self._completion_seed,
        )

    def _partition_pairs(
        self,
        pairs: list[tuple[Task, Task]],
        task_lookup: dict[str, Task],
    ) -> tuple[list[BinaryPreferenceMeasurement], list[tuple[Task, Task]]]:
        """Split pairs into cached hits and pairs needing API calls."""
        cached_raw = self.get_measurements()
        cached_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
        for m in cached_raw:
            cached_by_pair.setdefault((m["task_a"], m["task_b"]), []).append(m)

        cached_hits_raw: list[dict[str, str]] = []
        to_query: list[tuple[Task, Task]] = []

        for a, b in pairs:
            key = (a.id, b.id)
            if key in cached_by_pair and cached_by_pair[key]:
                cached_hits_raw.append(cached_by_pair[key].pop())
            else:
                to_query.append((a, b))

        return reconstruct_measurements(cached_hits_raw, task_lookup), to_query

    async def get_or_measure_async(
        self,
        pairs: list[tuple[Task, Task]],
        completion_lookup: dict[str, str],
        measure_fn: Callable[[list[tuple[Task, Task, str, str]]], Awaitable[MeasurementBatch]],
        task_lookup: dict[str, Task],
    ) -> tuple[list[BinaryPreferenceMeasurement], MeasurementStats]:
        """Check cache for each pair, call measure_fn for misses, return combined."""
        if not pairs:
            return [], MeasurementStats()

        cached_hits, to_query = self._partition_pairs(pairs, task_lookup)
        stats = MeasurementStats(cache_hits=len(cached_hits))

        if to_query:
            data = [
                (a, b, completion_lookup[a.id], completion_lookup[b.id])
                for a, b in to_query
            ]
            fresh_batch = await measure_fn(data)
            stats.api_successes = len(fresh_batch.successes)
            stats.api_failures = len(fresh_batch.failures)
            stats.failures = fresh_batch.failures
            self.append(fresh_batch.successes)
            return cached_hits + fresh_batch.successes, stats

        return cached_hits, stats
