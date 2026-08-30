"""Async experiment runners for shared semaphore usage."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ProgressCallback = Callable[["RunnerStats"], None]
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.measurement.elicitation.prompt_templates import PostTaskStatedPromptBuilder, PostTaskRevealedPromptBuilder, PreTaskRevealedPromptBuilder, PreTaskStatedPromptBuilder, PreTaskRankingPromptBuilder, PostTaskRankingPromptBuilder
from src.measurement.elicitation.prompt_templates.generator_config import TASK_LABELS
from src.measurement.elicitation import (
    measure_post_task_stated_async,
    measure_post_task_revealed_async,
    measure_pre_task_stated_async,
    measure_pre_task_revealed_async,
    StatedScoreMeasurer,
    RevealedPreferenceMeasurer,
    get_stated_response_format,
    get_revealed_response_format,
)
from src.measurement.elicitation.measure import measure_pre_task_ranking_async, _measure_async
from src.measurement.elicitation.measurer import RankingMeasurer
from src.measurement.elicitation.response_format import get_ranking_response_format
from src.types import Message, RankingMeasurement
from src.measurement.storage import (
    CompletionStore, PostStatedCache, PostRevealedCache, PreTaskStatedCache, model_short_name,
    save_stated, stated_exist, MeasurementCache, MeasurementStats, save_yaml, reconstruct_measurements,
    ExperimentStore, RankingCache,
)
from src.measurement.storage.failures import save_run_failures
from src.types import MeasurementFailure
from src.measurement.storage.completions import generate_completions
from src.measurement.storage.base import build_measurement_config
from src.fitting.thurstonian_fitting import compute_pair_agreement, save_thurstonian, _config_hash
from src.fitting.trueskill_fitting import sample_ranking_groups, fit_trueskill_from_rankings
from src.fitting.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.fitting.thurstonian_fitting.active_learning_checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from src.measurement.runners.utils.experiment_utils import (
    setup_experiment,
    parse_scale_from_template,
    flip_pairs,
    apply_pair_order,
    compute_thurstonian_max_iter,
    build_fit_kwargs,
    build_configurations,
)
from src.measurement.runners.utils.runner_utils import RunnerStats, _get_activation_completions_path
from src.models import get_client
from src.models.registry import should_capture_reasoning

logger = logging.getLogger(__name__)


def _get_completion_client(config, ctx):
    """Get client for loading completions - uses completion_model if set, else rating model."""
    if config.completion_model:
        return get_client(config.completion_model)
    return ctx.client


def _build_run_name_suffix(config, model_short: str) -> str:
    """Build run name suffix that includes completion model if different from rating model."""
    suffix = ""
    if config.completion_model:
        comp_short = model_short_name(config.completion_model)
        if comp_short != model_short:
            suffix += f"_comp_{comp_short}"
    if config.measurement_system_prompt:
        h = hashlib.sha256(config.measurement_system_prompt.encode()).hexdigest()[:8]
        suffix += f"_sys{h}"
    if config.include_task_ids_file:
        suffix += f"_{config.include_task_ids_file.stem}"
    return suffix


class PostTaskRunConfig(TypedDict):
    model: str
    template_name: str
    template_tags: dict
    response_format: str
    completion_seed: int
    rating_seed: int
    temperature: float


class PostTaskRevealedRunConfig(PostTaskRunConfig):
    order: str


def build_stated_builder(
    template, response_format_name: str, post_task: bool = False, system_prompt: str | None = None, context_messages: list[Message] | None = None,
):
    """Build a stated preference prompt builder."""
    response_format = get_stated_response_format(parse_scale_from_template(template), response_format_name)
    builder_cls = PostTaskStatedPromptBuilder if post_task else PreTaskStatedPromptBuilder
    kwargs: dict = {
        "measurer": StatedScoreMeasurer(),
        "response_format": response_format,
        "template": template,
    }
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    if context_messages:
        kwargs["context_messages"] = context_messages
    return builder_cls(**kwargs)


def build_revealed_builder(template, response_format_name: str, post_task: bool = False, reasoning_mode: bool = False, system_prompt: str | None = None, context_messages: list[Message] | None = None):
    """Build a revealed preference prompt builder."""
    tags = template.tags_dict
    language = tags.get("language", "en")
    task_label_names = tags.get("task_label_names", "letter")
    task_a_label, task_b_label = TASK_LABELS[(task_label_names, language)]
    response_format = get_revealed_response_format(task_a_label, task_b_label, response_format_name, reasoning_mode)
    builder_cls = PostTaskRevealedPromptBuilder if post_task else PreTaskRevealedPromptBuilder
    kwargs: dict = {
        "measurer": RevealedPreferenceMeasurer(),
        "response_format": response_format,
        "template": template,
    }
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    if context_messages:
        kwargs["context_messages"] = context_messages
    return builder_cls(**kwargs)


async def run_post_task_stated_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run post-task stated measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="post_task_stated", config_overrides=config_overrides, client=client)
    config = ctx.config

    completion_seeds = config.completion_seeds or config.generation_seeds
    model_short = model_short_name(ctx.client.canonical_model_name)

    configurations = build_configurations(ctx, config)
    stats = RunnerStats(total_runs=len(completion_seeds) * len(configurations))

    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None
    activation_completions_path = _get_activation_completions_path(config.activations_model)
    completion_client = _get_completion_client(config, ctx)
    run_name_suffix = _build_run_name_suffix(config, model_short)

    for completion_seed in completion_seeds:
        store = CompletionStore(client=completion_client, seed=completion_seed, activation_completions_path=activation_completions_path)
        if not store.exists():
            continue

        # When use_all_completions=True, load all completions; otherwise filter to sampled tasks
        task_lookup = None if config.use_all_completions else ctx.task_lookup
        task_completions = store.load(task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}
        tasks_data = [tc.task for tc in task_completions] * config.n_samples

        for cfg in configurations:
            run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}{run_name_suffix}_cseed{completion_seed}_rseed{cfg.seed}"

            # Skip if already done in experiment store
            if exp_store and exp_store.exists("post_task_stated", run_name):
                stats.mark_skipped()
                if progress_callback:
                    progress_callback(stats)
                continue

            cache = PostStatedCache(
                ctx.client.canonical_model_name, cfg.template, cfg.response_format,
                completion_seed, cfg.seed,
                system_prompt=config.measurement_system_prompt,
                completion_model=config.completion_model,
            )

            builder = build_stated_builder(
                cfg.template, cfg.response_format, post_task=True,
                system_prompt=config.measurement_system_prompt,
            )

            async def measure_fn(data: list[tuple[Task, str]]) -> MeasurementBatch:
                return await measure_post_task_stated_async(
                    client=ctx.client,
                    data=data,
                    builder=builder,
                    semaphore=semaphore,
                    temperature=config.temperature,
                    seed=cfg.seed,
                )

            scores, batch_stats = await cache.get_or_measure_async(
                tasks_data, completion_lookup, measure_fn
            )
            stats.add_from_batch_stats(batch_stats)

            run_config: PostTaskRunConfig = {
                "model": ctx.client.model_name,
                "template_name": cfg.template.name,
                "template_tags": dict(cfg.template.tags_dict),
                "response_format": cfg.response_format,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }

            if exp_store and scores:
                measurements_dicts = [{"task_id": s.task.id, "score": s.score, "origin": s.task.origin.name} for s in scores]
                exp_store.save_stated("post_task_stated", run_name, measurements_dicts, dict(run_config))

            if progress_callback:
                progress_callback(stats)

    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "post_task_stated")
    return stats.to_dict()


async def run_pre_task_revealed_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run pre-task revealed measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="pre_task_revealed", config_overrides=config_overrides, client=client)
    config = ctx.config

    configurations = build_configurations(ctx, config, include_order=True)
    all_pairs = list(combinations(ctx.tasks, 2))
    stats = RunnerStats(total_runs=len(configurations))
    model_short = model_short_name(ctx.client.canonical_model_name)

    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None
    activation_completions_path = _get_activation_completions_path(config.activations_model)
    sys_suffix = _build_run_name_suffix(config, model_short)

    for cfg in configurations:
        seed_suffix = f"_seed{cfg.seed}" if cfg.seed is not None else ""
        run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_{cfg.order}{seed_suffix}{sys_suffix}"

        # Skip if already done in experiment store
        if exp_store and exp_store.exists("pre_task_revealed", run_name):
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats)
            continue

        cache = MeasurementCache(cfg.template, ctx.client, cfg.response_format, cfg.order, seed=cfg.seed, system_prompt=config.measurement_system_prompt)
        pairs = apply_pair_order(all_pairs, cfg.order, config.pair_order_seed, config.include_reverse_order)
        pairs_with_repeats = pairs * config.n_samples

        builder = build_revealed_builder(cfg.template, cfg.response_format, post_task=False, reasoning_mode=config.reasoning_mode, system_prompt=config.measurement_system_prompt)

        async def measure_fn(pairs_to_query: list[tuple[Task, Task]]) -> MeasurementBatch:
            return await measure_pre_task_revealed_async(
                client=ctx.client,
                pairs=pairs_to_query,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=cfg.seed,
            )

        measurements, batch_stats = await cache.get_or_measure_async(
            pairs_with_repeats, measure_fn, ctx.task_lookup
        )
        stats.add_from_batch_stats(batch_stats)

        if exp_store and measurements:
            measurements_dicts = [
                {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice, "origin_a": m.task_a.origin.name, "origin_b": m.task_b.origin.name}
                | ({"raw_response": m.raw_response} if m.raw_response else {})
                for m in measurements
            ]
            config_dict = build_measurement_config(
                template=cfg.template,
                client=ctx.client,
                response_format=cfg.response_format,
                order=cfg.order,
                seed=cfg.seed,
                temperature=config.temperature,
            )
            exp_store.save_revealed("pre_task_revealed", run_name, measurements_dicts, config_dict)

        if progress_callback:
            progress_callback(stats)

    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "pre_task_revealed")
    return stats.to_dict()


async def run_post_task_revealed_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run post-task revealed measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="post_task_revealed", config_overrides=config_overrides, client=client)
    config = ctx.config

    completion_seeds = config.completion_seeds or config.generation_seeds
    model_short = model_short_name(ctx.client.canonical_model_name)
    configurations = build_configurations(ctx, config, include_order=True)
    all_pairs = list(combinations(ctx.tasks, 2))
    stats = RunnerStats(total_runs=len(completion_seeds) * len(configurations))

    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None
    activation_completions_path = _get_activation_completions_path(config.activations_model)
    completion_client = _get_completion_client(config, ctx)
    run_name_suffix = _build_run_name_suffix(config, model_short)

    for completion_seed in completion_seeds:
        store = CompletionStore(client=completion_client, seed=completion_seed, activation_completions_path=activation_completions_path)
        if not store.exists():
            continue

        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

        # Filter pairs to only those where both tasks have completions
        pairs_with_completions = [
            (a, b) for a, b in all_pairs
            if a.id in completion_lookup and b.id in completion_lookup
        ]

        for cfg in configurations:
            seed_suffix = f"_seed{cfg.seed}" if cfg.seed is not None else ""
            run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_{cfg.order}{run_name_suffix}_cseed{completion_seed}{seed_suffix}"

            # Skip if already done in experiment store
            if exp_store and exp_store.exists("post_task_revealed", run_name):
                stats.mark_skipped()
                if progress_callback:
                    progress_callback(stats)
                continue

            cache = PostRevealedCache(
                ctx.client.canonical_model_name, cfg.template, cfg.response_format,
                cfg.order, completion_seed, cfg.seed,
            )
            pairs = apply_pair_order(pairs_with_completions, cfg.order, config.pair_order_seed, config.include_reverse_order)
            pairs_with_repeats = pairs * config.n_samples

            builder = build_revealed_builder(cfg.template, cfg.response_format, post_task=True, reasoning_mode=config.reasoning_mode, system_prompt=config.measurement_system_prompt)

            async def measure_fn(data: list[tuple[Task, Task, str, str]]) -> MeasurementBatch:
                return await measure_post_task_revealed_async(
                    client=ctx.client,
                    data=data,
                    builder=builder,
                    semaphore=semaphore,
                    temperature=config.temperature,
                    seed=cfg.seed,
                )

            measurements, batch_stats = await cache.get_or_measure_async(
                pairs_with_repeats, completion_lookup, measure_fn, ctx.task_lookup
            )
            stats.add_from_batch_stats(batch_stats)

            run_config: PostTaskRevealedRunConfig = {
                "model": ctx.client.model_name,
                "template_name": cfg.template.name,
                "template_tags": dict(cfg.template.tags_dict),
                "response_format": cfg.response_format,
                "order": cfg.order,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }

            if exp_store and measurements:
                measurements_dicts = [
                    {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice, "origin_a": m.task_a.origin.name, "origin_b": m.task_b.origin.name}
                    | ({"raw_response": m.raw_response} if m.raw_response else {})
                    for m in measurements
                ]
                exp_store.save_revealed("post_task_revealed", run_name, measurements_dicts, dict(run_config))

            if progress_callback:
                progress_callback(stats)

    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "post_task_revealed")
    return stats.to_dict()


async def run_pre_task_stated_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run pre-task stated measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="pre_task_stated", config_overrides=config_overrides, client=client)
    config = ctx.config

    configurations = build_configurations(ctx, config)
    tasks_data = ctx.tasks * config.n_samples
    stats = RunnerStats(total_runs=len(configurations))
    model_short = model_short_name(ctx.client.canonical_model_name)

    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None
    activation_completions_path = _get_activation_completions_path(config.activations_model)
    sys_suffix = _build_run_name_suffix(config, model_short)

    for cfg in configurations:
        run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_seed{cfg.seed}{sys_suffix}"

        # Skip if already done in experiment store
        if exp_store and exp_store.exists("pre_task_stated", run_name):
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats)
            continue

        cache = PreTaskStatedCache(cfg.template, ctx.client, cfg.response_format, cfg.seed)
        builder = build_stated_builder(cfg.template, cfg.response_format, post_task=False)

        async def measure_fn(tasks_to_query: list[Task]) -> MeasurementBatch:
            return await measure_pre_task_stated_async(
                client=ctx.client,
                tasks=tasks_to_query,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=cfg.seed,
            )

        scores, batch_stats = await cache.get_or_measure_async(tasks_data, measure_fn)
        stats.add_from_batch_stats(batch_stats)

        config_dict = build_measurement_config(
            template=cfg.template,
            client=ctx.client,
            response_format=cfg.response_format,
            seed=cfg.seed,
            temperature=config.temperature,
        )

        if exp_store and scores:
            measurements_dicts = [{"task_id": s.task.id, "score": s.score, "origin": s.task.origin.name} for s in scores]
            exp_store.save_stated("pre_task_stated", run_name, measurements_dicts, config_dict)

        if progress_callback:
            progress_callback(stats)

    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "pre_task_stated")
    return stats.to_dict()


async def _measure_revealed_pairs(
    pairs: list,
    builder,
    client,
    semaphore: asyncio.Semaphore,
    temperature: float,
    seed: int,
    completion_lookup: dict[str, str] | None = None,
    async_client: "AsyncOpenAI | None" = None,
):
    """Measure revealed preferences for pairs, with or without completions."""
    if completion_lookup is not None:
        data = [
            (a, b, completion_lookup[a.id], completion_lookup[b.id])
            for a, b in pairs
        ]
        return await measure_post_task_revealed_async(
            client=client, data=data, builder=builder,
            semaphore=semaphore, temperature=temperature, seed=seed,
        )
    else:
        return await measure_pre_task_revealed_async(
            client=client, pairs=pairs, builder=builder,
            semaphore=semaphore, temperature=temperature, seed=seed,
            async_client=async_client,
        )


async def run_active_learning_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    post_task: bool = False,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run active learning with shared semaphore. Supports both pre-task and post-task modes."""
    # Pop resume before setup_experiment — it's a runtime flag, not config
    overrides = dict(config_overrides) if config_overrides else {}
    resume = overrides.pop("resume", False)

    expected_mode = "post_task_active_learning" if post_task else "pre_task_active_learning"
    ctx = setup_experiment(config_path, expected_mode=expected_mode, config_overrides=overrides or None, client=client)
    config, al = ctx.config, ctx.config.active_learning
    model_short = model_short_name(ctx.client.canonical_model_name)
    measurement_type = "post_task_active_learning" if post_task else "pre_task_active_learning"

    if not config.experiment_id:
        raise ValueError("experiment_id is required for active learning runs")
    exp_store = ExperimentStore(config.experiment_id)

    # For post-task, load completions
    completion_lookup: dict[str, str] | None = None
    completion_seed: int | None = None
    tasks_for_learning = ctx.tasks
    activation_completions_path = _get_activation_completions_path(config.activations_model)
    completion_client = _get_completion_client(config, ctx)
    run_name_suffix = _build_run_name_suffix(config, model_short)
    if post_task:
        completion_seeds = config.completion_seeds or config.generation_seeds
        # Use first completion seed for active learning
        completion_seed = completion_seeds[0]
        store = CompletionStore(client=completion_client, seed=completion_seed, activation_completions_path=activation_completions_path)
        if not store.exists():
            raise ValueError(f"Completions not found for seed {completion_seed}")
        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}
        # Filter tasks to only those with completions
        tasks_for_learning = [t for t in ctx.tasks if t.id in completion_lookup]

    rng = np.random.default_rng(al.seed)
    max_iter = compute_thurstonian_max_iter(config)
    configurations = build_configurations(ctx, config, include_order=True)
    stats = RunnerStats(total_runs=len(configurations))

    async_client = ctx.client._make_async_client()
    for cfg in configurations:
        cache = MeasurementCache(cfg.template, ctx.client, cfg.response_format, cfg.order, seed=cfg.seed, completion_seed=completion_seed, system_prompt=config.measurement_system_prompt)

        # Build run_name consistent with other runners
        if post_task:
            run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_{cfg.order}{run_name_suffix}_cseed{completion_seed}_rseed{cfg.seed}"
        else:
            seed_suffix = f"_seed{cfg.seed}" if cfg.seed is not None else ""
            run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_{cfg.order}{seed_suffix}{run_name_suffix}"

        # Skip if already done in experiment store (unless resuming)
        if exp_store and exp_store.exists(measurement_type, run_name) and not resume:
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats)
            continue

        state = ActiveLearningState(tasks=tasks_for_learning)
        builder = build_revealed_builder(cfg.template, cfg.response_format, post_task=post_task, reasoning_mode=config.reasoning_mode, system_prompt=config.measurement_system_prompt)

        # Create measure function that cache can call for API requests
        # Use default args to capture current loop values (avoid closure capture bug)
        async def measure_fn(
            pairs: list[tuple[Task, Task]],
            *,
            _builder=builder,
            _seed=cfg.seed,
            _completion_lookup=completion_lookup,
            _async_client=async_client,
        ):
            return await _measure_revealed_pairs(
                pairs=pairs,
                builder=_builder,
                client=ctx.client,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=_seed,
                completion_lookup=_completion_lookup,
                async_client=_async_client,
            )

        rank_correlations = []
        config_stats = MeasurementStats()
        start_iteration = 0

        # Compute the checkpoint directory (same path as final run_dir)
        checkpoint_dir = exp_store.base_dir / measurement_type / run_name

        # Resume from checkpoint
        if resume and checkpoint_exists(checkpoint_dir):
            ckpt = load_checkpoint(checkpoint_dir)
            restored = reconstruct_measurements(ckpt["comparisons"], ctx.task_lookup)
            state.add_comparisons(restored)
            state.iteration = ckpt["iteration"]
            rank_correlations = ckpt["rank_correlations"]
            state.fit(**build_fit_kwargs(config, max_iter))
            start_iteration = ckpt["iteration"]

            pairs_to_query = select_next_pairs(
                state, batch_size=al.batch_size,
                p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
            )
            pairs_to_query = apply_pair_order(pairs_to_query, cfg.order, config.pair_order_seed, config.include_reverse_order)
            logger.info("Resumed from checkpoint at iteration %d with %d comparisons", start_iteration, len(state.comparisons))
        elif resume:
            raise FileNotFoundError(f"--resume passed but no checkpoint.yaml found in {checkpoint_dir}")
        else:
            # Always start with d-regular graph for broad coverage
            # Cache handles skipping already-measured pairs at query time
            pairs_to_query = generate_d_regular_pairs(tasks_for_learning, al.initial_degree, rng)
            pairs_to_query = apply_pair_order(pairs_to_query, cfg.order, config.pair_order_seed, config.include_reverse_order)

        def on_chunk(chunk_stats: MeasurementStats, chunk: int, total_chunks: int):
            stats.successes = config_stats.api_successes + chunk_stats.api_successes
            stats.failures = config_stats.api_failures + chunk_stats.api_failures
            stats.cache_hits = config_stats.cache_hits + chunk_stats.cache_hits
            stats.chunk = chunk
            stats.total_chunks = total_chunks
            if progress_callback:
                progress_callback(stats)

        consecutive_api_failures = 0
        aborted_api_failures = False

        for iteration in range(start_iteration, al.max_iterations):
            if not pairs_to_query:
                break

            # Request measurements - cache handles checking what's cached vs needs API
            pairs_with_repeats = pairs_to_query * config.n_samples

            # Update stats with iteration info before API calls
            stats.iteration = iteration + 1
            stats.iteration_pairs = len(pairs_to_query)
            if progress_callback:
                progress_callback(stats)

            measurements, iter_stats = await cache.get_or_measure_async(
                pairs_with_repeats, measure_fn, ctx.task_lookup,
                on_chunk_complete=on_chunk,
            )
            config_stats += iter_stats

            # Update display stats (actual totals updated after loop via config_stats)
            stats.successes = config_stats.api_successes
            stats.failures = config_stats.api_failures
            stats.cache_hits = config_stats.cache_hits
            stats.chunk = None
            stats.total_chunks = None
            if progress_callback:
                progress_callback(stats)

            # Check for API-dominated iteration (false convergence detection)
            iter_total = iter_stats.api_successes + iter_stats.api_failures
            api_dominated = iter_total > 0 and iter_stats.api_side_failure_count / iter_total > 0.5

            # Always add comparisons and checkpoint — even api-dominated
            # iterations contain valid comparisons alongside the failures
            state.add_comparisons(measurements)
            state.iteration = iteration + 1
            state.fit(**build_fit_kwargs(config, max_iter))

            ckpt_comparisons = [
                {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice}
                for m in state.comparisons
            ]
            save_checkpoint(
                checkpoint_dir,
                iteration=state.iteration,
                comparisons_dicts=ckpt_comparisons,
                rank_correlations=rank_correlations,
            )

            if api_dominated:
                consecutive_api_failures += 1
                logger.warning(
                    "Iteration %d: %d/%d API-side failures (%.0f%%) — skipping convergence check",
                    iteration + 1, iter_stats.api_side_failure_count, iter_total,
                    100 * iter_stats.api_side_failure_count / iter_total,
                )
                if consecutive_api_failures >= 3:
                    logger.error(
                        "Aborting: %d consecutive API-failure-dominated iterations", consecutive_api_failures,
                    )
                    aborted_api_failures = True
                    break
                continue

            consecutive_api_failures = 0
            converged, correlation = check_convergence(state, al.convergence_threshold)
            if math.isnan(correlation):
                break
            rank_correlations.append(float(correlation))

            # Update stats with correlation info
            stats.rank_correlation = correlation
            stats.total_comparisons = len(state.comparisons)
            if progress_callback:
                progress_callback(stats)

            if converged:
                break

            pairs_to_query = select_next_pairs(
                state, batch_size=al.batch_size,
                p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
            )
            pairs_to_query = apply_pair_order(pairs_to_query, cfg.order, config.pair_order_seed, config.include_reverse_order)

        # Update runner stats from this configuration
        stats.add_from_batch_stats(config_stats)
        # Mark as skipped if we only used cached data (no API calls made)
        if config_stats.api_successes == 0 and config_stats.api_failures == 0:
            stats.skipped += 1

        final_converged, _ = check_convergence(state, al.convergence_threshold)

        # Build config dict for saving
        run_config = {"n_tasks": config.n_tasks, "seed": al.seed, "generation_seed": cfg.seed}
        if config.measurement_system_prompt:
            run_config["system_prompt"] = config.measurement_system_prompt
        config_dict = build_measurement_config(
            template=cfg.template,
            client=ctx.client,
            response_format=cfg.response_format,
            order=cfg.order,
            seed=cfg.seed,
            temperature=config.temperature,
        )
        if post_task:
            config_dict["completion_seed"] = completion_seed

        active_learning_summary = {
            "n_tasks": config.n_tasks,
            "seed": al.seed,
            "generation_seed": cfg.seed,
            "converged": bool(final_converged) and not aborted_api_failures,
            "aborted_api_failures": aborted_api_failures,
            "n_iterations": state.iteration,
            "unique_pairs_queried": len(state.sampled_pairs),
            "total_comparisons": len(state.comparisons),
            "pair_agreement": float(compute_pair_agreement(state.comparisons)),
            "rank_correlations": rank_correlations,
        }
        if config.measurement_system_prompt:
            active_learning_summary["system_prompt"] = config.measurement_system_prompt

        # Save to experiment store
        if state.comparisons:
            measurements_dicts = [
                {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice, "origin_a": m.task_a.origin.name, "origin_b": m.task_b.origin.name}
                | ({"raw_response": m.raw_response} if m.raw_response else {})
                for m in state.comparisons
            ]
            run_dir = exp_store.save(
                measurement_type, run_name, measurements_dicts, config_dict,
                extra_files={"active_learning.yaml": active_learning_summary},
            )
            # Save Thurstonian fit to experiment store
            save_thurstonian(state.current_fit, run_dir / "thurstonian.yaml", "active_learning", run_config)

        # Clean up checkpoint on successful completion (keep on abort for --resume)
        if not aborted_api_failures:
            ckpt_path = checkpoint_dir / "checkpoint.yaml"
            if ckpt_path.exists():
                ckpt_path.unlink()

        if progress_callback:
            progress_callback(stats)

    save_run_failures(stats.all_failures, exp_store.base_dir, measurement_type)
    await async_client.close()
    return stats.to_dict()


async def run_post_task_active_learning_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run post-task active learning with shared semaphore."""
    return await run_active_learning_async(config_path, semaphore, progress_callback, post_task=True, config_overrides=config_overrides, client=client)


async def run_pre_task_active_learning_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run pre-task active learning with shared semaphore."""
    return await run_active_learning_async(config_path, semaphore, progress_callback, post_task=False, config_overrides=config_overrides, client=client)


async def run_completion_generation_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run completion generation with shared semaphore."""
    # Completion generation needs more tokens - use 1024 as minimum, but config can override higher
    overrides = dict(config_overrides) if config_overrides else {}
    if "max_new_tokens" not in overrides:
        overrides["max_new_tokens"] = max(1024, overrides.get("max_new_tokens", 1024))
    ctx = setup_experiment(config_path, expected_mode="completion_generation", require_templates=False, config_overrides=overrides, client=client)
    config = ctx.config
    activation_completions_path = _get_activation_completions_path(config.activations_model)

    stats = RunnerStats(total_runs=len(config.generation_seeds))

    for seed in config.generation_seeds:
        store = CompletionStore(client=ctx.client, seed=seed, activation_completions_path=activation_completions_path)

        existing_ids = store.get_existing_task_ids()
        tasks_to_complete = [t for t in ctx.tasks if t.id not in existing_ids]

        if not tasks_to_complete:
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats)
            continue

        # generate_completions is sync, run in executor
        capture_reasoning = should_capture_reasoning(ctx.client.canonical_model_name)

        loop = asyncio.get_event_loop()
        completions = await loop.run_in_executor(
            None,
            lambda: generate_completions(
                client=ctx.client,
                tasks=tasks_to_complete,
                temperature=config.temperature,
                max_concurrent=ctx.max_concurrent,
                seed=seed,
                detect_refusals=config.detect_refusals,
                system_prompt=config.measurement_system_prompt,
                capture_reasoning=capture_reasoning,
            )
        )

        run_config = {
            "model": ctx.client.model_name,
            "n_tasks": config.n_tasks,
            "task_origins": config.task_origins,
            "temperature": config.temperature,
            "seed": seed,
        }
        store.save(completions, run_config)

        stats.successes += len(completions)
        stats.failures += len(tasks_to_complete) - len(completions)
        stats.completed += 1

        if progress_callback:
            progress_callback(stats)

    return stats.to_dict()


def _save_trueskill_result(result, ctx, template_name: str, response_format: str, seed: int) -> None:
    """Save TrueSkill result to YAML file."""
    output_dir = Path("results/trueskill") / model_short_name(ctx.client.canonical_model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{template_name}_{response_format}_seed{seed}.yaml"
    data = result.to_dict()
    data.update({
        "method": "trueskill",
        "model": ctx.client.canonical_model_name,
        "template_name": template_name,
        "response_format": response_format,
        "seed": seed,
        "ranking": [t.id for t in result.ranking()],
    })
    save_yaml(data, output_path)


def _shuffle_task_groups(
    groups: list[list["Task"]], rng: np.random.Generator
) -> list[list["Task"]]:
    """Shuffle task order within each group to control for position bias."""
    return [list(rng.permutation(group)) for group in groups]


def _compute_n_groups(n_tasks: int, appearances_per_task: int, n_tasks_per_ranking: int) -> int:
    """Compute number of groups needed for target appearances per task."""
    return math.ceil(n_tasks * appearances_per_task / n_tasks_per_ranking)


async def run_pre_task_ranking_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run pre-task ranking measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="pre_task_ranking", config_overrides=config_overrides, client=client)
    config = ctx.config
    ranking_cfg = config.ranking

    configurations = build_configurations(ctx, config)
    stats = RunnerStats(total_runs=len(configurations))
    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None

    rng = np.random.default_rng(ranking_cfg.seed)
    cache = RankingCache(ctx.client.canonical_model_name)

    n_groups = _compute_n_groups(len(ctx.tasks), ranking_cfg.appearances_per_task, ranking_cfg.n_tasks_per_ranking)

    for cfg in configurations:
        # Sample task groups
        task_groups = sample_ranking_groups(
            ctx.tasks, ranking_cfg.n_tasks_per_ranking, n_groups, rng
        )

        # Shuffle task order within groups to control for position bias
        if ranking_cfg.shuffle_task_order:
            task_groups = _shuffle_task_groups(task_groups, rng)

        # Filter already-measured groups
        existing = cache.get_measured_groups(cfg.template.name, cfg.response_format, cfg.seed)
        groups_to_measure = [g for g in task_groups if frozenset(t.id for t in g) not in existing]

        if not groups_to_measure:
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats)
            continue

        # Build task labels (A, B, C, D, E)
        task_labels = tuple(chr(65 + i) for i in range(ranking_cfg.n_tasks_per_ranking))
        response_format = get_ranking_response_format(task_labels, cfg.response_format)
        builder = PreTaskRankingPromptBuilder(
            measurer=RankingMeasurer(),
            response_format=response_format,
            template=cfg.template,
        )

        batch = await measure_pre_task_ranking_async(
            ctx.client, groups_to_measure, builder, semaphore,
            config.temperature, cfg.seed
        )

        cache.add(batch.successes, cfg.template.name, cfg.response_format, cfg.seed)
        stats.add_batch_with_failures(len(batch.successes), batch.failures)

        # Fit TrueSkill on all measurements for this config
        all_rankings = cache.get_all_measurements(
            cfg.template.name, cfg.response_format, cfg.seed, ctx.task_lookup
        )
        if all_rankings:
            result = fit_trueskill_from_rankings(all_rankings)
            _save_trueskill_result(result, ctx, cfg.template.name, cfg.response_format, cfg.seed)

        if progress_callback:
            progress_callback(stats)

    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "pre_task_ranking")
    return stats.to_dict()


async def run_post_task_ranking_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
    config_overrides: dict | None = None,
    client=None,
) -> dict:
    """Run post-task ranking measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="post_task_ranking", config_overrides=config_overrides, client=client)
    config = ctx.config
    ranking_cfg = config.ranking

    completion_seeds = config.completion_seeds or config.generation_seeds
    configurations = build_configurations(ctx, config)
    stats = RunnerStats(total_runs=len(completion_seeds) * len(configurations))
    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None
    activation_completions_path = _get_activation_completions_path(config.activations_model)
    completion_client = _get_completion_client(config, ctx)
    model_short = model_short_name(ctx.client.canonical_model_name)
    run_name_suffix = _build_run_name_suffix(config, model_short)

    for completion_seed in completion_seeds:
        store = CompletionStore(client=completion_client, seed=completion_seed, activation_completions_path=activation_completions_path)
        if not store.exists():
            continue

        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

        # Filter tasks to only those with completions
        tasks_with_completions = [t for t in ctx.tasks if t.id in completion_lookup]

        rng = np.random.default_rng(ranking_cfg.seed)
        cache = RankingCache(ctx.client.canonical_model_name)
        n_groups = _compute_n_groups(len(tasks_with_completions), ranking_cfg.appearances_per_task, ranking_cfg.n_tasks_per_ranking)

        for cfg in configurations:
            # Sample task groups from tasks with completions
            task_groups = sample_ranking_groups(
                tasks_with_completions, ranking_cfg.n_tasks_per_ranking, n_groups, rng
            )

            # Shuffle task order within groups to control for position bias
            if ranking_cfg.shuffle_task_order:
                task_groups = _shuffle_task_groups(task_groups, rng)

            # Filter already-measured groups (keyed by completion_seed as well)
            cache_key_suffix = f"_cseed{completion_seed}"
            existing = cache.get_measured_groups(
                cfg.template.name + cache_key_suffix, cfg.response_format, cfg.seed
            )
            groups_to_measure = [g for g in task_groups if frozenset(t.id for t in g) not in existing]

            if not groups_to_measure:
                stats.mark_skipped()
                if progress_callback:
                    progress_callback(stats)
                continue

            # Build task labels (A, B, C, D, E)
            task_labels = tuple(chr(65 + i) for i in range(ranking_cfg.n_tasks_per_ranking))
            response_format = get_ranking_response_format(task_labels, cfg.response_format)
            builder = PostTaskRankingPromptBuilder(
                measurer=RankingMeasurer(),
                response_format=response_format,
                template=cfg.template,
            )

            # Build prompts with completions
            prompts = []
            for group in groups_to_measure:
                completions = [completion_lookup[t.id] for t in group]
                prompts.append(builder.build(group, completions))

            batch = await _measure_async(
                ctx.client, prompts, semaphore, config.temperature, cfg.seed, RankingMeasurement
            )

            cache.add(
                batch.successes, cfg.template.name + cache_key_suffix, cfg.response_format, cfg.seed
            )
            stats.add_batch_with_failures(len(batch.successes), batch.failures)

            # Fit TrueSkill on all measurements for this config
            all_rankings = cache.get_all_measurements(
                cfg.template.name + cache_key_suffix, cfg.response_format, cfg.seed, ctx.task_lookup
            )
            if all_rankings:
                result = fit_trueskill_from_rankings(all_rankings)
                _save_trueskill_result(
                    result, ctx,
                    f"{cfg.template.name}_cseed{completion_seed}",
                    cfg.response_format, cfg.seed
                )

            if progress_callback:
                progress_callback(stats)

    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "post_task_ranking")
    return stats.to_dict()


def _get_runners():
    """Lazy load runners to avoid circular imports."""
    from src.measurement.runners.open_ended_runners import run_open_ended_async
    return {
        "completion_generation": run_completion_generation_async,
        "pre_task_stated": run_pre_task_stated_async,
        "pre_task_revealed": run_pre_task_revealed_async,
        "pre_task_active_learning": run_pre_task_active_learning_async,
        "post_task_stated": run_post_task_stated_async,
        "post_task_revealed": run_post_task_revealed_async,
        "post_task_active_learning": run_post_task_active_learning_async,
        "pre_task_ranking": run_pre_task_ranking_async,
        "post_task_ranking": run_post_task_ranking_async,
        "open_ended": run_open_ended_async,
    }

RUNNERS = _get_runners()
