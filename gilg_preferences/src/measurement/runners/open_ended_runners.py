"""Runner for open-ended valence measurement experiments."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from collections.abc import Iterable

from src.models import OpenAICompatibleClient
from src.task_data import Task
from src.measurement.elicitation.open_ended_measure import measure_open_ended_stated_async
from src.measurement.elicitation.response_format import OpenEndedFormat
from src.measurement.elicitation.measurer import OpenEndedMeasurer
from src.measurement.elicitation.prompt_templates import PromptTemplate
from src.measurement.elicitation.prompt_templates.builders import OpenEndedPromptBuilder
from src.measurement.runners.config import set_experiment_id
from src.measurement.runners.open_ended_config import OpenEndedMeasurementConfig, load_open_ended_config
from src.measurement.runners.utils.runner_utils import RunnerStats, _get_activation_completions_path
from src.measurement.runners.utils.experiment_utils import setup_experiment
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.storage import CompletionStore, ExperimentStore, model_short_name
from src.measurement.storage.failures import save_run_failures
from src.models import get_client, GenerateRequest
from src.task_data import load_tasks
from src.types import OpenEndedResponse

if TYPE_CHECKING:
    ProgressCallback = Callable[[RunnerStats], None]


class OpenEndedExperimentContext:
    """Context for open-ended measurement experiments."""

    def __init__(
        self,
        config: OpenEndedMeasurementConfig,
        client: OpenAICompatibleClient,
        tasks: list[Task],
        ood_tasks: list[Task] | None = None,
        task_lookup: dict[str, Task] | None = None,
    ):
        self.config = config
        self.client = client
        self.tasks = tasks
        self.ood_tasks = ood_tasks or []
        self.task_lookup = task_lookup or {t.id: t for t in tasks}


async def _save_open_ended_results(
    responses: list[OpenEndedResponse],
    experiment_id: str,
    variant: str,
    rating_seed: int,
    run_config: dict,
) -> Path:
    """Save open-ended measurement results to JSON.

    Returns:
        Path to saved results file
    """
    exp_dir = Path("results/experiments") / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    filename = f"open_ended_{variant}_rseed{rating_seed}.json"
    filepath = exp_dir / filename

    # Convert responses to JSON-serializable format
    results = []
    for resp in responses:
        results.append({
            "task_id": resp.task.id,
            "task_origin": resp.task.origin.value,
            "raw_response": resp.raw_response,
            "semantic_valence_score": resp.semantic_valence_score,
        })

    # Save results
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath


async def run_open_ended_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: "ProgressCallback | None" = None,
) -> dict:
    """Run open-ended measurement with shared semaphore.

    Args:
        config_path: Path to OpenEndedMeasurementConfig YAML file
        semaphore: Semaphore for rate limiting concurrent requests
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with aggregated statistics
    """
    # Load config
    config = load_open_ended_config(config_path)
    if config.experiment_id is None:
        config.experiment_id = set_experiment_id()

    # Setup experiment context
    # Load client and tasks without requiring full experiment setup
    client = get_client(config.model)

    # Load in-distribution tasks
    origins = config.get_origin_datasets()
    tasks = load_tasks(n=config.n_tasks, origins=origins, seed=config.task_sampling_seed)

    # Create task lookup for later
    task_lookup = {t.id: t for t in tasks}

    activation_completions_path = _get_activation_completions_path(config.activations_model)

    # Filter to tasks with activations if needed
    in_dist_task_ids = {t.id for t in tasks}

    # Load OOD tasks if needed
    ood_tasks: list[Task] = []
    if config.include_out_of_distribution:
        ood_dataset_origins = config.get_ood_origin_datasets()
        all_ood_tasks = load_tasks(ood_dataset_origins, max_tasks=config.n_ood_tasks)
        # Filter out any overlap with in-dist tasks
        ood_tasks = [t for t in all_ood_tasks if t.id not in in_dist_task_ids][:config.n_ood_tasks]

    # Load completions
    store = CompletionStore(
        client=client,
        seed=config.completion_seed,
        activation_completions_path=activation_completions_path,
    )
    if not store.exists():
        raise ValueError(f"Completions not found for seed {config.completion_seed}")

    task_completions = store.load(task_lookup)
    completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

    # Filter tasks to only those with completions
    in_dist_tasks_with_completions = [t for t in tasks if t.id in completion_lookup]

    stats = RunnerStats(total_runs=len(config.prompt_variants) * len(config.rating_seeds))
    model_short = model_short_name(client.canonical_model_name)

    exp_store = ExperimentStore(config.experiment_id) if config.experiment_id else None

    # For each variant and rating seed combination
    for variant in config.prompt_variants:
        for rating_seed in config.rating_seeds:
            # Build run name
            run_name = f"open_ended_{variant}_{model_short}_cseed{config.completion_seed}_rseed{rating_seed}"

            # Skip if already done
            if exp_store and exp_store.exists("open_ended", run_name):
                stats.mark_skipped()
                if progress_callback:
                    progress_callback(stats)
                continue

            # Load template for this variant
            templates = load_templates_from_yaml(Path("src/measurement/elicitation/prompt_templates/data/open_ended_v1.yaml"))
            variant_templates = [t for t in templates if variant in t.name]
            if not variant_templates:
                raise ValueError(f"No templates found for variant '{variant}' in open_ended_v1.yaml")
            template = variant_templates[0]

            # Create builder and measurer
            response_format = OpenEndedFormat()
            measurer = OpenEndedMeasurer()
            builder = OpenEndedPromptBuilder(
                measurer=measurer,
                response_format=response_format,
                template=template,
            )

            # Prepare data for measurement
            # In-distribution
            in_dist_data = [
                (t, completion_lookup[t.id])
                for t in in_dist_tasks_with_completions
            ] * config.n_samples

            # OOD (if enabled)
            ood_data: list[tuple[Task, str]] = []
            if config.include_out_of_distribution and ood_tasks:
                # For OOD tasks, generate completions on-the-fly or use cached
                ood_completions = {}
                for t in ood_tasks:
                    if t.id in completion_lookup:
                        ood_completions[t.id] = completion_lookup[t.id]
                    else:
                        # Generate completion for OOD task
                        req = GenerateRequest(messages=[{"role": "user", "content": t.prompt}])
                        result = await client.generate_batch_async([req], semaphore)
                        if result[0].ok:
                            ood_completions[t.id] = result[0].unwrap()

                ood_data = [
                    (t, ood_completions[t.id])
                    for t in ood_tasks
                    if t.id in ood_completions
                ] * config.n_samples

            # Combine data
            all_data = in_dist_data + ood_data

            # Measure
            batch = await measure_open_ended_stated_async(
                client=client,
                data=all_data,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=rating_seed,
            )

            # Update stats
            stats.add_batch_with_failures(
                len(batch.successes),
                batch.failures,
                cache_hits=0,
            )

            # Save results
            if batch.successes:
                run_config = {
                    "model": client.model_name,
                    "variant": variant,
                    "completion_seed": config.completion_seed,
                    "rating_seed": rating_seed,
                    "temperature": config.temperature,
                    "n_tasks": len(in_dist_tasks_with_completions),
                    "n_ood_tasks": len(ood_tasks),
                    "n_samples": config.n_samples,
                }
                await _save_open_ended_results(
                    batch.successes,
                    config.experiment_id,
                    variant,
                    rating_seed,
                    run_config,
                )

            if progress_callback:
                progress_callback(stats)

    # Save failures
    if exp_store:
        save_run_failures(stats.all_failures, exp_store.base_dir, "open_ended")

    return stats.to_dict()
