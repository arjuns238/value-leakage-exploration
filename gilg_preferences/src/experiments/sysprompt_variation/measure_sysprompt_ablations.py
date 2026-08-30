"""Measure post-task stated preferences with system prompt ablations.

Loads completions from a completions experiment and runs all ablations:
(completion_condition × measurement_system_prompt)

Usage:
    python -m src.experiments.sysprompt_variation.measure_sysprompt_ablations configs/sysprompt_variation/measure.yaml
"""

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.measurement.storage import ExperimentStore, TaskCompletion
from src.models import get_client
from src.measurement.elicitation import measure_post_task_stated_async, StatedScoreMeasurer, get_stated_response_format
from src.measurement.elicitation.prompt_templates import PostTaskStatedPromptBuilder, load_templates_from_yaml
from src.measurement.elicitation.semantic_parser import parse_sysprompt_effect_async
from src.measurement.runners.progress import MultiExperimentProgress, console, print_summary
from src.task_data import OriginDataset, Task


load_dotenv()


class MeasurementConfig(BaseModel):
    experiment_name: str
    completions_path: Path
    completion_conditions: list[str]

    rating_model: str
    temperature: float = 1.0
    max_concurrent: int = 50
    generation_seeds: list[int] = [0]

    templates: Path
    template_name: str = "anchored_precise_1_5"

    measurement_system_prompts: dict[str, str | None]

    filter_sysprompt_references: bool = False


def load_config(path: Path) -> MeasurementConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    # Support referencing external system_prompts file
    if "measurement_system_prompts_file" in data:
        prompts_path = path.parent / data.pop("measurement_system_prompts_file")
        with open(prompts_path) as f:
            data["measurement_system_prompts"] = yaml.safe_load(f)

    return MeasurementConfig.model_validate(data)


def load_completions(completions_path: Path, conditions: list[str]) -> tuple[dict[str, list[TaskCompletion]], dict[str, str | None]]:
    """Load completions for specified conditions.

    Returns: (completions dict, system_prompts dict)
    """
    completions = {}
    system_prompts = {}

    for condition in conditions:
        condition_dir = completions_path / "completions" / condition
        condition_path = condition_dir / "completions.json"
        config_path = condition_dir / "config.json"

        if not condition_path.exists():
            raise FileNotFoundError(f"Completions not found: {condition_path}")

        with open(condition_path) as f:
            data = json.load(f)

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            system_prompts[condition] = config.get("system_prompt")
        else:
            system_prompts[condition] = None

        completions[condition] = [
            TaskCompletion(
                task=Task(
                    prompt=c["task_prompt"],
                    origin=OriginDataset[c["origin"]],
                    id=c["task_id"],
                    metadata={},
                ),
                completion=c["completion"],
            )
            for c in data
        ]
    return completions, system_prompts


async def filter_sysprompt_references(
    completions: list[TaskCompletion],
    system_prompt: str | None,
    condition_name: str,
    max_concurrent: int = 20,
) -> list[TaskCompletion]:
    """Filter out completions that reference the system prompt."""
    if system_prompt is None:
        return completions

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    semaphore = asyncio.Semaphore(max_concurrent)
    done_count = [0]

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Filtering {condition_name}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(completions))

        async def check_one(tc: TaskCompletion) -> tuple[TaskCompletion, bool]:
            try:
                async with semaphore:
                    result = await parse_sysprompt_effect_async(
                        system_prompt, tc.task.prompt, tc.completion
                    )
                return tc, result.sysprompt_reference
            except Exception:
                # If parsing fails, keep the completion
                return tc, False
            finally:
                done_count[0] += 1
                progress.update(task, completed=done_count[0])

        results = await asyncio.gather(*[check_one(tc) for tc in completions])

    filtered = [tc for tc, has_ref in results if not has_ref]

    n_removed = len(completions) - len(filtered)
    if n_removed > 0:
        console.print(f"  [yellow]Filtered {n_removed}/{len(completions)} completions with sysprompt references[/yellow]")

    return filtered


async def run_condition(
    completions: list[TaskCompletion],
    system_prompt: str | None,
    template,
    generation_seeds: list[int],
    client,
    semaphore: asyncio.Semaphore,
    progress: MultiExperimentProgress,
    run_name: str,
    temperature: float,
) -> tuple[list[dict], int, int]:
    """Run measurements for a single condition. Returns (results, successes, failures)."""
    response_format = get_stated_response_format((1, 5), "regex")
    builder = PostTaskStatedPromptBuilder(
        measurer=StatedScoreMeasurer(),
        response_format=response_format,
        template=template,
        system_prompt=system_prompt,
    )

    data = [(tc.task, tc.completion) for tc in completions]

    def on_complete():
        progress.update(run_name, advance=1)

    all_results = []
    total_successes = 0
    total_failures = 0

    for seed in generation_seeds:
        batch = await measure_post_task_stated_async(
            client=client,
            data=data,
            builder=builder,
            semaphore=semaphore,
            temperature=temperature,
            seed=seed,
            on_complete=on_complete,
        )

        for s in batch.successes:
            all_results.append({"task_id": s.task.id, "score": s.score, "seed": seed})
        total_successes += len(batch.successes)
        total_failures += len(batch.failures)

    return all_results, total_successes, total_failures


async def main(config_path: Path):
    config = load_config(config_path)

    templates = load_templates_from_yaml(config.templates)
    template = next(t for t in templates if t.name == config.template_name)

    console.print(f"[bold]Experiment: {config.experiment_name}")
    console.print(f"[bold]Config: {config_path}")
    console.print(f"  Completions path: {config.completions_path}")
    console.print(f"  Completion conditions: {config.completion_conditions}")
    console.print(f"  Rating model: {config.rating_model}")
    console.print(f"  Temperature: {config.temperature}")
    console.print(f"  Generation seeds: {config.generation_seeds}")
    console.print(f"  Measurement prompts: {list(config.measurement_system_prompts.keys())}")
    console.print()

    # Load completions
    console.print("[bold]Loading completions...")
    completion_sources, completion_system_prompts = load_completions(config.completions_path, config.completion_conditions)
    for name, comps in completion_sources.items():
        console.print(f"  {name}: {len(comps)} completions")

    # Filter out completions that reference the system prompt
    if config.filter_sysprompt_references:
        console.print("\n[bold]Filtering completions that reference system prompt...")
        for name in completion_sources:
            completion_sources[name] = await filter_sysprompt_references(
                completion_sources[name],
                completion_system_prompts[name],
                name,
                config.max_concurrent,
            )
        console.print()
    else:
        console.print()

    rating_client = get_client(config.rating_model)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    exp_store = ExperimentStore(config.experiment_name)

    # Build list of conditions to run
    conditions_to_run = []
    for comp_name, completions in completion_sources.items():
        for ctx_name, system_prompt in config.measurement_system_prompts.items():
            run_name = f"completion_{comp_name}_context_{ctx_name}"
            if not exp_store.exists("post_task_stated", run_name):
                conditions_to_run.append((run_name, completions, system_prompt, comp_name, ctx_name))

    if not conditions_to_run:
        console.print("[green]All conditions already complete!")
        return

    n_tasks = len(next(iter(completion_sources.values())))
    n_measurements_per_condition = n_tasks * len(config.generation_seeds)
    console.print(f"[bold]Running {len(conditions_to_run)} conditions")
    console.print(f"  {n_tasks} tasks × {len(config.generation_seeds)} samples = {n_measurements_per_condition} measurements per condition\n")

    results_summary: dict[str, dict] = {}

    with MultiExperimentProgress() as progress:
        for run_name, completions, _, _, _ in conditions_to_run:
            progress.add_experiment(run_name, total=len(completions) * len(config.generation_seeds))

        async def run_one(run_name: str, completions: list[TaskCompletion], system_prompt: str | None, comp_name: str, ctx_name: str):
            progress.set_status(run_name, "running...")

            results, successes, failures = await run_condition(
                completions, system_prompt, template, config.generation_seeds,
                rating_client, semaphore, progress, run_name, config.temperature
            )

            run_config = {
                "completions_path": str(config.completions_path),
                "rating_model": config.rating_model,
                "completion_source": comp_name,
                "measurement_context": ctx_name,
                "system_prompt": system_prompt,
                "template": template.name,
                "generation_seeds": config.generation_seeds,
                "temperature": config.temperature,
                "n_tasks": len(completions),
                "n_samples": len(config.generation_seeds),
                "n_results": len(results),
            }
            exp_store.save_stated("post_task_stated", run_name, results, run_config)

            status = f"[green]{successes}✓[/green] [red]{failures}✗[/red]"
            progress.complete(run_name, status=status)

            return run_name, {"successes": successes, "failures": failures, "total_runs": 1}

        # Run all conditions concurrently
        run_tasks = [run_one(*cond) for cond in conditions_to_run]
        completed = await asyncio.gather(*run_tasks)
        results_summary = dict(completed)

    print_summary(results_summary)
    console.print("\n[bold green]Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure post-task stated with system prompt ablations")
    parser.add_argument("config", type=Path, help="Path to config YAML")
    args = parser.parse_args()
    asyncio.run(main(args.config))
