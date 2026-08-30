"""Unified entry point for running experiments.

Usage:
  # Single config
  python -m src.measurement.runners.run config.yaml
  python -m src.measurement.runners.run config.yaml --max-concurrent 100

  # Override model from CLI
  python -m src.measurement.runners.run config.yaml --model qwen3-32b

  # Multiple models from same config
  python -m src.measurement.runners.run config.yaml --model llama-3.3-70b qwen3-32b gemma-2-27b

  # Override mode (e.g., generate completions first, then run measurements)
  python -m src.measurement.runners.run config.yaml --mode completion_generation
  python -m src.measurement.runners.run config.yaml --mode post_task_stated

  # Multiple configs in parallel
  python -m src.measurement.runners.run config1.yaml config2.yaml --max-concurrent 50
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import yaml

from dotenv import load_dotenv

load_dotenv()

from src.measurement.runners.config import load_experiment_config, set_experiment_id, get_experiment_id
from src.measurement.runners.open_ended_config import load_open_ended_config
from src.measurement.runners.runners import RUNNERS, RunnerStats
from src.measurement.runners.progress import (
    MultiExperimentProgress,
    print_summary,
    console,
)

DEFAULT_MAX_CONCURRENT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preference measurement experiments")
    parser.add_argument("configs", nargs="+", type=Path, help="Config file(s) to run")
    parser.add_argument("--model", nargs="+", type=str, default=None,
                        help="Override model(s) - runs config once per model specified")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Override measurement_system_prompt (e.g., '/no_think' for Qwen3)")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help=f"Max concurrent API requests (default: {DEFAULT_MAX_CONCURRENT})")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment ID for tracking (auto-generated if not provided)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Override preference_mode (e.g., completion_generation, post_task_stated)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted active learning run from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="List experiments without running")
    parser.add_argument("--debug", action="store_true", help="Show example errors for each failure category")
    return parser.parse_args()


async def run_experiments(
    config_paths: list[Path],
    semaphore: asyncio.Semaphore,
    experiment_id: str | None = None,
    model_overrides: list[str] | None = None,
    system_prompt_override: str | None = None,
    mode_override: str | None = None,
    resume: bool = False,
) -> dict[str, dict | Exception]:
    """Run experiments with concurrent progress display.

    If model_overrides is provided, runs each config once per model.
    """
    # Build list of (path, config_overrides, label) tuples
    configs = []
    for path in config_paths:
        # Try to load as OpenEndedMeasurementConfig first, fall back to ExperimentConfig
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if data.get("preference_mode") == "open_ended":
                base_config = load_open_ended_config(path)
            else:
                base_config = load_experiment_config(path)
        except Exception:
            base_config = load_experiment_config(path)

        # Determine which models to run
        models = model_overrides if model_overrides else [base_config.model]

        for model in models:
            overrides: dict = {}
            if model != base_config.model:
                overrides["model"] = model
            if system_prompt_override is not None:
                overrides["measurement_system_prompt"] = system_prompt_override

            # CLI experiment_id overrides config file
            if experiment_id is not None:
                overrides["experiment_id"] = experiment_id

            if resume:
                overrides["resume"] = True

            # CLI mode overrides config file
            effective_mode = base_config.preference_mode
            if mode_override is not None:
                overrides["preference_mode"] = mode_override
                effective_mode = mode_override

            label = f"{path.stem}:{model}"
            configs.append((path, overrides if overrides else None, label, base_config, effective_mode))

    results: dict[str, dict | Exception] = {}

    with MultiExperimentProgress() as progress:
        # Add all experiments to progress display
        for path, overrides, label, base_config, effective_mode in configs:
            # Estimate total based on config type
            if hasattr(base_config, "response_formats"):
                # Standard ExperimentConfig
                n_configs = len(base_config.response_formats) * len(base_config.generation_seeds)
                if base_config.n_template_samples:
                    n_configs = base_config.n_template_samples
                # Post-task experiments iterate over completion seeds
                if base_config.preference_mode.startswith("post_task"):
                    completion_seeds = base_config.completion_seeds or base_config.generation_seeds
                    n_configs *= len(completion_seeds)
            else:
                # OpenEndedMeasurementConfig
                n_configs = len(base_config.prompt_variants) * len(base_config.rating_seeds)
            progress.add_experiment(label, total=n_configs)

        async def run_one(path: Path, overrides: dict | None, label: str, base_config, effective_mode: str) -> tuple[str, dict | Exception]:
            runner = RUNNERS.get(effective_mode)

            if runner is None:
                progress.complete(label, status="[red]no runner")
                return label, ValueError(f"No runner for mode: {effective_mode}")

            progress.set_status(label, "running...")
            last_update_time: list[float | None] = [None]

            def on_progress(stats: RunnerStats):
                now = time.time()
                if last_update_time[0] is None:
                    iter_str = ""
                else:
                    iter_time = now - last_update_time[0]
                    iter_str = f" [dim]{iter_time:.1f}s[/dim]"
                last_update_time[0] = now
                status = f"[green]{stats.successes}✓[/green] [red]{stats.failures}✗[/red]"
                if stats.cache_hits:
                    status += f" [cyan]{stats.cache_hits}⚡[/cyan]"
                # Active learning iteration info
                if stats.iteration is not None:
                    status += f" [magenta]iter {stats.iteration}[/magenta]"
                    if stats.iteration_pairs:
                        status += f"[dim]/{stats.iteration_pairs}p[/dim]"
                    if stats.chunk is not None and stats.total_chunks is not None:
                        status += f" [blue]chunk {stats.chunk}/{stats.total_chunks}[/blue]"
                    if stats.rank_correlation is not None:
                        status += f" [yellow]r={stats.rank_correlation:.3f}[/yellow]"
                    if stats.total_comparisons:
                        status += f" [dim]{stats.total_comparisons}cmp[/dim]"
                status += iter_str
                progress.progress.update(progress.tasks[label], completed=stats.completed, total=stats.total_runs, status=status)

            try:
                result = await runner(path, semaphore, progress_callback=on_progress, config_overrides=overrides)
                cache_hits = result.get('cache_hits', 0)
                status = f"[green]{result['successes']}✓[/green] [red]{result['failures']}✗[/red]"
                if cache_hits:
                    status += f" [cyan]{cache_hits}⚡[/cyan]"
                progress.complete(label, status=status)
                return label, result
            except Exception as e:
                progress.complete(label, status=f"[red]error: {e}")
                return label, e

        # Run all experiments concurrently
        tasks = [run_one(path, overrides, label, base_config, effective_mode) for path, overrides, label, base_config, effective_mode in configs]
        completed = await asyncio.gather(*tasks)
        results = dict(completed)

    return results


def main():
    args = parse_args()

    # Validate configs
    for config_path in args.configs:
        if not config_path.exists():
            console.print(f"[red]Error: Config not found: {config_path}")
            return 1

    if args.dry_run:
        console.print("[bold]Experiments to run:")
        for config_path in args.configs:
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f)
                if data.get("preference_mode") == "open_ended":
                    config = load_open_ended_config(config_path)
                else:
                    config = load_experiment_config(config_path)
            except Exception:
                config = load_experiment_config(config_path)

            models = args.model if args.model else [config.model]
            for model in models:
                suffix = f" (system: {args.system_prompt})" if args.system_prompt else ""
                console.print(f"  • {config_path.stem}: {model}{suffix}")
        return 0

    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Set experiment ID for this run (auto-generates timestamp if not provided)
    exp_id = set_experiment_id(args.experiment_id)
    console.print(f"[bold]Experiment ID: {exp_id}")

    n_runs = len(args.configs) * (len(args.model) if args.model else 1)
    console.print(f"[bold]Running {n_runs} experiment(s) with max {args.max_concurrent} concurrent requests\n")

    results = asyncio.run(run_experiments(
        args.configs, semaphore, experiment_id=exp_id,
        model_overrides=args.model, system_prompt_override=args.system_prompt,
        mode_override=args.mode, resume=args.resume,
    ))
    print_summary(results, debug=args.debug)

    # Return non-zero if any failures
    has_errors = any(isinstance(r, Exception) for r in results.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    exit(main())
