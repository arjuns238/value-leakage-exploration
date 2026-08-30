"""Analyze completions where post-task ratings changed vs neutral baseline.

Runs semantic parser on completions with large rating deltas to check for
sysprompt references and sentiment.

Usage:
    python -m src.experiments.sysprompt_variation.analyze_rating_changes results/experiments/sysprompt_comp_alpaca_prompts
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
load_dotenv()

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from src.measurement.elicitation.semantic_parser import (
    parse_sysprompt_effect_async,
    SyspromptEffectResult,
)


def load_measurements(measurements_dir: Path) -> dict[str, dict[str, float]]:
    """Load measurements from post_task_stated subdirectories.

    Returns: {sysprompt_name: {task_id: score}}
    """
    results = {}
    for subdir in measurements_dir.iterdir():
        if not subdir.is_dir():
            continue
        # Parse sysprompt name from directory name like "completion_neutral_context_neutral"
        # Format: completion_{sysprompt}_context_neutral
        parts = subdir.name.split("_context_")
        if len(parts) != 2:
            continue
        sysprompt_name = parts[0].replace("completion_", "")

        measurements_file = subdir / "measurements.yaml"
        if not measurements_file.exists():
            continue

        with open(measurements_file) as f:
            measurements = yaml.safe_load(f)

        results[sysprompt_name] = {m["task_id"]: m["score"] for m in measurements}

    return results


def load_completions(completions_dir: Path) -> tuple[dict[str, dict[str, dict]], dict[str, str | None]]:
    """Load completions and system prompts from completions subdirectories.

    Returns:
        completions: {sysprompt_name: {task_id: {task_prompt, completion, origin}}}
        system_prompts: {sysprompt_name: system_prompt_text}
    """
    completions = {}
    system_prompts = {}

    for subdir in completions_dir.iterdir():
        if not subdir.is_dir():
            continue

        completions_file = subdir / "completions.json"
        config_file = subdir / "config.json"
        if not completions_file.exists() or not config_file.exists():
            continue

        with open(completions_file) as f:
            completion_list = json.load(f)
        with open(config_file) as f:
            config = json.load(f)

        completions[subdir.name] = {c["task_id"]: c for c in completion_list}
        system_prompts[subdir.name] = config.get("system_prompt")

    return completions, system_prompts


def compute_deltas(
    measurements: dict[str, dict[str, float]],
    baseline_name: str = "neutral",
) -> dict[str, dict[str, float]]:
    """Compute rating deltas vs baseline for each sysprompt.

    Returns: {sysprompt_name: {task_id: delta}}
    """
    baseline = measurements[baseline_name]
    deltas = {}

    for sysprompt_name, scores in measurements.items():
        if sysprompt_name == baseline_name:
            continue
        deltas[sysprompt_name] = {
            task_id: score - baseline[task_id]
            for task_id, score in scores.items()
            if task_id in baseline
        }

    return deltas


def filter_changed_tasks(
    deltas: dict[str, dict[str, float]],
    threshold: float = 1.0,
) -> list[tuple[str, str, float]]:
    """Filter to tasks with large rating changes, sorted by |delta| descending.

    Returns: [(sysprompt_name, task_id, delta), ...]
    """
    changed = []
    for sysprompt_name, task_deltas in deltas.items():
        for task_id, delta in task_deltas.items():
            if abs(delta) >= threshold:
                changed.append((sysprompt_name, task_id, delta))
    # Sort by |delta| descending to prioritize largest changes
    return sorted(changed, key=lambda x: abs(x[2]), reverse=True)


async def analyze_completions(
    changed_tasks: list[tuple[str, str, float]],
    completions: dict[str, dict[str, dict]],
    system_prompts: dict[str, str | None],
    max_concurrent: int = 20,
) -> list[dict]:
    """Run semantic parser on completions with changed ratings."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Analyzing completions", total=len(changed_tasks))

        async def analyze_one(sysprompt_name: str, task_id: str, delta: float) -> dict | None:
            completion_data = completions[sysprompt_name][task_id]
            sysprompt = system_prompts[sysprompt_name]

            try:
                async with semaphore:
                    judgment = await parse_sysprompt_effect_async(
                        sysprompt,
                        completion_data["task_prompt"],
                        completion_data["completion"],
                    )
            except Exception as e:
                progress.update(task, advance=1)
                print(f"\nSkipping {task_id}: {type(e).__name__}")
                return None

            progress.update(task, advance=1)

            return {
                "sysprompt_name": sysprompt_name,
                "task_id": task_id,
                "task_prompt": completion_data["task_prompt"],
                "completion": completion_data["completion"],
                "origin": completion_data["origin"],
                "rating_delta": delta,
                "sysprompt_reference": judgment.sysprompt_reference,
                "sentiment": judgment.sentiment,
                "refusal": judgment.refusal,
            }

        all_results = await asyncio.gather(*[
            analyze_one(sp, tid, delta)
            for sp, tid, delta in changed_tasks
        ])
        results = [r for r in all_results if r is not None]

    return list(results)


def print_summary(results: list[dict]) -> None:
    """Print correlation summary."""
    if not results:
        print("No tasks with rating changes found.")
        return

    # Group by sysprompt
    by_sysprompt = defaultdict(list)
    for r in results:
        by_sysprompt[r["sysprompt_name"]].append(r)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(results)} completions with rating changes")
    print(f"{'='*80}")

    for sysprompt_name, items in sorted(by_sysprompt.items()):
        n = len(items)
        avg_delta = sum(r["rating_delta"] for r in items) / n
        avg_sentiment = sum(r["sentiment"] for r in items) / n
        ref_rate = sum(1 for r in items if r["sysprompt_reference"]) / n
        refusal_rate = sum(1 for r in items if r["refusal"]) / n

        print(f"\n{sysprompt_name} (n={n}):")
        print(f"  avg rating delta: {avg_delta:+.2f}")
        print(f"  avg sentiment:    {avg_sentiment:+.2f}")
        print(f"  sysprompt ref:    {ref_rate:.1%}")
        print(f"  refusal rate:     {refusal_rate:.1%}")


def print_examples(results: list[dict], n_examples: int = 3) -> None:
    """Print example completions with largest deltas."""
    if not results:
        return

    sorted_results = sorted(results, key=lambda r: abs(r["rating_delta"]), reverse=True)

    print(f"\n{'='*80}")
    print(f"TOP {n_examples} EXAMPLES BY RATING DELTA")
    print(f"{'='*80}")

    for r in sorted_results[:n_examples]:
        print(f"\n[{r['sysprompt_name']}] delta={r['rating_delta']:+.1f}")
        print(f"task: {r['task_prompt'][:100]}...")
        print(f"sentiment={r['sentiment']:.2f} ref={r['sysprompt_reference']} refusal={r['refusal']}")
        print(f"{'-'*40}")
        print(r["completion"][:500] + ("..." if len(r["completion"]) > 500 else ""))


def main():
    parser = argparse.ArgumentParser(description="Analyze completions with rating changes")
    parser.add_argument("experiment_dir", type=Path, help="Path to experiment directory")
    parser.add_argument("--threshold", type=float, default=1.0, help="Min |delta| to include")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent API calls")
    parser.add_argument("--n-examples", type=int, default=5, help="Number of examples to print")
    parser.add_argument("--limit", type=int, help="Max tasks to analyze (for testing)")
    parser.add_argument("--output", "-o", type=Path, help="Save results to JSON file")
    parser.add_argument("--sysprompt", type=str, help="Filter to specific sysprompt condition")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    measurements_dir = experiment_dir / "post_task_stated"
    completions_dir = experiment_dir / "completions"

    print(f"Loading data from {experiment_dir}")
    measurements = load_measurements(measurements_dir)
    completions, system_prompts = load_completions(completions_dir)

    print(f"Loaded {len(measurements)} sysprompt conditions")
    print(f"Loaded {len(completions)} completion sets")

    deltas = compute_deltas(measurements)
    if args.sysprompt:
        deltas = {k: v for k, v in deltas.items() if k == args.sysprompt}
    changed_tasks = filter_changed_tasks(deltas, args.threshold)
    print(f"Found {len(changed_tasks)} tasks with |delta| >= {args.threshold}")

    if args.limit:
        changed_tasks = changed_tasks[:args.limit]
        print(f"Limited to {len(changed_tasks)} tasks")

    if not changed_tasks:
        print("No tasks to analyze.")
        return

    results = asyncio.run(analyze_completions(
        changed_tasks, completions, system_prompts, args.max_concurrent
    ))

    print_summary(results)
    print_examples(results, args.n_examples)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
