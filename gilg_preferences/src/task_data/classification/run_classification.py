"""Run topic classification on tasks.

Usage:
    python -m src.task_data.classification.run_classification \
        --origins wildchat alpaca math bailbench stress_test --n-tasks 2000

    python -m src.task_data.classification.run_classification \
        --origins stress_test --n-tasks 50000 \
        --categories-file data/topics/categories.json \
        --filter-topics knowledge_qa coding math \
        --output data/topics/stresstest_reclassified.json

Pass --harm-override to run the second-pass harm-intent check.
Pass --filter-topics to only reclassify tasks currently in those topics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.task_data import load_filtered_tasks, parse_origins
from src.task_data.classification.classify import (
    MODELS,
    Cache,
    TaskInput,
    apply_harm_overrides,
    classify_tasks_batch,
    discover_categories,
    load_cache,
    save_cache,
)

load_dotenv()

DISCOVERY_SAMPLE_SIZE = 300


def load_tasks_from_origins(
    origins: list[str],
    n_tasks: int,
    seed: int,
) -> list[TaskInput]:
    origin_enums = parse_origins(origins)
    tasks = load_filtered_tasks(
        n=n_tasks, origins=origin_enums, seed=seed, stratified=True,
    )
    return [{"task_id": t.id, "prompt": t.prompt} for t in tasks]


def filter_tasks_by_current_topic(
    tasks: list[TaskInput],
    topics_path: Path,
    filter_topics: list[str],
) -> list[TaskInput]:
    """Keep only tasks whose current primary topic is in filter_topics."""
    with open(topics_path) as f:
        topics_cache = json.load(f)

    filtered = []
    for task in tasks:
        tid = task["task_id"]
        if tid not in topics_cache:
            continue
        primary = next(iter(topics_cache[tid].values()))["primary"]
        if primary in filter_topics:
            filtered.append(task)
    print(f"Filtered to {len(filtered)} tasks currently in {filter_topics}")
    return filtered


async def run_discovery(tasks: list[TaskInput], n_sample: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    sample_size = min(n_sample, len(tasks))
    indices = rng.choice(len(tasks), size=sample_size, replace=False)
    sample_prompts = [tasks[i]["prompt"] for i in indices]

    print(f"\nDiscovering categories from {sample_size} sampled tasks...")
    categories = await discover_categories(sample_prompts)
    print(f"Discovered {len(categories)} categories:")
    for cat in categories:
        print(f"  - {cat}")

    return categories


async def run_classification(
    tasks: list[TaskInput],
    categories: list[str],
    cache_path: Path,
    max_concurrent: int,
    category_descriptions: dict[str, str] | None = None,
) -> Cache:
    cache = load_cache(cache_path)
    print(f"\nClassifying {len(tasks)} tasks into {len(categories)} categories...")

    cache = await classify_tasks_batch(
        tasks, categories, cache, max_concurrent, category_descriptions,
    )
    save_cache(cache, cache_path)
    print(f"Cache saved to {cache_path}")

    return cache


def print_distribution(cache: Cache, task_ids: set[str]) -> None:
    relevant = {tid: entry for tid, entry in cache.items() if tid in task_ids}

    for model in MODELS:
        entries_with_model = {
            tid: e[model] for tid, e in relevant.items() if model in e
        }
        if not entries_with_model:
            continue
        counts = Counter(e["primary"] for e in entries_with_model.values())
        print(f"\n{model} distribution ({len(entries_with_model)} tasks):")
        for cat, count in counts.most_common():
            pct = count / len(entries_with_model) * 100
            print(f"  {cat:<30} {count:>5} ({pct:5.1f}%)")

    missing = task_ids - set(relevant.keys())
    if missing:
        print(f"  {'(unclassified)':<30} {len(missing):>5}")


async def main():
    parser = argparse.ArgumentParser(description="Run topic classification on tasks")
    parser.add_argument("--origins", nargs="+", required=True, help="Task origins to load")
    parser.add_argument("--n-tasks", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discovery-sample-size", type=int, default=DISCOVERY_SAMPLE_SIZE)
    parser.add_argument("--max-concurrent", type=int, default=60)
    parser.add_argument("--categories-file", type=str, default=None,
                        help="Path to JSON file with pre-defined categories (skip discovery)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for classification cache JSON")
    parser.add_argument("--harm-override", action="store_true",
                        help="Run second-pass harm-intent override")
    parser.add_argument("--filter-topics", nargs="+", default=None,
                        help="Only reclassify tasks currently in these topics")
    parser.add_argument("--existing-topics", type=str, default=None,
                        help="Path to existing topics.json (required with --filter-topics)")
    args = parser.parse_args()

    if args.filter_topics and not args.existing_topics:
        parser.error("--filter-topics requires --existing-topics")

    # Load tasks
    tasks = load_tasks_from_origins(args.origins, args.n_tasks, args.seed)
    print(f"Loaded {len(tasks)} tasks")

    # Filter to specific current topics if requested
    if args.filter_topics:
        tasks = filter_tasks_by_current_topic(
            tasks, Path(args.existing_topics), args.filter_topics,
        )

    task_ids = {t["task_id"] for t in tasks}
    output_path = Path(args.output)

    # Discover or load categories
    category_descriptions: dict[str, str] | None = None
    if args.categories_file:
        with open(args.categories_file) as f:
            cat_data = json.load(f)
        if isinstance(cat_data, list):
            categories = cat_data
        else:
            categories = cat_data["categories"]
            category_descriptions = cat_data.get("descriptions")
        print(f"Loaded {len(categories)} categories from {args.categories_file}")
    else:
        categories = await run_discovery(tasks, args.discovery_sample_size, args.seed)

    # Classify
    cache = await run_classification(
        tasks, categories, output_path, args.max_concurrent, category_descriptions,
    )

    # Harm override
    if args.harm_override:
        print("\nRunning harm-intent override pass...")
        cache = await apply_harm_overrides(tasks, cache, max_concurrent=args.max_concurrent)
        save_cache(cache, output_path)
        print(f"Updated cache saved to {output_path}")

    print_distribution(cache, task_ids)


if __name__ == "__main__":
    asyncio.run(main())
