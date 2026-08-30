"""Generate uniform-sample pairwise evaluation data.

Samples N pairs uniformly from the final-half eval tasks (matching the split
in train_ridge_heldout), measures them via the revealed-preference pipeline,
and saves as a standard run directory for use in probe evaluation.

Usage:
    python -m src.measurement.uniform_eval configs/uniform_eval/gemma3_27b.yaml \
        --eval-run-dir <eval_run_dir> \
        --exclude-run-dir <training_run_dir> \
        --eval-split-seed 42 \
        --n-pairs 500 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

import numpy as np

from src.measurement.elicitation import measure_pre_task_revealed_async
from src.measurement.runners.runners import build_revealed_builder
from src.measurement.runners.utils.experiment_utils import (
    apply_pair_order,
    build_configurations,
    setup_experiment,
)
from src.measurement.storage import MeasurementCache, ExperimentStore, model_short_name
from src.measurement.storage.base import build_measurement_config, load_yaml
from src.measurement.storage.loading import load_run_utilities
from src.probes.residualization import build_task_groups
from src.task_data import Task, load_filtered_tasks, parse_origins

DEFAULT_ORIGINS = ["wildchat", "alpaca", "math", "bailbench", "stress_test"]


def _load_task_ids_from_run(run_dir: Path) -> set[str]:
    raw = load_yaml(run_dir / "measurements.yaml")
    ids: set[str] = set()
    for m in raw:
        ids.add(m["task_a"])
        ids.add(m["task_b"])
    return ids


def _sample_random_pairs(
    tasks: list[Task], n_pairs: int, rng: np.random.Generator,
) -> list[tuple[Task, Task]]:
    n_tasks = len(tasks)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        batch_size = (n_pairs - len(pairs)) * 2
        idx_a = rng.integers(0, n_tasks, size=batch_size)
        idx_b = rng.integers(0, n_tasks, size=batch_size)
        for a, b in zip(idx_a, idx_b):
            if a != b:
                pair = (min(a, b), max(a, b))
                pairs.add(pair)
                if len(pairs) >= n_pairs:
                    break
    return [(tasks[a], tasks[b]) for a, b in sorted(pairs)]


def _load_eval_final_half_tasks(
    eval_run_dir: Path,
    exclude_run_dirs: list[Path],
    eval_split_seed: int,
    origins: list[str] = DEFAULT_ORIGINS,
) -> list[Task]:
    """Load tasks from the final-half of the eval split.

    Replicates the sweep/final split from train_ridge_heldout.
    """
    _, task_id_array = load_run_utilities(eval_run_dir)
    eval_task_ids = sorted(task_id_array if isinstance(task_id_array, list) else task_id_array.tolist())
    print(f"Eval tasks: {len(eval_task_ids)}")

    exclude_ids: set[str] = set()
    for run_dir in exclude_run_dirs:
        exclude_ids |= _load_task_ids_from_run(run_dir)
    eval_task_ids = [tid for tid in eval_task_ids if tid not in exclude_ids]
    if exclude_ids:
        print(f"After removing training overlap: {len(eval_task_ids)}")

    rng_split = np.random.default_rng(eval_split_seed)
    perm = rng_split.permutation(len(eval_task_ids))
    half = len(eval_task_ids) // 2
    final_ids = [eval_task_ids[i] for i in perm[half:]]
    print(f"Final-half eval tasks: {len(final_ids)}")

    tasks = load_filtered_tasks(
        n=len(final_ids),
        origins=parse_origins(origins),
        task_ids=set(final_ids),
        seed=None,
    )
    print(f"Loaded {len(tasks)} tasks from datasets")

    origin_counts = Counter(t.origin.name for t in tasks)
    for origin, count in sorted(origin_counts.items()):
        print(f"  {origin}: {count}")

    return tasks


def sample_from_eval_final_half(
    eval_run_dir: Path,
    exclude_run_dirs: list[Path],
    n_pairs: int,
    seed: int,
    eval_split_seed: int = 42,
    origins: list[str] = DEFAULT_ORIGINS,
    topics_json: Path | None = None,
    pairs_per_topic: int = 35,
) -> list[tuple[Task, Task]]:
    """Sample uniform pairs from the final-half eval tasks.

    If topics_json is provided, samples pairs_per_topic within-topic pairs
    per topic (for HOO evaluation) in addition to n_pairs random pairs.
    """
    tasks = _load_eval_final_half_tasks(
        eval_run_dir, exclude_run_dirs, eval_split_seed, origins,
    )
    rng = np.random.default_rng(seed)

    # Random pairs (test-set metric)
    pairs = _sample_random_pairs(tasks, n_pairs, rng)
    print(f"Sampled {len(pairs)} random pairs")

    # Within-topic pairs (HOO metric)
    if topics_json is not None:
        task_groups = build_task_groups(
            task_ids={t.id for t in tasks}, grouping="topic", topics_json=topics_json,
        )
        task_by_id = {t.id: t for t in tasks}
        topic_to_tasks: dict[str, list[Task]] = {}
        for tid, topic in task_groups.items():
            topic_to_tasks.setdefault(topic, []).append(task_by_id[tid])

        for topic in sorted(topic_to_tasks):
            topic_tasks = topic_to_tasks[topic]
            if len(topic_tasks) < 5:
                print(f"  Skipping {topic} ({len(topic_tasks)} tasks)")
                continue
            n = min(pairs_per_topic, len(topic_tasks) * (len(topic_tasks) - 1) // 2)
            topic_pairs = _sample_random_pairs(topic_tasks, n, rng)
            print(f"  {topic}: {len(topic_pairs)} within-topic pairs")
            pairs.extend(topic_pairs)

        print(f"Total: {len(pairs)} pairs (random + within-topic)")

    return pairs


async def run_uniform_eval(
    config_path: Path,
    pairs: list[tuple[Task, Task]],
    max_concurrent: int = 50,
) -> Path:
    task_lookup = {}
    for a, b in pairs:
        task_lookup[a.id] = a
        task_lookup[b.id] = b

    ctx = setup_experiment(config_path, expected_mode="pre_task_revealed")
    config = ctx.config

    ctx.tasks = list(task_lookup.values())
    ctx.task_lookup = task_lookup

    configurations = build_configurations(ctx, config, include_order=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    model_short = model_short_name(ctx.client.canonical_model_name)
    exp_store = ExperimentStore(config.experiment_id)

    print(f"\nMeasuring {len(pairs)} pairs x {config.n_samples} samples "
          f"x {len(configurations)} configurations")

    for cfg in configurations:
        seed_suffix = f"_seed{cfg.seed}" if cfg.seed is not None else ""
        run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_{cfg.order}{seed_suffix}_uniform_eval"

        if exp_store.exists("pre_task_revealed", run_name):
            print(f"  Skipping {run_name} (already exists)")
            continue

        cache = MeasurementCache(
            cfg.template, ctx.client, cfg.response_format, cfg.order,
            seed=cfg.seed, system_prompt=config.measurement_system_prompt,
        )
        ordered_pairs = apply_pair_order(
            pairs, cfg.order, config.pair_order_seed, config.include_reverse_order,
        )
        pairs_with_repeats = ordered_pairs * config.n_samples

        builder = build_revealed_builder(
            cfg.template, cfg.response_format, post_task=False,
            reasoning_mode=config.reasoning_mode,
            system_prompt=config.measurement_system_prompt,
        )

        async def measure_fn(pairs_to_query: list[tuple[Task, Task]]):
            return await measure_pre_task_revealed_async(
                client=ctx.client,
                pairs=pairs_to_query,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=cfg.seed,
            )

        measurements, batch_stats = await cache.get_or_measure_async(
            pairs_with_repeats, measure_fn, task_lookup,
        )
        print(f"  {run_name}: {len(measurements)} measurements "
              f"({batch_stats.cache_hits} cached, {batch_stats.api_successes} API)")

        if measurements:
            measurements_dicts = [
                {
                    "task_a": m.task_a.id,
                    "task_b": m.task_b.id,
                    "choice": m.choice,
                    "origin_a": m.task_a.origin.name,
                    "origin_b": m.task_b.origin.name,
                }
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
            run_dir = exp_store.save_revealed(
                "pre_task_revealed", run_name, measurements_dicts, config_dict,
            )
            print(f"  Saved to {run_dir}")

    return exp_store.base_dir / "pre_task_revealed"


def main():
    parser = argparse.ArgumentParser(
        description="Generate uniform-sample pairwise evaluation data",
    )
    parser.add_argument("config", type=Path, help="Measurement config YAML (pre_task_revealed)")
    parser.add_argument("--eval-run-dir", type=Path, required=True,
                        help="Eval run dir — samples pairs from its final-half tasks")
    parser.add_argument("--exclude-run-dir", type=Path, nargs="+", default=[],
                        help="Run dirs whose tasks to exclude (training run)")
    parser.add_argument("--eval-split-seed", type=int, default=42,
                        help="Seed for sweep/final split (must match probe config)")
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--origins", nargs="+", default=DEFAULT_ORIGINS)
    parser.add_argument("--topics-json", type=Path, default=None,
                        help="If set, also sample within-topic pairs for HOO eval")
    parser.add_argument("--pairs-per-topic", type=int, default=35)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    pairs = sample_from_eval_final_half(
        eval_run_dir=args.eval_run_dir,
        exclude_run_dirs=args.exclude_run_dir,
        n_pairs=args.n_pairs,
        seed=args.seed,
        eval_split_seed=args.eval_split_seed,
        origins=args.origins,
        topics_json=args.topics_json,
        pairs_per_topic=args.pairs_per_topic,
    )

    result_dir = asyncio.run(run_uniform_eval(
        config_path=args.config,
        pairs=pairs,
        max_concurrent=args.max_concurrent,
    ))
    print(f"\nDone. Results in: {result_dir}")


if __name__ == "__main__":
    main()
