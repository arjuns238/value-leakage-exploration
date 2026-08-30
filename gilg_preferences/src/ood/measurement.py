"""Pairwise measurement with caching for OOD experiments.

Each measurement is a (condition_id, task_a, task_b) triple.
Results are cached in pairwise.json so we never re-measure the same comparison.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.models import OpenAICompatibleClient, get_client
from src.ood.config import OODMeasurementConfig
from src.ood.prompts import OODPromptSet
from src.task_data import load_filtered_tasks, OriginDataset
from src.task_data.task import Task
from src.measurement.elicitation.prompt_templates import (
    PreTaskRevealedPromptBuilder,
    PromptTemplate,
    load_templates_from_yaml,
)
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation.measure import measure_pre_task_revealed_async
from src.types import MeasurementBatch, BinaryPreferenceMeasurement

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]

console = Console()


# --- Loading helpers ---


def load_triples(config: OODMeasurementConfig) -> list[dict]:
    with open(Path(config.mappings_dir) / config.mapping) as f:
        mapping = json.load(f)
    return mapping["pairs"]


def load_system_prompts(config: OODMeasurementConfig) -> dict[str, str]:
    prompts_dir = Path(config.prompts_dir)
    system_prompts: dict[str, str] = {}
    for prompts_file in config.prompts:
        prompt_set = OODPromptSet.load(prompts_dir / prompts_file)
        system_prompts["baseline"] = prompt_set.baseline_prompt
        for c in prompt_set.conditions:
            system_prompts[c.condition_id] = c.system_prompt
    return system_prompts


def _load_custom_tasks(json_path: Path) -> dict[str, Task]:
    with open(json_path) as f:
        data = json.load(f)
    return {
        t["task_id"]: Task(prompt=t["prompt"], origin=OriginDataset.SYNTHETIC, id=t["task_id"], metadata=t)
        for t in data
    }


def _collect_task_ids(triples: list[dict]) -> set[str]:
    ids = set()
    for t in triples:
        ids.add(t["task_a"])
        ids.add(t["task_b"])
    return ids


def load_tasks(config: OODMeasurementConfig, triples: list[dict]) -> dict[str, Task]:
    needed_ids = _collect_task_ids(triples)
    if not config.custom_tasks:
        tasks = load_filtered_tasks(n=len(needed_ids), origins=ALL_ORIGINS, task_ids=needed_ids)
        return {t.id: t for t in tasks}
    custom = _load_custom_tasks(Path(config.tasks_dir) / config.custom_tasks)
    anchor_ids = needed_ids - set(custom.keys())
    standard = load_filtered_tasks(n=len(anchor_ids), origins=ALL_ORIGINS, task_ids=anchor_ids)
    return {**custom, **{t.id: t for t in standard}}


def _output_path(config: OODMeasurementConfig) -> Path:
    config_name = config.mapping.removesuffix(".json")
    return Path(config.output_dir) / config_name / "pairwise.json"


# --- Caching ---


def _canonicalize(task_a: str, task_b: str) -> tuple[str, str]:
    """Ensure task_a < task_b lexicographically."""
    if task_a <= task_b:
        return task_a, task_b
    return task_b, task_a


def _cache_key(condition_id: str, task_a: str, task_b: str) -> tuple[str, str, str]:
    a, b = _canonicalize(task_a, task_b)
    return (condition_id, a, b)


def load_pairwise(path: Path) -> dict:
    """Load pairwise.json, returning empty structure if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"model": "", "baseline_prompt": "", "results": []}


def _build_cache(results: list[dict]) -> dict[tuple[str, str, str], dict]:
    cache = {}
    for entry in results:
        key = _cache_key(entry["condition_id"], entry["task_a"], entry["task_b"])
        cache[key] = entry
    return cache


def _save_results(existing: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)


# --- Template ---


def _make_template(config: OODMeasurementConfig) -> PromptTemplate:
    templates = load_templates_from_yaml(config.template_file)
    matches = [t for t in templates if t.name == config.template_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 template named '{config.template_name}' "
            f"in {config.template_file}, found {len(matches)}"
        )
    return matches[0]


# --- Measurement ---


def _aggregate_condition_results(
    batch: MeasurementBatch[BinaryPreferenceMeasurement],
    pairs_with_meta: list[tuple],
    cond_triples: list[dict],
    cid: str,
) -> list[dict]:
    """Aggregate raw measurements into per-triple win counts."""
    success_queue: dict[tuple[str, str], list] = defaultdict(list)
    for m in batch.successes:
        success_queue[(m.task_a.id, m.task_b.id)].append(m)

    triple_wins: dict[tuple[str, str], int] = defaultdict(int)
    triple_total: dict[tuple[str, str], int] = defaultdict(int)
    triple_refusals: dict[tuple[str, str], int] = defaultdict(int)

    for api_task_a, api_task_b, canon_a, canon_b, a_is_first in pairs_with_meta:
        queue = success_queue[(api_task_a.id, api_task_b.id)]
        if not queue:
            continue
        m = queue.pop(0)

        triple_key = (canon_a, canon_b)
        if m.choice == "refusal":
            triple_refusals[triple_key] += 1
        else:
            triple_total[triple_key] += 1
            chose_canon_a = a_is_first == (m.choice == "a")
            if chose_canon_a:
                triple_wins[triple_key] += 1

    results = []
    for triple in cond_triples:
        ta, tb = _canonicalize(triple["task_a"], triple["task_b"])
        tkey = (ta, tb)
        n_a = triple_wins[tkey]
        n_total = triple_total[tkey]
        n_b = n_total - n_a
        results.append({
            "condition_id": cid,
            "task_a": ta,
            "task_b": tb,
            "n_a": n_a,
            "n_b": n_b,
            "n_refusals": triple_refusals[tkey],
            "n_total": n_total + triple_refusals[tkey],
        })
    return results


async def _measure_condition(
    client: OpenAICompatibleClient,
    cid: str,
    cond_triples: list[dict],
    system_prompt: str,
    tasks: dict[str, Task],
    template: PromptTemplate,
    temperature: float,
    n_repeats: int,
    semaphore: asyncio.Semaphore,
    on_complete: Callable[[], None] | None = None,
    async_client: "AsyncOpenAI | None" = None,
) -> tuple[list[dict], MeasurementBatch[BinaryPreferenceMeasurement]]:
    """Measure all triples for a single condition. Returns (aggregated results, raw batch)."""
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=template,
        system_prompt=system_prompt,
    )

    # Build pairs with metadata: half A-first, half B-first
    pairs_with_meta = []
    rng = np.random.RandomState(hash(cid) % (2**31))

    for triple in cond_triples:
        ta_id, tb_id = _canonicalize(triple["task_a"], triple["task_b"])
        task_a = tasks[ta_id]
        task_b = tasks[tb_id]

        n_a_first = n_repeats // 2
        n_b_first = n_repeats - n_a_first

        for _ in range(n_a_first):
            pairs_with_meta.append((task_a, task_b, ta_id, tb_id, True))
        for _ in range(n_b_first):
            pairs_with_meta.append((task_b, task_a, ta_id, tb_id, False))

    rng.shuffle(pairs_with_meta)

    api_pairs = [(a, b) for a, b, _, _, _ in pairs_with_meta]

    batch = await measure_pre_task_revealed_async(
        client=client,
        pairs=api_pairs,
        builder=builder,
        semaphore=semaphore,
        temperature=temperature,
        seed=None,
        on_complete=on_complete,
        async_client=async_client,
    )

    return _aggregate_condition_results(batch, pairs_with_meta, cond_triples, cid), batch


async def _measure_all_conditions(
    client: OpenAICompatibleClient,
    uncached: list[dict],
    system_prompts: dict[str, str],
    tasks: dict[str, Task],
    template: PromptTemplate,
    config: OODMeasurementConfig,
    existing: dict,
    output_path: Path,
    n_cached: int,
    progress: Progress,
    task_id: int,
) -> None:
    """Measure all uncached triples, saving after each condition completes."""
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for t in uncached:
        by_condition[t["condition_id"]].append(t)

    semaphore = asyncio.Semaphore(config.max_concurrent)
    async_client = client._make_async_client()
    successes = 0
    failures = 0

    def _update_status():
        status = f"[green]{successes}✓[/green] [red]{failures}✗[/red]"
        if n_cached:
            status += f" [cyan]{n_cached}⚡[/cyan]"
        progress.update(task_id, status=status)

    def on_complete():
        progress.advance(task_id)

    async def measure_and_save(cid: str, cond_triples: list[dict]) -> None:
        nonlocal successes, failures
        results, batch = await _measure_condition(
            client, cid, cond_triples, system_prompts[cid], tasks,
            template, config.temperature, config.n_repeats, semaphore, on_complete,
            async_client=async_client,
        )
        successes += len(batch.successes)
        failures += len(batch.failures)
        _update_status()
        existing["results"].extend(results)
        _save_results(existing, output_path)
        console.log(f"Saved {cid} ({len(results)} triples, {len(batch.successes)}✓ {len(batch.failures)}✗)")

    # Run conditions concurrently but cap in-flight conditions to limit memory
    condition_sem = asyncio.Semaphore(3)

    async def gated(cid: str, cond_triples: list[dict]) -> None:
        async with condition_sem:
            await measure_and_save(cid, cond_triples)

    _update_status()
    try:
        await asyncio.gather(*[
            gated(cid, cond_triples)
            for cid, cond_triples in by_condition.items()
        ])
    finally:
        await async_client.close()


def _measure_pairs(
    client: OpenAICompatibleClient,
    config: OODMeasurementConfig,
    triples: list[dict],
    system_prompts: dict[str, str],
    tasks: dict[str, Task],
    output_path: Path,
) -> None:
    """Measure pairwise comparisons with caching and progress display."""
    template = _make_template(config)

    existing = load_pairwise(output_path)
    cache = _build_cache(existing["results"])

    uncached = [t for t in triples if _cache_key(t["condition_id"], t["task_a"], t["task_b"]) not in cache]

    total_api_calls = len(uncached) * config.n_repeats
    n_cached_triples = len(triples) - len(uncached)
    console.print(
        f"Triples: {len(triples)} total, {n_cached_triples} cached, "
        f"{len(uncached)} to measure ({total_api_calls} API calls)"
    )

    if not uncached:
        return

    existing["model"] = config.model
    existing["baseline_prompt"] = system_prompts["baseline"]

    n_cached = len(triples) - len(uncached)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )

    with progress:
        task_id = progress.add_task("API calls", total=total_api_calls, status="")

        asyncio.run(_measure_all_conditions(
            client, uncached, system_prompts, tasks, template, config,
            existing, output_path, n_cached, progress, task_id,
        ))

    console.print(f"Saved {len(existing['results'])} total results to {output_path}")


# --- Entry point ---


def run_config(config: OODMeasurementConfig) -> None:
    """Load all data from config and run measurement."""
    config_name = config.mapping.removesuffix(".json")
    console.print(f"\n[bold]{'=' * 60}")
    console.print(f"[bold]Experiment: {config_name}")

    triples = load_triples(config)
    system_prompts = load_system_prompts(config)
    tasks = load_tasks(config, triples)
    output_path = _output_path(config)
    client = get_client(config.model, max_new_tokens=config.max_new_tokens)

    _measure_pairs(
        client=client,
        config=config,
        triples=triples,
        system_prompts=system_prompts,
        tasks=tasks,
        output_path=output_path,
    )
