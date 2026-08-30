"""Core extraction logic: batched and generation-based paths."""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.measurement.storage.completions import extract_completion_text
from src.models.base import COMPLETION_SELECTORS, split_selectors
from src.models.huggingface_model import HuggingFaceModel
from src.task_data import load_filtered_tasks, OriginDataset, Task
from src.types import Message

from .config import ExtractionConfig
from .metadata import ExtractionMetadata, ExtractionStats
from .persistence import (
    load_existing_data,
    save_activations,
    save_manifest,
    save_extraction_metadata,
)

def _gpu_mem_gb() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


def _load_model(config: ExtractionConfig) -> HuggingFaceModel:
    return HuggingFaceModel(
        config.model,
        max_new_tokens=config.max_new_tokens,
        subfolder=config.subfolder,
        device=config.device,
        max_memory=config.max_memory,
    )


def _load_task_ids_filter(task_ids_file: Path | None) -> set[str] | None:
    if task_ids_file is None:
        return None
    raw = task_ids_file.read_text().strip()
    if raw.startswith("{"):
        data = json.loads(raw)
        task_ids = set(data["task_ids"])
    else:
        task_ids = set(raw.splitlines())
    print(f"Filtering to {len(task_ids)} task IDs from {task_ids_file}")
    return task_ids


def _load_tasks(config: ExtractionConfig) -> list[Task]:
    task_ids_filter = _load_task_ids_filter(config.task_ids_file)

    if config.custom_tasks_file is not None:
        with open(config.custom_tasks_file) as f:
            custom_data = json.load(f)
        tasks = [
            Task(prompt=t["prompt"], origin=OriginDataset.SYNTHETIC, id=t["task_id"], metadata=t)
            for t in custom_data
        ]
        if task_ids_filter is not None:
            tasks = [t for t in tasks if t.id in task_ids_filter]
        print(f"Loaded {len(tasks)} custom tasks from {config.custom_tasks_file}")
        return tasks[:config.n_tasks] if config.n_tasks is not None else tasks

    task_origins = [OriginDataset[o.upper()] for o in config.task_origins]
    n = config.n_tasks if config.n_tasks is not None else len(task_ids_filter)
    return load_filtered_tasks(
        n=n,
        origins=task_origins,
        seed=config.seed,
        task_ids=task_ids_filter,
    )


def _build_messages(task_prompt: str, system_prompt: str | None, prompt_template: str | None = None) -> list[Message]:
    msgs: list[Message] = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    user_content = prompt_template.format(task=task_prompt) if prompt_template is not None else task_prompt
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _build_metadata(
    config: ExtractionConfig,
    resolved_layers: list[int],
    n_model_layers: int,
    n_existing: int,
    stats: ExtractionStats,
    source_completions: str | None = None,
) -> ExtractionMetadata:
    task_origin_names = [OriginDataset[o.upper()].name for o in config.task_origins] if config.task_origins else []
    return ExtractionMetadata(
        model=config.model,
        n_tasks=config.n_tasks or 0,
        task_origins=task_origin_names,
        layers_config=config.layers_to_extract,
        layers_resolved=resolved_layers,
        n_model_layers=n_model_layers,
        selectors=config.selectors,
        batch_size=config.batch_size,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        n_existing=n_existing,
        n_new=stats.n_new,
        n_failures=stats.n_failures,
        n_truncated=stats.n_truncated,
        n_ooms=stats.n_ooms,
        source_completions=source_completions,
        system_prompt=config.system_prompt,
    )


def run_extraction(config: ExtractionConfig) -> None:
    output_dir = config.resolved_output_dir
    needs_generation = bool(set(config.selectors) & COMPLETION_SELECTORS)

    print(f"Loading model: {config.model}...")
    model = _load_model(config)

    resolved_layers = [model.resolve_layer(layer) for layer in config.layers_to_extract]
    print(f"Layers: {config.layers_to_extract} -> {resolved_layers} ({model.n_layers} total)")
    print(f"Selectors: {config.selectors}")

    tasks = _load_tasks(config)

    point_selectors, span_selectors = split_selectors(config.selectors)

    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in point_selectors}
    span_activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in span_selectors}
    completions: list[dict] = []
    n_existing = 0

    if config.resume:
        task_ids, activations, completions = load_existing_data(output_dir, point_selectors)
        if span_selectors:
            _, span_activations, _ = load_existing_data(output_dir, span_selectors)
        n_existing = len(task_ids)
        tasks = [t for t in tasks if t.id not in set(task_ids)]
        print(f"Resume: found {n_existing} existing, {len(tasks)} remaining")

    if not tasks:
        print("No tasks remaining.")
        return

    print(f"{len(tasks)} tasks to process")

    if needs_generation:
        stats = generation_extraction(
            model=model, tasks=tasks, layers=resolved_layers,
            selectors=config.selectors, temperature=config.temperature,
            max_new_tokens=config.max_new_tokens, task_ids=task_ids,
            activations=activations, span_activations=span_activations,
            completions=completions,
            output_dir=output_dir, save_every=config.save_every,
            system_prompt=config.system_prompt,
            prompt_template=config.prompt_template,
        )
    else:
        task_lookup = {task.id: task for task in tasks}
        items: list[tuple[str, list[Message]]] = [
            (task.id, _build_messages(task.prompt, config.system_prompt, config.prompt_template))
            for task in tasks
        ]
        n_before = len(task_ids)
        stats = batched_extraction(
            model=model, items=items, layers=resolved_layers,
            selectors=config.selectors, batch_size=config.batch_size,
            task_ids=task_ids, activations=activations,
            span_activations=span_activations,
            output_dir=output_dir, save_every=config.save_every,
        )
        for tid in task_ids[n_before:]:
            task = task_lookup[tid]
            completions.append({
                "task_id": task.id,
                "task_prompt": task.prompt,
                "origin": task.origin.name,
            })

    if stats.n_new > 0:
        print(f"Saving {len(task_ids)} total activations...")
        save_activations(output_dir, task_ids, activations)
        if span_selectors:
            save_activations(output_dir, task_ids, span_activations, span=True)
        save_manifest(output_dir, completions)

    metadata = _build_metadata(config, resolved_layers, model.n_layers, n_existing, stats)
    save_extraction_metadata(output_dir, metadata)
    print(f"\nDone! {stats.n_new} new, {stats.n_failures} failures, {stats.n_ooms} OOMs")


def run_from_completions(
    config: ExtractionConfig,
    completions_path: Path,
    model: HuggingFaceModel | None = None,
) -> None:
    output_dir = config.resolved_output_dir

    if model is None:
        print(f"Loading model: {config.model}...")
        model = _load_model(config)

    resolved_layers = [model.resolve_layer(layer) for layer in config.layers_to_extract]
    print(f"Layers: {config.layers_to_extract} -> {resolved_layers} ({model.n_layers} total)")
    print(f"Selectors: {config.selectors}")

    with open(completions_path) as f:
        completions_data: list[dict] = json.load(f)

    point_selectors, span_selectors = split_selectors(config.selectors)

    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in point_selectors}
    span_activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in span_selectors}
    n_existing = 0

    if config.resume:
        task_ids, activations, _ = load_existing_data(output_dir, point_selectors)
        if span_selectors:
            _, span_activations, _ = load_existing_data(output_dir, span_selectors)
        n_existing = len(task_ids)
        completions_data = [c for c in completions_data if c["task_id"] not in set(task_ids)]
        print(f"Resume: found {n_existing} existing, {len(completions_data)} remaining")

    if not completions_data:
        print("No completions remaining.")
        return

    items: list[tuple[str, list[Message]]] = []
    for c in completions_data:
        if "messages" in c:
            items.append((c["task_id"], c["messages"]))
        else:
            msgs = _build_messages(c["task_prompt"], config.system_prompt, config.prompt_template) + [
                {"role": "assistant", "content": c["completion"]},
            ]
            items.append((c["task_id"], msgs))

    stats = batched_extraction(
        model=model, items=items, layers=resolved_layers,
        selectors=config.selectors, batch_size=config.batch_size,
        task_ids=task_ids, activations=activations,
        span_activations=span_activations,
        output_dir=output_dir, save_every=config.save_every,
    )

    if stats.n_new > 0:
        save_activations(output_dir, task_ids, activations)
        if span_selectors:
            save_activations(output_dir, task_ids, span_activations, span=True)

    metadata = _build_metadata(
        config, resolved_layers, model.n_layers, n_existing, stats,
        source_completions=str(completions_path),
    )
    save_extraction_metadata(output_dir, metadata)
    print(f"\nDone! Extracted {stats.n_new} new, {stats.n_failures} failures")


def batched_extraction(
    model: HuggingFaceModel,
    items: list[tuple[str, list[Message]]],
    layers: list[int],
    selectors: list[str],
    batch_size: int,
    task_ids: list[str],
    activations: dict[str, dict[int, list[np.ndarray]]],
    output_dir: Path,
    save_every: int,
    span_activations: dict[str, dict[int, list[np.ndarray]]] | None = None,
) -> ExtractionStats:
    """Batched forward-pass extraction with OOM retry via recursive halving.

    Mutates task_ids, activations, and span_activations in place.
    """
    point_selectors, span_selectors = split_selectors(selectors)
    if span_activations is None:
        span_activations = {s: defaultdict(list) for s in span_selectors}
    stats = ExtractionStats()

    # LOCAL PATCH (vd): length-sort so batches pad to similar lengths — the
    # unsorted order made every batch pay its longest member's padding cost
    # (~10s/batch); sorted, most batches are short. Safe under --resume:
    # task_ids travel with their rows.
    items = sorted(items, key=lambda it: sum(len(m.get("content", "") or "") for m in it[1]))

    alloc, res = _gpu_mem_gb()
    print(f"Batched extraction: {len(items)} items, batch_size={batch_size} (GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved)")

    for batch_start in tqdm(range(0, len(items), batch_size), desc="Batches"):
        batch = items[batch_start:batch_start + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_messages = [item[1] for item in batch]

        succeeded_ids, per_sample_point, per_sample_span, batch_ooms, batch_fails = _extract_batch_with_oom_retry(
            model, batch_ids, batch_messages, layers, selectors,
        )

        stats.n_ooms += batch_ooms
        stats.n_failures += batch_fails

        for i, task_id in enumerate(succeeded_ids):
            task_ids.append(task_id)
            for selector in point_selectors:
                for layer, layer_acts in per_sample_point[selector].items():
                    activations[selector][layer].append(layer_acts[i])
            for selector in span_selectors:
                for layer, layer_acts_list in per_sample_span[selector].items():
                    span_activations[selector][layer].append(layer_acts_list[i])
            stats.n_new += 1

        gc.collect()
        torch.cuda.empty_cache()

        batch_num = batch_start // batch_size + 1
        if batch_num % 10 == 0:
            alloc, res = _gpu_mem_gb()
            tqdm.write(f"[batch {batch_num}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res | OOMs: {stats.n_ooms}")

        if stats.n_new > 0 and stats.n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, activations)
            if span_activations:
                save_activations(output_dir, task_ids, span_activations, span=True)

    return stats


def _extract_batch_with_oom_retry(
    model: HuggingFaceModel,
    ids: list[str],
    messages_batch: list[list[Message]],
    layers: list[int],
    selectors: list[str],
) -> tuple[list[str], dict[str, dict[int, np.ndarray]], dict[str, dict[int, list[np.ndarray]]], int, int]:
    """Try full batch; on OOM, split in half and retry each half.

    Single-sample OOM is absorbed (empty result for that sample).
    Returns: (succeeded_ids, point_results, span_results, n_ooms, n_failures)
    """
    point_selectors, span_selectors = split_selectors(selectors)
    try:
        result = model.get_activations_batch(
            messages_batch, layers=layers, selector_names=selectors,
        )
        return ids, result.point, result.span, 0, 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()

        if len(ids) == 1:
            tqdm.write(f"OOM on single sample {ids[0]}, skipping")
            empty_point: dict[str, dict[int, np.ndarray]] = {s: {} for s in point_selectors}
            empty_span: dict[str, dict[int, list[np.ndarray]]] = {s: {} for s in span_selectors}
            return [], empty_point, empty_span, 1, 1

        mid = len(ids) // 2
        tqdm.write(f"OOM on batch of {len(ids)}, splitting into {mid} + {len(ids) - mid}")

        ids_a, point_a, span_a, ooms_a, fails_a = _extract_batch_with_oom_retry(
            model, ids[:mid], messages_batch[:mid], layers, selectors,
        )
        ids_b, point_b, span_b, ooms_b, fails_b = _extract_batch_with_oom_retry(
            model, ids[mid:], messages_batch[mid:], layers, selectors,
        )

        merged_ids = ids_a + ids_b

        # Merge point results (concat along batch axis)
        merged_point: dict[str, dict[int, np.ndarray]] = {s: {} for s in point_selectors}
        for selector in point_selectors:
            for layer in layers:
                parts = []
                if layer in point_a.get(selector, {}):
                    parts.append(point_a[selector][layer])
                if layer in point_b.get(selector, {}):
                    parts.append(point_b[selector][layer])
                if parts:
                    merged_point[selector][layer] = np.concatenate(parts, axis=0)

        # Merge span results (concat lists)
        merged_span: dict[str, dict[int, list[np.ndarray]]] = {s: {} for s in span_selectors}
        for selector in span_selectors:
            for layer in layers:
                list_a = span_a.get(selector, {}).get(layer, [])
                list_b = span_b.get(selector, {}).get(layer, [])
                if list_a or list_b:
                    merged_span[selector][layer] = list_a + list_b

        return merged_ids, merged_point, merged_span, ooms_a + ooms_b, fails_a + fails_b


def generation_extraction(
    model: HuggingFaceModel,
    tasks: list[Task],
    layers: list[int],
    selectors: list[str],
    temperature: float,
    max_new_tokens: int,
    task_ids: list[str],
    activations: dict[str, dict[int, list[np.ndarray]]],
    completions: list[dict],
    output_dir: Path,
    save_every: int,
    system_prompt: str | None = None,
    prompt_template: str | None = None,
    span_activations: dict[str, dict[int, list[np.ndarray]]] | None = None,
) -> ExtractionStats:
    """Sequential generate-then-extract. Mutates task_ids/activations/completions in place."""
    point_selectors, span_selectors = split_selectors(selectors)
    if span_activations is None:
        span_activations = {s: defaultdict(list) for s in span_selectors}
    stats = ExtractionStats()
    failures: list[tuple[str, str]] = []

    alloc, res = _gpu_mem_gb()
    print(f"Generation extraction: {len(tasks)} tasks (GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved)")

    for i, task in enumerate(tqdm(tasks, desc="Tasks")):
        for attempt in range(2):
            try:
                messages = _build_messages(task.prompt, system_prompt, prompt_template)
                result = model.generate_with_activations(
                    messages, layers=layers, selector_names=selectors, temperature=temperature,
                )
                truncated = result.completion_tokens >= max_new_tokens

                if truncated:
                    stats.n_truncated += 1

                task_ids.append(task.id)
                for selector in point_selectors:
                    for layer, act in result.activations[selector].items():
                        activations[selector][layer].append(act)
                for selector in span_selectors:
                    for layer, act in result.activations.span[selector].items():
                        span_activations[selector][layer].append(act)

                completions.append({
                    "task_id": task.id,
                    "task_prompt": task.prompt,
                    "origin": task.origin.name,
                    "completion": extract_completion_text(result.completion),
                    "truncated": truncated,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                })
                stats.n_new += 1
                break

            except torch.cuda.OutOfMemoryError as e:
                stats.n_ooms += 1
                tqdm.write(f"OOM on task {task.id} (attempt {attempt + 1}/2): {e}")
                torch.cuda.empty_cache()
                if attempt == 1:
                    failures.append((task.id, f"OOM after retry: {e}"))
                    stats.n_failures += 1
            except Exception as e:
                failures.append((task.id, str(e)))
                stats.n_failures += 1
                break

        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0:
            alloc, res = _gpu_mem_gb()
            tqdm.write(f"[{i+1}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res | OOMs: {stats.n_ooms}")

        if stats.n_new > 0 and stats.n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, activations)
            if span_activations:
                save_activations(output_dir, task_ids, span_activations, span=True)
            save_manifest(output_dir, completions)

    if failures:
        print(f"First few failures: {failures[:3]}")

    return stats
