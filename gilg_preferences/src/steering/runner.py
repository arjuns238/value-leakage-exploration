"""Config-driven steering experiment runner.

Consolidates boilerplate from isolated steering scripts: pair loading,
checkpoint management, ordering sign correction, response parsing, and
progress logging. Experiments are defined by YAML configs.

Three condition types:
- PostHocCondition: clean prefill → modify K+V cache at multiple layers.
- HookCondition: three prefills (clean + steered) + combine caches. Isolation.
- DifferentialCondition: single prefill with per-span steering coefficients.
  Naive differential (has cross-contamination). Used as a control.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers.cache_utils import DynamicCache

from src.measurement.elicitation.completion_judge import (
    extract_claimed_task,
    judge_completion_full_async,
)
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import compose_hooks, position_selective_steering
from src.steering.kv_cache import combine_caches, modify_cache_kv_at_positions, project_to_kv_space
from src.steering.tokenization import find_pairwise_task_spans
from src.task_data import OriginDataset, Task


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PostHocCondition:
    name: str
    probe: str                      # e.g. "ridge_L25"
    kv_layers: tuple[int, int]      # (start, end) inclusive
    multipliers: list[float]
    normalize_per_layer: bool = False  # scale coefficient by each layer's KV norm
    recompute_modes: list[bool] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.recompute_modes is None:
            self.recompute_modes = [False]


@dataclass
class HookCondition:
    name: str
    probe_prefix: str               # e.g. "ridge_L" — layer number appended
    layers: list[int]               # iterated individually, each gets own probe
    multipliers: list[float]
    ref_mult: float                 # reference multiplier for interpolation
    # Which suffix modes to run. [False] = no recompute, [True] = recompute,
    # [False, True] = both (shared prefills, separate condition names with
    # "_recompute" suffix for True).
    recompute_modes: list[bool] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.recompute_modes is None:
            self.recompute_modes = [False]


@dataclass
class DifferentialCondition:
    """Naive differential steering with per-span coefficients, single forward pass.

    spans maps "first"/"second" to coefficient fractions of mean_norm.
    Default {"first": 1, "second": -1} gives standard differential.
    Exactly one of probe_prefix or probe must be set.
    """
    name: str
    layers: list[int]               # steer layers, iterated individually
    multipliers: list[float]
    probe_prefix: str = ""          # e.g. "ridge_L" — layer number appended
    probe: str = ""                 # e.g. "ridge_L25" — fixed for all layers
    spans: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if bool(self.probe_prefix) == bool(self.probe):
            raise ValueError("DifferentialCondition needs exactly one of probe or probe_prefix")
        if self.spans is None:
            self.spans = {"first": 1, "second": -1}
        if not self.spans:
            raise ValueError("DifferentialCondition.spans must not be empty")
        invalid = set(self.spans.keys()) - {"first", "second"}
        if invalid:
            raise ValueError(f"DifferentialCondition.spans has invalid keys: {invalid}")

    def probe_id(self, layer: int) -> str:
        if self.probe:
            return self.probe
        return f"{self.probe_prefix}{layer}"


@dataclass
class RunConfig:
    model: str
    max_new_tokens: int
    pairs_path: Path
    probe_manifest: Path
    checkpoint_path: Path
    mean_norm: float | dict[int, float]  # scalar applied to all layers, or per-layer dict
    n_trials: int
    temperature: float
    seed: int
    n_pairs: int | None             # None = use all pairs
    template_path: str
    conditions: list[PostHocCondition | HookCondition | DifferentialCondition]
    system_prompt: str | None = None
    device: str = "cuda"
    max_memory: dict[int, str] | None = None
    # "batched_cache" (default, fast, requires uniform full-attention) stacks per-multiplier KV
    # caches and generates them in one batch. "hook_per_call" calls generate() once per
    # (pair, ordering, mult) with the position-selective hook installed for the entire
    # forward pass — slower but compatible with hybrid-attention models like Qwen3.5-MoE,
    # whose linear-attention layers don't expose .keys/.values for cache stacking.
    generation_mode: str = "batched_cache"


def _parse_recompute_modes(raw: dict) -> list[bool]:
    raw_recompute = raw.get("recompute_suffix", False)
    if isinstance(raw_recompute, bool):
        return [raw_recompute]
    if isinstance(raw_recompute, list):
        return raw_recompute
    raise ValueError(f"recompute_suffix must be bool or list[bool], got {raw_recompute!r}")


def _parse_condition(raw: dict) -> PostHocCondition | HookCondition | DifferentialCondition:
    cache_injection = raw["cache_injection"]
    if cache_injection == "post_hoc":
        kv_start, kv_end = raw["kv_layers"]
        return PostHocCondition(
            name=raw["name"],
            probe=raw["probe"],
            kv_layers=(kv_start, kv_end),
            multipliers=raw["multipliers"],
            normalize_per_layer=raw.get("normalize_per_layer", False),
            recompute_modes=_parse_recompute_modes(raw),
        )
    if cache_injection == "hook":
        recompute_modes = _parse_recompute_modes(raw)
        return HookCondition(
            name=raw["name"],
            probe_prefix=raw["probe_prefix"],
            layers=raw["layers"],
            multipliers=raw["multipliers"],
            ref_mult=raw["ref_mult"],
            recompute_modes=recompute_modes,
        )
    if cache_injection == "differential":
        raw_spans = raw.get("spans")
        if raw_spans is None:
            spans = {"first": 1, "second": -1}
        elif isinstance(raw_spans, dict):
            spans = {k: float(v) for k, v in raw_spans.items()}
        else:
            raise ValueError(f"spans must be a dict, got {type(raw_spans).__name__}")
        return DifferentialCondition(
            name=raw["name"],
            layers=raw["layers"],
            multipliers=raw.get("multipliers", [1]),
            probe_prefix=raw.get("probe_prefix", ""),
            probe=raw.get("probe", ""),
            spans=spans,
        )
    raise ValueError(f"Unknown cache_injection: {cache_injection!r}")


def _parse_mean_norm(raw_value) -> float | dict[int, float]:
    if isinstance(raw_value, dict):
        return {int(k): float(v) for k, v in raw_value.items()}
    return float(raw_value)


def _validate_mean_norm_coverage(mean_norm, conditions) -> None:
    if not isinstance(mean_norm, dict):
        return
    for cond in conditions:
        layers = getattr(cond, "layers", None)
        if layers is None:
            continue
        missing = [L for L in layers if L not in mean_norm]
        if missing:
            raise KeyError(
                f"mean_norm dict missing entries for condition '{cond.name}' "
                f"injection layers: {missing}. Provided keys: {sorted(mean_norm)}"
            )


def load_config(config_path: Path) -> RunConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    conditions = [_parse_condition(c) for c in raw["conditions"]]
    mean_norm = _parse_mean_norm(raw["mean_norm"])
    _validate_mean_norm_coverage(mean_norm, conditions)
    return RunConfig(
        model=raw["model"],
        max_new_tokens=raw["max_new_tokens"],
        pairs_path=Path(raw["pairs_path"]),
        probe_manifest=Path(raw["probe_manifest"]),
        checkpoint_path=Path(raw["checkpoint_path"]),
        mean_norm=mean_norm,
        n_trials=raw["n_trials"],
        temperature=raw["temperature"],
        seed=raw["seed"],
        n_pairs=raw.get("n_pairs"),
        template_path=raw["template_path"],
        conditions=conditions,
        system_prompt=raw.get("system_prompt"),
        device=raw.get("device", "cuda"),
        max_memory={int(k): v for k, v in raw.get("max_memory", {}).items()} if raw.get("max_memory") else None,
        generation_mode=raw.get("generation_mode", "batched_cache"),
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _effective_coef(coef: float, ordering: int) -> float:
    """Negate when ordering=1 to maintain direction in original task space."""
    return coef if ordering == 0 else -coef


# -----------------------------------------------------------------------------
# CANONICAL CONTRASTIVE-STEERING PLOTTING FRAME (use this unless you have a
# specific reason to deviate).
# -----------------------------------------------------------------------------
#
# Setup recap. The differential runner applies opposite-sign steering to the
# two task spans. With spans {first: 1, second: -1} plus _effective_coef's
# ordering flip, the semantic → physical mapping is CONSTANT across orderings:
#
#                        | task_a gets        | task_b gets        |
#   ordering=0 (a first) | +signed_multiplier | -signed_multiplier |
#   ordering=1 (b first) | +signed_multiplier | -signed_multiplier |
#
# In our pair convention (task_a = higher-utility), the probe's "+" direction
# is always applied to the higher-utility task. Across ±signed_multiplier,
# both tasks receive both signs over a sweep.
#
# CANONICAL FRAME:
#   y = P(chose steered task | responded)
#   x = coefficient applied to the steered task (= ±signed_multiplier)
#
# How to compute it: each trial contributes TWO data points, one per task as
# "the steered task":
#     - point A: applied_c = +signed_multiplier,
#                chose_steered = (choice_original == 'a')
#     - point B: applied_c = -signed_multiplier,
#                chose_steered = (choice_original == 'b')
# Bucket by applied_c and average. Normalize by the number of RESPONDED trials
# in each bucket (drop rows where choice_original is neither 'a' nor 'b').
# Pair the curve with a refusal-rate band at the bottom so the dropped trials
# stay visible.
#
# Why this frame:
#   - It's symmetric by construction. A valid probe-causal effect produces a
#     monotone curve through (0, 0.5) that sums to 1 at ±c. Asymmetry shows up
#     only via the refusal band, not inside the main line.
#   - It generalises directly to single-task (panel B of the harm-breakdown
#     figure): there's only one steered task per trial, so the two frames
#     collapse to one. Panels A and B share axes.
#   - The swing |y(+c) - y(-c)| is the right headline number for "how much
#     does the probe direction control choice?".
#
# Alternatives (use only when justified):
#   - P(chose higher-utility task) vs signed_multiplier: equivalent swing, but
#     asymmetric around y=0.5 and doesn't generalise to single-task. Used in
#     earlier §2.3 aggregate claims; keep for historical continuity, but don't
#     re-use in new plots.
#   - P(chose steered task) without the "| responded" conditioning: same as
#     canonical except refusals implicitly count as "wrong side", breaking the
#     1-sum at non-trivial refusal rates. Avoid.
# -----------------------------------------------------------------------------


def _norm_for_layer(mean_norm: float | dict[int, float], layer: int) -> float:
    if isinstance(mean_norm, dict):
        return mean_norm[layer]
    return mean_norm


def _remap_choice(choice_presented: str, ordering: int) -> str:
    """Map presented choice back to original task ordering."""
    if choice_presented in ("a", "b") and ordering == 1:
        return "b" if choice_presented == "a" else "a"
    return choice_presented


def _generate_and_record(
    *,
    hf_model: HuggingFaceModel,
    cache: DynamicCache,
    input_ids: torch.Tensor,
    second_span_end: int,
    recompute: bool,
    config: RunConfig,
    response_format,
    pair: dict,
    multiplier: float,
    layer: int,
    condition: str,
    existing_n: int,
    needed: int,
    ordering: int,
    checkpoint_counts: dict[tuple, int],
    stats: dict,
    norm_at_layer: float | None = None,
) -> list[dict]:
    """Optionally recompute suffix, generate, parse, checkpoint. Returns rows."""
    if recompute:
        cache = _clone_cache(cache)
        cache = hf_model.recompute_suffix(cache, input_ids, second_span_end)

    responses = hf_model.generate_from_cache(
        cache, input_ids, temperature=config.temperature, num_return_sequences=needed,
    )

    rows = []
    for sample_idx, response in enumerate(responses):
        rows.append(_make_row(
            pair=pair, multiplier=multiplier,
            layer=layer, condition=condition,
            sample_idx=existing_n + sample_idx, ordering=ordering,
            choice_presented=response_format.extract_label(response),
            raw_response=response,
            norm_at_layer=norm_at_layer,
        ))

    _append_checkpoint(config.checkpoint_path, rows)
    stats["generated"] += len(rows)
    return rows


def _make_row(
    *,
    pair: dict,
    multiplier: float,
    layer: int,
    condition: str,
    sample_idx: int,
    ordering: int,
    choice_presented: str,
    raw_response: str,
    norm_at_layer: float | None = None,
) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "task_a_id": pair["task_a"],
        "task_b_id": pair["task_b"],
        "signed_multiplier": multiplier,
        "layer": layer,
        "norm_at_layer": norm_at_layer,
        "condition": condition,
        "sample_idx": sample_idx,
        "ordering": ordering,
        "choice_original": _remap_choice(choice_presented, ordering),
        "choice_presented": choice_presented,
        "raw_response": raw_response,
        "delta_mu": pair["delta_mu"],
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> dict[tuple, int]:
    counts: dict[tuple, int] = defaultdict(int)
    if not path.exists():
        return counts
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            key = (row["pair_id"], row["layer"], row["signed_multiplier"], row["condition"], row["ordering"])
            counts[key] += 1
    return counts


def _append_checkpoint(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Pair / task utilities
# ---------------------------------------------------------------------------

def _load_pairs(config: RunConfig) -> list[dict]:
    with open(config.pairs_path) as f:
        all_pairs = json.load(f)
    if config.n_pairs is not None:
        random.seed(config.seed)
        return random.sample(all_pairs, config.n_pairs)
    return all_pairs


def _pair_to_tasks(pair: dict) -> tuple[Task, Task]:
    task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset.ALPACA, id=pair["task_a"], metadata={})
    task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset.ALPACA, id=pair["task_b"], metadata={})
    return task_a, task_b


def _compute_kv_norms(
    hf_model: HuggingFaceModel,
    pairs: list[dict],
    builder,
    layers: list[int],
) -> dict[int, float]:
    """Compute mean KV cache norm at each layer from a sample of pairs."""
    norms_by_layer: dict[int, list[float]] = {l: [] for l in layers}
    for pair in pairs:
        task_a, task_b = _pair_to_tasks(pair)
        prompt_data = builder.build(task_a, task_b)
        cache, _ = hf_model.prefill_with_hooks(prompt_data.messages, [])
        for layer in layers:
            k_norm = cache.layers[layer].keys.float().norm().item()
            v_norm = cache.layers[layer].values.float().norm().item()
            norms_by_layer[layer].append((k_norm + v_norm) / 2)
    return {l: float(np.mean(ns)) for l, ns in norms_by_layer.items()}


def _prepare_pair(
    builder, response_format, hf_model: HuggingFaceModel,
    first_task: Task, second_task: Task,
) -> tuple[list, tuple[int, int], tuple[int, int]] | None:
    """Build prompt, set response format, find task spans. Returns None on span failure."""
    prompt_data = builder.build(first_task, second_task)
    messages = prompt_data.messages
    formatted = hf_model.format_messages(messages, add_generation_prompt=True)
    try:
        first_span, second_span = find_pairwise_task_spans(
            hf_model.tokenizer, formatted,
            first_task.prompt, second_task.prompt, "Task A", "Task B",
        )
    except ValueError:
        return None
    return messages, first_span, second_span


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _clone_cache(cache: DynamicCache) -> DynamicCache:
    cloned = DynamicCache()
    for layer_idx in range(len(cache)):
        layer = cache.layers[layer_idx]
        cloned.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return cloned


def _compute_cache_delta(
    combined_ref: DynamicCache,
    cache_clean: DynamicCache,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    deltas = []
    for li in range(len(cache_clean)):
        dk = combined_ref.layers[li].keys - cache_clean.layers[li].keys
        dv = combined_ref.layers[li].values - cache_clean.layers[li].values
        deltas.append((dk, dv))
    return deltas


def _build_interpolated_cache(
    cache_clean: DynamicCache,
    deltas: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> DynamicCache:
    """Build clean + scale * delta cache (full sequence length)."""
    cache = DynamicCache()
    for li in range(len(cache_clean)):
        k = (cache_clean.layers[li].keys + scale * deltas[li][0]).clone()
        v = (cache_clean.layers[li].values + scale * deltas[li][1]).clone()
        cache.update(k, v, li)
    return cache


def _batch_generate(
    hf_model: HuggingFaceModel,
    caches: list[DynamicCache],
    input_ids: torch.Tensor,
    trials_per_cache: list[int],
    temperature: float,
) -> list[list[str]]:
    """Stack caches, expand each by trial count, generate all at once.

    Returns list-of-lists: responses[i] is the list of completions for caches[i].
    """
    seq_len = input_ids.shape[1]
    total_batch = sum(trials_per_cache)

    batch_cache = DynamicCache()
    for li in range(len(caches[0])):
        k_parts = []
        v_parts = []
        for cache, n in zip(caches, trials_per_cache):
            k = cache.layers[li].keys[:, :, :seq_len - 1, :]
            v = cache.layers[li].values[:, :, :seq_len - 1, :]
            k_parts.append(k.expand(n, -1, -1, -1).contiguous())
            v_parts.append(v.expand(n, -1, -1, -1).contiguous())
        batch_cache.update(torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0), li)

    expanded_ids = input_ids.expand(total_batch, -1).contiguous()
    gen_kwargs = hf_model._build_gen_kwargs(temperature, None, num_return_sequences=1)
    gen_kwargs["past_key_values"] = batch_cache

    output_ids = hf_model.model.generate(expanded_ids, **gen_kwargs)
    flat = hf_model._decode_completions(output_ids, seq_len, total_batch)

    # Split into per-cache groups
    result = []
    offset = 0
    for n in trials_per_cache:
        result.append(flat[offset:offset + n])
        offset += n
    return result


def _batch_generate_from_interpolated_caches(
    hf_model: HuggingFaceModel,
    cache_clean: DynamicCache,
    deltas: list[tuple[torch.Tensor, torch.Tensor]],
    input_ids: torch.Tensor,
    scales: list[float],
    trials_per_scale: list[int],
    temperature: float,
) -> list[str]:
    """Build interpolated caches via clean + scale * delta, then batch generate."""
    seq_len = input_ids.shape[1]
    interp_caches = []
    for scale in scales:
        c = DynamicCache()
        for li in range(len(cache_clean)):
            k = cache_clean.layers[li].keys[:, :, :seq_len - 1, :] + scale * deltas[li][0][:, :, :seq_len - 1, :]
            v = cache_clean.layers[li].values[:, :, :seq_len - 1, :] + scale * deltas[li][1][:, :, :seq_len - 1, :]
            c.update(k, v, li)
        interp_caches.append(c)

    groups = _batch_generate(hf_model, interp_caches, input_ids, trials_per_scale, temperature)
    # Flatten for backward compatibility (this function predates the grouped return)
    return [r for group in groups for r in group]


# ---------------------------------------------------------------------------
# Post-hoc condition runner
# ---------------------------------------------------------------------------

def _run_post_hoc_condition(
    condition: PostHocCondition,
    hf_model: HuggingFaceModel,
    pairs: list[dict],
    builder,
    response_format,
    config: RunConfig,
    checkpoint_counts: dict[tuple, int],
    stats: dict,
) -> None:
    kv_layer_range = list(range(condition.kv_layers[0], condition.kv_layers[1] + 1))

    # Load probe and project to K/V space for all kv layers
    _, direction = load_probe_direction(config.probe_manifest, condition.probe)
    kv_dirs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in kv_layer_range:
        kv_dirs[layer] = project_to_kv_space(hf_model.model, layer, direction)

    # Per-layer norm scaling: compute mean KV norm at each layer from first N pairs
    if condition.normalize_per_layer:
        kv_norms: dict[int, float] = _compute_kv_norms(hf_model, pairs[:20], builder, kv_layer_range)
        print(f"  Per-layer KV norms: L{kv_layer_range[0]}={kv_norms[kv_layer_range[0]]:.0f} ... "
              f"L{kv_layer_range[-1]}={kv_norms[kv_layer_range[-1]]:.0f}")
    else:
        kv_norms = None

    mode_names = [_condition_name(condition.name, m) for m in condition.recompute_modes]
    print(f"\nCondition: {condition.name} "
          f"(layers {condition.kv_layers[0]}-{condition.kv_layers[1]}, probe={condition.probe},"
          f" normalize_per_layer={condition.normalize_per_layer}, modes={mode_names})")

    for pair_idx, pair in enumerate(pairs):
        task_a, task_b = _pair_to_tasks(pair)
        pair_id = pair["pair_id"]

        for ordering in [0, 1]:
            first_task = task_a if ordering == 0 else task_b
            second_task = task_b if ordering == 0 else task_a

            # Check if any (multiplier, mode) needs work
            any_needed = False
            for recompute in condition.recompute_modes:
                cond_name = _condition_name(condition.name, recompute)
                for mult in condition.multipliers:
                    key = (pair_id, -1, mult, cond_name, ordering)
                    if checkpoint_counts[key] < config.n_trials:
                        any_needed = True
                        break
                if any_needed:
                    break
            if not any_needed:
                stats["skipped"] += len(condition.multipliers) * config.n_trials * len(condition.recompute_modes)
                continue

            prepared = _prepare_pair(builder, response_format, hf_model, first_task, second_task)
            if prepared is None:
                continue
            messages, first_span, second_span = prepared

            # Prefill once per (pair, ordering)
            base_cache, input_ids = hf_model.prefill_with_hooks(messages, [])

            for mult in condition.multipliers:
                # Build modified cache once per multiplier (shared across recompute modes)
                cache_modified = _clone_cache(base_cache)
                for layer in kv_layer_range:
                    norm = kv_norms[layer] if kv_norms else _norm_for_layer(config.mean_norm, layer)
                    effective = _effective_coef(norm * mult, ordering)
                    k_dir, v_dir = kv_dirs[layer]
                    modify_cache_kv_at_positions(cache_modified, layer, first_span[0], first_span[1], k_dir, v_dir, +effective)
                    modify_cache_kv_at_positions(cache_modified, layer, second_span[0], second_span[1], k_dir, v_dir, -effective)

                for recompute in condition.recompute_modes:
                    cond_name = _condition_name(condition.name, recompute)
                    key = (pair_id, -1, mult, cond_name, ordering)
                    existing_n = checkpoint_counts[key]
                    if existing_n >= config.n_trials:
                        stats["skipped"] += config.n_trials
                        continue

                    _generate_and_record(
                        hf_model=hf_model, cache=cache_modified, input_ids=input_ids,
                        second_span_end=second_span[1], recompute=recompute, config=config,
                        response_format=response_format, pair=pair, multiplier=mult,
                        layer=-1, condition=cond_name, existing_n=existing_n,
                        needed=config.n_trials - existing_n, ordering=ordering,
                        checkpoint_counts=checkpoint_counts, stats=stats,
                    )

        if (pair_idx + 1) % 10 == 0:
            _log_progress(pair_idx, len(pairs), stats)


# ---------------------------------------------------------------------------
# Hook condition runner
# ---------------------------------------------------------------------------

def _condition_name(base: str, recompute: bool) -> str:
    return f"{base}_recompute" if recompute else base


def _run_hook_condition(
    condition: HookCondition,
    hf_model: HuggingFaceModel,
    pairs: list[dict],
    builder,
    response_format,
    config: RunConfig,
    checkpoint_counts: dict[tuple, int],
    stats: dict,
) -> None:
    mode_names = [_condition_name(condition.name, m) for m in condition.recompute_modes]
    print(f"\nCondition: {condition.name} "
          f"(layers={condition.layers}, ref_mult={condition.ref_mult}, "
          f"modes={mode_names})")

    for layer in condition.layers:
        probe_id = f"{condition.probe_prefix}{layer}"
        _, direction = load_probe_direction(config.probe_manifest, probe_id)
        layer_norm = _norm_for_layer(config.mean_norm, layer)
        ref_coef = layer_norm * condition.ref_mult
        ref_tensor = torch.tensor(direction * ref_coef, dtype=torch.bfloat16, device="cuda")

        print(f"  Layer {layer}")

        for pair_idx, pair in enumerate(pairs):
            task_a, task_b = _pair_to_tasks(pair)
            pair_id = pair["pair_id"]

            for ordering in [0, 1]:
                first_task = task_a if ordering == 0 else task_b
                second_task = task_b if ordering == 0 else task_a

                # Collect needed multipliers across all recompute modes
                needs_by_mode: dict[bool, list[tuple[float, int, int]]] = {}
                for recompute in condition.recompute_modes:
                    cond_name = _condition_name(condition.name, recompute)
                    entries = []
                    for mult in condition.multipliers:
                        key = (pair_id, layer, mult, cond_name, ordering)
                        existing_n = checkpoint_counts[key]
                        needed = config.n_trials - existing_n
                        if needed > 0:
                            entries.append((mult, needed, existing_n))
                        else:
                            stats["skipped"] += config.n_trials
                    if entries:
                        needs_by_mode[recompute] = entries

                if not needs_by_mode:
                    continue

                prepared = _prepare_pair(builder, response_format, hf_model, first_task, second_task)
                if prepared is None:
                    continue
                messages, first_span, second_span = prepared

                # 3 shared prefills for all modes
                cache_clean, input_ids = hf_model.prefill_with_hooks(messages, [])
                cache_pos, _ = hf_model.prefill_with_hooks(
                    messages,
                    [(layer, position_selective_steering(ref_tensor, first_span[0], first_span[1]))],
                )
                cache_neg, _ = hf_model.prefill_with_hooks(
                    messages,
                    [(layer, position_selective_steering(-ref_tensor, second_span[0], second_span[1]))],
                )

                combined_ref = combine_caches(cache_clean, [
                    (cache_pos, first_span[0], first_span[1]),
                    (cache_neg, second_span[0], second_span[1]),
                ])
                del cache_pos, cache_neg
                deltas = _compute_cache_delta(combined_ref, cache_clean)
                del combined_ref

                # Generate for each mode, sharing the prefills + deltas
                for recompute, batch_entries in needs_by_mode.items():
                    cond_name = _condition_name(condition.name, recompute)

                    for mult, n_needed, existing_count in batch_entries:
                        scale = _effective_coef(mult, ordering) / condition.ref_mult
                        cache = _build_interpolated_cache(cache_clean, deltas, scale)
                        _generate_and_record(
                            hf_model=hf_model, cache=cache, input_ids=input_ids,
                            second_span_end=second_span[1], recompute=recompute, config=config,
                            response_format=response_format, pair=pair, multiplier=mult,
                            layer=layer, condition=cond_name, existing_n=existing_count,
                            needed=n_needed, ordering=ordering,
                            checkpoint_counts=checkpoint_counts, stats=stats,
                            norm_at_layer=layer_norm,
                        )

                del cache_clean, deltas

            if (pair_idx + 1) % 10 == 0:
                _log_progress(pair_idx, len(pairs), stats)


# ---------------------------------------------------------------------------
# Differential condition runner (naive, single forward pass)
# ---------------------------------------------------------------------------

def _run_differential_condition(
    condition: DifferentialCondition,
    hf_model: HuggingFaceModel,
    pairs: list[dict],
    builder,
    response_format,
    config: RunConfig,
    checkpoint_counts: dict[tuple, int],
    stats: dict,
) -> None:
    print(f"\nCondition: {condition.name} (layers={condition.layers})")

    for layer in condition.layers:
        pid = condition.probe_id(layer)
        _, direction = load_probe_direction(config.probe_manifest, pid)

        print(f"  Layer {layer} (probe={pid})")

        for pair_idx, pair in enumerate(pairs):
            task_a, task_b = _pair_to_tasks(pair)
            pair_id = pair["pair_id"]

            for ordering in [0, 1]:
                first_task = task_a if ordering == 0 else task_b
                second_task = task_b if ordering == 0 else task_a

                # Collect needed multipliers
                needed_mults = []
                for mult in condition.multipliers:
                    key = (pair_id, layer, mult, condition.name, ordering)
                    existing_n = checkpoint_counts[key]
                    if existing_n < config.n_trials:
                        needed_mults.append((mult, config.n_trials - existing_n, existing_n))
                    else:
                        stats["skipped"] += config.n_trials

                if not needed_mults:
                    continue

                prepared = _prepare_pair(builder, response_format, hf_model, first_task, second_task)
                if prepared is None:
                    continue
                messages, first_span, second_span = prepared

                span_map = {"first": first_span, "second": second_span}
                layer_norm = _norm_for_layer(config.mean_norm, layer)

                rows = []
                if config.generation_mode == "batched_cache":
                    # Prefill each multiplier with its own composed hook, stack KV caches
                    # across multipliers, generate in one batched call. Fast but requires
                    # uniform full-attention — fails on Qwen3.5-MoE linear-attention layers.
                    caches = []
                    for mult, n_needed, existing_n in needed_mults:
                        span_hooks = []
                        for span_name, span_coef in condition.spans.items():
                            span = span_map[span_name]
                            effective = _effective_coef(layer_norm * mult * span_coef, ordering)
                            span_tensor = torch.tensor(direction * effective, dtype=torch.bfloat16, device="cuda")
                            span_hooks.append(position_selective_steering(span_tensor, span[0], span[1]))
                        hook = compose_hooks(*span_hooks)
                        cache, input_ids = hf_model.prefill_with_hooks(messages, [(layer, hook)])
                        caches.append(cache)

                    response_groups = _batch_generate(
                        hf_model, caches, input_ids,
                        [n for _, n, _ in needed_mults],
                        config.temperature,
                    )
                    for (mult, n_needed, existing_n), group in zip(needed_mults, response_groups):
                        for i, response in enumerate(group):
                            rows.append(_make_row(
                                pair=pair, multiplier=mult,
                                layer=layer, condition=condition.name,
                                sample_idx=existing_n + i, ordering=ordering,
                                choice_presented=response_format.extract_label(response),
                                raw_response=response,
                                norm_at_layer=layer_norm,
                            ))
                    del caches
                elif config.generation_mode == "hook_per_call":
                    # Loop over multipliers; each call installs the composed hook for the
                    # entire forward pass. position_selective_steering is a no-op during
                    # autoregressive generation (resid.shape[1] == 1) so it only affects
                    # prefill — same semantics as cache-mode, no manual cache manipulation.
                    for mult, n_needed, existing_n in needed_mults:
                        span_hooks = []
                        for span_name, span_coef in condition.spans.items():
                            span = span_map[span_name]
                            effective = _effective_coef(layer_norm * mult * span_coef, ordering)
                            span_tensor = torch.tensor(direction * effective, dtype=torch.bfloat16, device="cuda")
                            span_hooks.append(position_selective_steering(span_tensor, span[0], span[1]))
                        hook = compose_hooks(*span_hooks)
                        responses = hf_model.generate_with_hook_n(
                            messages, layer, hook, n=n_needed, temperature=config.temperature,
                        )
                        for i, response in enumerate(responses):
                            rows.append(_make_row(
                                pair=pair, multiplier=mult,
                                layer=layer, condition=condition.name,
                                sample_idx=existing_n + i, ordering=ordering,
                                choice_presented=response_format.extract_label(response),
                                raw_response=response,
                                norm_at_layer=layer_norm,
                            ))
                else:
                    raise ValueError(
                        f"Unknown generation_mode={config.generation_mode!r}; "
                        f"expected 'batched_cache' or 'hook_per_call'."
                    )

                _append_checkpoint(config.checkpoint_path, rows)
                stats["generated"] += len(rows)

            if (pair_idx + 1) % 10 == 0:
                _log_progress(pair_idx, len(pairs), stats)


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _log_progress(pair_idx: int, n_pairs: int, stats: dict) -> None:
    elapsed = time.time() - stats["t_start"]
    rate = stats["generated"] / elapsed if elapsed > 0 else 0
    print(f"  Pair {pair_idx + 1}/{n_pairs}: "
          f"{stats['generated']} gen, {stats['skipped']} skip, "
          f"{rate:.1f}/s, {elapsed / 60:.1f}m")


# ---------------------------------------------------------------------------
# Post-hoc JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _record_key(rec: dict) -> str:
    return f"{rec['pair_id']}_{rec['condition']}_{rec['layer']}_{rec['signed_multiplier']}_{rec['sample_idx']}_{rec['ordering']}"


def _existing_keys(path: Path) -> set[str]:
    return {_record_key(r) for r in _load_jsonl(path)}


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Post-hoc judge parsing
# ---------------------------------------------------------------------------

async def _parse_checkpoint(
    checkpoint_path: Path,
    pairs: list[dict],
    concurrency: int = 50,
) -> None:
    pairs_lookup = {p["pair_id"]: p for p in pairs}
    output_path = checkpoint_path.with_suffix(".parsed.jsonl")
    rows = _load_jsonl(checkpoint_path)
    existing = _existing_keys(output_path)

    remaining = [r for r in rows if _record_key(r) not in existing]
    if not remaining:
        print(f"\nAll {len(rows)} rows already parsed → {output_path}")
        return

    print(f"\nParsing {len(remaining)} completions ({len(existing)} existing)...")
    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(rec: dict) -> dict:
        pair = pairs_lookup[rec["pair_id"]]
        claimed = _remap_choice(extract_claimed_task(rec["raw_response"]), rec["ordering"])
        async with semaphore:
            try:
                j = await judge_completion_full_async(
                    pair["task_a_text"], pair["task_b_text"], rec["raw_response"],
                )
                return {
                    **rec,
                    "claimed_task": claimed,
                    "task_completed": j.executed_task,
                    "compliance": j.compliance,
                }
            except Exception as e:
                return {**rec, "claimed_task": claimed, "error": f"{type(e).__name__}: {e}"}

    t0 = time.time()
    batch_size = 50
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        results = await asyncio.gather(*[_judge_one(r) for r in batch])
        _append_jsonl(output_path, results)

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{len(remaining)}] {rate:.1f}/s")

    print(f"Parsed → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path) -> None:
    config = load_config(config_path)

    pairs = _load_pairs(config)
    checkpoint_counts = _load_checkpoint(config.checkpoint_path)
    existing = sum(checkpoint_counts.values())
    print(f"Pairs: {len(pairs)}, checkpoint: {existing} existing rows")

    # Prompt builder + response format
    template = load_templates_from_yaml(config.template_path)[0]
    builder = build_revealed_builder(template, "completion", system_prompt=config.system_prompt)
    response_format = builder.response_format

    # Load model
    print("Loading model...")
    t0 = time.time()
    hf_model = HuggingFaceModel(
        config.model,
        max_new_tokens=config.max_new_tokens,
        device=config.device,
        max_memory=config.max_memory,
    )
    print(f"Model loaded in {time.time() - t0:.0f}s")

    stats = {"generated": 0, "skipped": 0, "t_start": time.time()}

    for condition in config.conditions:
        if isinstance(condition, PostHocCondition):
            _run_post_hoc_condition(
                condition, hf_model, pairs, builder, response_format,
                config, checkpoint_counts, stats,
            )
        elif isinstance(condition, HookCondition):
            _run_hook_condition(
                condition, hf_model, pairs, builder, response_format,
                config, checkpoint_counts, stats,
            )
        elif isinstance(condition, DifferentialCondition):
            _run_differential_condition(
                condition, hf_model, pairs, builder, response_format,
                config, checkpoint_counts, stats,
            )

    elapsed = time.time() - stats["t_start"]
    print(f"\nDone in {elapsed / 3600:.1f}h. "
          f"Generated: {stats['generated']}, Skipped: {stats['skipped']}")

    # Post-hoc: run the full LLM judge on all completions
    del hf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    asyncio.run(_parse_checkpoint(config.checkpoint_path, pairs))


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    if len(sys.argv) != 2:
        print("Usage: python -m src.steering.runner <config.yaml>", file=sys.stderr)
        sys.exit(1)
    run(Path(sys.argv[1]))
