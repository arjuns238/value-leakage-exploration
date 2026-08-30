"""Activation loading and filtering utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_activations(
    activations_path: Path,
    task_id_filter: set[str] | None = None,
    layers: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load activations npz file, returning (task_ids, {layer: activations})."""
    data = np.load(activations_path, allow_pickle=True)

    task_ids = data["task_ids"]

    if task_id_filter is not None:
        mask = np.array([tid in task_id_filter for tid in task_ids])
        task_ids = task_ids[mask]
    else:
        mask = None

    available_layers = sorted(int(k.split("_")[1]) for k in data.keys() if k.startswith("layer_"))
    layers_to_load = layers if layers is not None else available_layers

    activations = {}
    for layer in layers_to_load:
        arr = data[f"layer_{layer}"]
        activations[layer] = arr[mask] if mask is not None else arr

    return task_ids, activations


def load_span_activations(
    activations_path: Path,
    task_id_filter: set[str] | None = None,
    layers: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
    """Load span activations (concat+offsets format).

    Returns (task_ids, {layer: list of (n_tokens_i, d_model) arrays}).
    """
    data = np.load(activations_path, allow_pickle=True)
    task_ids = data["task_ids"]
    offsets = data["offsets"]

    available_layers = sorted(int(k.split("_")[1]) for k in data.keys() if k.startswith("layer_"))
    layers_to_load = layers if layers is not None else available_layers

    if task_id_filter is not None:
        mask = np.array([tid in task_id_filter for tid in task_ids])
        kept_indices = np.where(mask)[0]
        task_ids = task_ids[mask]
    else:
        kept_indices = range(len(task_ids))

    activations: dict[int, list[np.ndarray]] = {}
    for layer in layers_to_load:
        concat = data[f"layer_{layer}"]
        activations[layer] = [
            concat[offsets[i]:offsets[i + 1]] for i in kept_indices
        ]

    return task_ids, activations


def compute_activation_norms(
    activations_path: Path,
    layers: list[int] | None = None,
) -> dict[int, float]:
    """Compute mean L2 activation norm per layer. Returns {layer: mean_norm}."""
    data = np.load(activations_path, allow_pickle=True)
    available_layers = sorted(int(k.split("_")[1]) for k in data.keys() if k.startswith("layer_"))
    layers_to_compute = layers if layers is not None else available_layers
    return {
        layer: float(np.linalg.norm(data[f"layer_{layer}"], axis=1).mean())
        for layer in layers_to_compute
    }


def get_mean_norms(activations_path: Path, layers: list[int] | None = None) -> dict[int, float]:
    """Get mean L2 norms, reading from extraction_metadata.json if cached.

    Caches are keyed by activation filename (e.g. "activations_turn_boundary:-5.npz")
    since multiple npz files may share one metadata file.
    """
    metadata_path = activations_path.parent / "extraction_metadata.json"
    cache_key = activations_path.name

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        file_norms = metadata.get("mean_norms", {}).get(cache_key, {})
        if file_norms:
            cached = {int(k): v for k, v in file_norms.items()}
            if layers is None:
                return cached
            missing = [l for l in layers if l not in cached]
            if not missing:
                return {l: cached[l] for l in layers}

    norms = compute_activation_norms(activations_path, layers)

    # Cache to metadata, keyed by filename
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    all_norms = metadata.get("mean_norms", {})
    existing_file_norms = {int(k): v for k, v in all_norms.get(cache_key, {}).items()}
    existing_file_norms.update(norms)
    all_norms[cache_key] = {str(k): v for k, v in sorted(existing_file_norms.items())}
    metadata["mean_norms"] = all_norms
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return norms


def load_probe_data(
    activations_path: Path,
    scores: dict[str, float],
    task_ids: list[str],
    layer: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load activations and align with scores for a set of tasks.

    Filters to task_ids present in both sources. Use with
    src.probes.data_loading.load_thurstonian_scores() to get scores from a run dir.

    Returns (activations, scores, matched_task_ids).
    """
    act_task_ids, act_dict = load_activations(
        activations_path, task_id_filter=set(task_ids), layers=[layer]
    )
    act_matrix = act_dict[layer]
    act_id_to_idx = {tid: i for i, tid in enumerate(act_task_ids)}

    matched_ids = []
    matched_indices = []
    matched_scores = []
    for tid in task_ids:
        if tid in act_id_to_idx and tid in scores:
            matched_ids.append(tid)
            matched_indices.append(act_id_to_idx[tid])
            matched_scores.append(scores[tid])

    if not matched_ids:
        raise ValueError(
            f"No tasks matched between activations ({len(act_task_ids)} tasks) "
            f"and scores ({len(scores)} tasks) for the {len(task_ids)} requested IDs"
        )

    return act_matrix[matched_indices], np.array(matched_scores), matched_ids


def load_task_origins(activations_dir: Path) -> dict[str, set[str]]:
    """Load all task origins mapping. Returns {origin: set of task_ids}."""
    completions_path = activations_dir / "completions_with_activations.json"
    if not completions_path.exists():
        return {}

    with open(completions_path) as f:
        completions = json.load(f)

    origins: dict[str, set[str]] = defaultdict(set)
    for c in completions:
        if c.get("origin"):
            origins[c["origin"].upper()].add(c["task_id"])

    return dict(origins)
