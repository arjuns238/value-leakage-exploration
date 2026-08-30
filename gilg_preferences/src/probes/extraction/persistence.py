from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metadata import ExtractionMetadata


def _atomic_json_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".json")
    tmp_path = Path(tmp_path_str)
    try:
        with open(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_existing_data(
    output_dir: Path, selectors: list[str]
) -> tuple[list[str], dict[str, dict[int, list[np.ndarray]]], list[dict]]:
    """Load existing activations and completions for resume."""
    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in selectors}
    completions: list[dict] = []

    for selector in selectors:
        selector_path = output_dir / f"activations_{selector}.npz"
        if selector_path.exists():
            data = np.load(selector_path, allow_pickle=True)
            if not task_ids:
                task_ids = list(data["task_ids"])
            if "offsets" in data:
                # Span format: concat+offsets
                offsets = data["offsets"]
                n_tasks = len(offsets) - 1
                for key in data.keys():
                    if key.startswith("layer_"):
                        layer = int(key.split("_")[1])
                        concat = data[key]
                        activations[selector][layer] = [
                            concat[offsets[i]:offsets[i + 1]] for i in range(n_tasks)
                        ]
            else:
                for key in data.keys():
                    if key.startswith("layer_"):
                        layer = int(key.split("_")[1])
                        activations[selector][layer] = list(data[key])

    completions_path = output_dir / "completions_with_activations.json"
    if completions_path.exists():
        with open(completions_path) as f:
            completions = json.load(f)

    return task_ids, activations, completions


def save_activations(
    output_dir: Path,
    task_ids: list[str],
    activations: dict[str, dict[int, list[np.ndarray]]],
    span: bool = False,
) -> None:
    """Save activations per selector with atomic temp+rename.

    When span=True, uses concat+offsets (CSR) format for variable-length arrays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for selector_name, layer_acts in activations.items():
        if not layer_acts:
            continue
        tmp_path = output_dir / f"activations_{selector_name}.tmp.npz"
        final_path = output_dir / f"activations_{selector_name}.npz"

        if span:
            # First layer to compute offsets (same for all layers)
            first_layer_acts = next(iter(layer_acts.values()))
            lengths = np.array([a.shape[0] for a in first_layer_acts])
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            layer_data = {
                f"layer_{layer}": np.concatenate(acts) for layer, acts in layer_acts.items()
            }
            np.savez(tmp_path, task_ids=np.array(task_ids), offsets=offsets, **layer_data)
        else:
            np.savez(
                tmp_path,
                task_ids=np.array(task_ids),
                **{f"layer_{layer}": np.stack(acts) for layer, acts in layer_acts.items()},
            )
        tmp_path.rename(final_path)


def save_manifest(output_dir: Path, entries: list[dict]) -> None:
    _atomic_json_write(output_dir / "completions_with_activations.json", entries)


def save_extraction_metadata(output_dir: Path, metadata: ExtractionMetadata) -> None:
    _atomic_json_write(output_dir / "extraction_metadata.json", metadata.to_dict())
