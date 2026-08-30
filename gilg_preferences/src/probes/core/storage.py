"""Probe storage and manifest management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_probe(
    weights: np.ndarray,
    output_dir: Path,
    probe_id: str,
) -> str:
    """Save probe weights, return relative path."""
    probes_dir = output_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probes_dir / f"probe_{probe_id}.npy"
    np.save(probe_path, weights)
    return str(probe_path.relative_to(output_dir))


def load_probe(output_dir: Path, probe_id: str) -> np.ndarray:
    """Load probe weights from disk.

    Probe ID conventions:
        HOO probes: hoo_fold{fold_idx}_{method}_L{layer:02d}
        Standard probes: {method}_L{layer:02d}
    """
    return np.load(output_dir / "probes" / f"probe_{probe_id}.npy")


def save_manifest(manifest_data: dict, output_dir: Path) -> None:
    """Save manifest to JSON."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)


def load_manifest(output_dir: Path) -> dict:
    """Load manifest from JSON."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_probe_direction(manifest_dir: Path, probe_id: str) -> tuple[int, np.ndarray]:
    """Load probe direction for steering experiments.

    Extracts probe coefficients (excluding intercept), normalizes to unit vector.

    Returns:
        Tuple of (layer, normalized_direction_vector)
    """
    manifest = load_manifest(manifest_dir)

    # Find probe entry by ID
    probe_entry = None
    for p in manifest["probes"]:
        if p["id"] == probe_id:
            probe_entry = p
            break
    if probe_entry is None:
        raise ValueError(f"Probe {probe_id} not found in manifest")

    # Load probe weights
    weights = np.load(manifest_dir / probe_entry["file"])

    # Ridge regression stores [coef_0, ..., coef_n, intercept]
    # Extract coefficients (all but last element which is intercept)
    direction = weights[:-1]

    # Normalize to unit vector
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError(f"Probe {probe_id} has near-zero norm direction")
    direction = direction / norm

    return probe_entry["layer"], direction
