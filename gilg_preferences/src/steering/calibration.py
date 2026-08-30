"""Coefficient calibration based on activation norms."""

from __future__ import annotations

from pathlib import Path

from src.probes.core.activations import get_mean_norms


def suggest_coefficient_range(
    activations_path: Path,
    layer: int,
    multipliers: list[float] | None = None,
) -> list[float]:
    """Return steering coefficients as multiples of the mean activation norm at the given layer.

    Uses cached norms from extraction_metadata.json when available.
    """
    if multipliers is None:
        multipliers = [-0.1, -0.05, 0.0, 0.05, 0.1]

    norms = get_mean_norms(activations_path, layers=[layer])
    mean_norm = norms[layer]

    return [mean_norm * m for m in multipliers]


def per_layer_norms(activations_path: Path, layers: list[int]) -> dict[int, float]:
    """Mean activation norm per layer, usable directly as the runner's `mean_norm`."""
    return get_mean_norms(activations_path, layers=layers)
