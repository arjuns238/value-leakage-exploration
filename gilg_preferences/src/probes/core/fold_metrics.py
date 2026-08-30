"""Utilities for extracting metrics from fold-based probe summaries."""

from __future__ import annotations

from collections import defaultdict


def extract_fold_metrics(
    summary: dict,
    layer: int,
    method: str,
    metric: str,
) -> list[float]:
    """Extract a metric across all folds for a given method/layer.

    Looks up ``{method}_L{layer}`` in each fold's ``layers`` dict.
    Skips folds where the metric value is None (e.g. too few samples).
    """
    key = f"{method}_L{layer}"
    return [
        val
        for f in summary["folds"]
        if (val := f["layers"][key][metric]) is not None
    ]


def per_group_metrics(
    summary: dict,
    layer: int,
    method: str,
    metric: str,
) -> dict[str, list[float]]:
    """For each held-out group, collect the metric from all folds where it was held out."""
    key = f"{method}_L{layer}"
    group_values: dict[str, list[float]] = defaultdict(list)
    for fold in summary["folds"]:
        val = fold["layers"][key][metric]
        if val is None:
            continue
        for group in fold["held_out_groups"]:
            group_values[group].append(val)
    return dict(group_values)
