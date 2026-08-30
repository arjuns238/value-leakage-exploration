"""Shared plotting utilities for correlation analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def build_correlation_matrix(
    pair_correlations: dict[tuple[str, str], list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build symmetric correlation matrix from pairwise correlations.

    Returns (mean_matrix, std_matrix, count_matrix, labels).
    """
    labels = set()
    for a, b in pair_correlations.keys():
        labels.add(a)
        labels.add(b)
    labels = sorted(labels)
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    mean_matrix = np.eye(n)
    std_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n), dtype=int)

    for (a, b), corrs in pair_correlations.items():
        i, j = label_to_idx[a], label_to_idx[b]
        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs) if len(corrs) > 1 else 0

        mean_matrix[i, j] = mean_corr
        mean_matrix[j, i] = mean_corr
        std_matrix[i, j] = std_corr
        std_matrix[j, i] = std_corr
        count_matrix[i, j] = len(corrs)
        count_matrix[j, i] = len(corrs)

    return mean_matrix, std_matrix, count_matrix, labels


def plot_correlation_heatmap(
    mean_matrix: np.ndarray,
    count_matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    show_counts: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot heatmap of correlations."""
    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(mean_matrix.min()), abs(mean_matrix.max()), 1.0)
    im = ax.imshow(mean_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            if i == j:
                text = "1.0"
            else:
                corr = mean_matrix[i, j]
                if show_counts:
                    count = count_matrix[i, j]
                    text = f"{corr:.2f}\n(n={count})"
                else:
                    text = f"{corr:.2f}"

            color = "white" if abs(mean_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_correlation_results(
    pair_correlations: dict[tuple[str, str], list[float]],
    output_path: Path,
    extra_metadata: dict | None = None,
) -> None:
    """Save correlation results to YAML."""
    results = {}

    if extra_metadata:
        results.update(extra_metadata)

    results["pairs"] = {}
    for (a, b), corrs in sorted(pair_correlations.items()):
        key = f"{a} vs {b}"
        results["pairs"][key] = {
            "mean": float(np.mean(corrs)),
            "std": float(np.std(corrs)) if len(corrs) > 1 else 0.0,
            "n": len(corrs),
            "min": float(np.min(corrs)),
            "max": float(np.max(corrs)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {output_path}")
