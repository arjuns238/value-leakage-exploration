"""Generic held-one-out analysis (any grouping dimension)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _eval_group(fold: dict) -> str:
    if "eval_dataset" in fold:
        return fold["eval_dataset"]
    return fold["eval_group"]


def _labeled_bar(
    ax: plt.Axes,
    groups: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    colors_fn: Callable[[float], str],
    ylim: tuple[float, float] | None = None,
) -> None:
    colors = [colors_fn(v) for v in values]
    bars = ax.bar(groups, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xlabel("Held-Out Group", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    if ylim is not None:
        ax.set_ylim(*ylim)
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{val:.3f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=10, fontweight="bold")


def run(results_path: Path, output_dir: Path | None = None) -> None:
    results = _load_json(results_path)
    if not results["folds"]:
        print("No folds to plot")
        return

    if output_dir is None:
        output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_groups: list[str] = []
    r2_values: list[float] = []
    r2_adj_values: list[float] = []
    pearson_values: list[float] = []
    n_samples_values: list[float] = []

    for fold in results["folds"]:
        eval_groups.append(_eval_group(fold).upper())

        r2_scores = [p["eval_metrics"]["r2"] for p in fold["probes"] if p["eval_metrics"]["r2"] is not None]
        r2_adj_scores = [p["eval_metrics"]["r2_adjusted"] for p in fold["probes"] if p["eval_metrics"].get("r2_adjusted") is not None]
        pearson_scores = [p["eval_metrics"]["pearson_r"] for p in fold["probes"] if p["eval_metrics"]["pearson_r"] is not None]
        samples = [p["eval_metrics"]["n_samples"] for p in fold["probes"]]

        r2_values.append(float(np.median(r2_scores)) if r2_scores else 0.0)
        r2_adj_values.append(float(np.median(r2_adj_scores)) if r2_adj_scores else 0.0)
        pearson_values.append(float(np.median(pearson_scores)) if pearson_scores else 0.0)
        n_samples_values.append(float(np.mean(samples)) if samples else 0.0)

    # Text table
    print("Held-One-Out Results:")
    print(f"  {'Group':<16} {'Median R²':<12} {'R² Adj.':<12} {'Pearson r':<12} {'N samples':<10}")
    print("  " + "-" * 62)
    for g, r2, r2a, pr, ns in zip(eval_groups, r2_values, r2_adj_values, pearson_values, n_samples_values):
        print(f"  {g:<16} {r2:>10.4f}   {r2a:>10.4f}   {pr:>10.4f}   {ns:>8.0f}")

    # 2x2 summary grid
    _plot_summary_grid(eval_groups, r2_values, r2_adj_values, pearson_values, output_dir, results)

    # Heatmap: layer x eval group
    _plot_heatmap(results, eval_groups, output_dir)


def _plot_summary_grid(
    eval_groups: list[str],
    r2_values: list[float],
    r2_adj_values: list[float],
    pearson_values: list[float],
    output_dir: Path,
    results: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Held-One-Out Probe Evaluation", fontsize=14, fontweight="bold")

    pos_neg = lambda v: "#2ecc71" if v > 0 else "#e74c3c"
    blue_orange = lambda v: "#3498db" if v > 0 else "#e67e22"

    _labeled_bar(axes[0, 0], eval_groups, r2_values, "Median R²", "R² by Held-Out Group", pos_neg)
    _labeled_bar(axes[0, 1], eval_groups, r2_adj_values, "Median R² (Mean-Adjusted)", "Mean-Adjusted R²", blue_orange)
    _labeled_bar(axes[1, 0], eval_groups, pearson_values, "Median Pearson r", "Pearson Correlation",
                 lambda v: "#3498db" if abs(v) < 0.5 else "#9b59b6", ylim=(-1, 1))

    improvements = [adj - raw for raw, adj in zip(r2_values, r2_adj_values)]
    _labeled_bar(axes[1, 1], eval_groups, improvements, "R² Improvement", "R² Improvement from Mean Adjustment", pos_neg)

    plt.tight_layout()
    timestamp = results["created_at"].split("T")[0].replace("-", "")
    plot_path = output_dir / f"plot_{timestamp}_hoo_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {plot_path}")


def _plot_heatmap(results: dict, eval_groups: list[str], output_dir: Path) -> None:
    layers_set: set[int] = set()
    for fold in results["folds"]:
        for probe in fold["probes"]:
            layers_set.add(probe["layer"])

    layers = sorted(layers_set)
    groups_sorted = sorted(set(eval_groups))

    r2_matrix = np.full((len(layers), len(groups_sorted)), np.nan)
    for fold in results["folds"]:
        col_idx = groups_sorted.index(_eval_group(fold).upper())
        for probe in fold["probes"]:
            row_idx = layers.index(probe["layer"])
            r2_val = probe["eval_metrics"]["r2"]
            if r2_val is not None:
                r2_matrix[row_idx, col_idx] = r2_val

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin = np.nanmin(r2_matrix)
    vmax = max(np.nanmax(r2_matrix), 0)
    im = ax.imshow(r2_matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(groups_sorted)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(groups_sorted)
    ax.set_yticklabels(layers)
    ax.set_xlabel("Held-Out Group", fontsize=11, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=11, fontweight="bold")
    ax.set_title("Probe R² by Layer and Held-Out Group", fontsize=12, fontweight="bold")

    for i in range(len(layers)):
        for j in range(len(groups_sorted)):
            val = r2_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R² Score", fontsize=10, fontweight="bold")

    plt.tight_layout()
    timestamp = results["created_at"].split("T")[0].replace("-", "")
    plot_path = output_dir / f"plot_{timestamp}_hoo_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {plot_path}")


def run_validation(
    validation_path: Path,
    hoo_path: Path,
    manifest_path: Path,
    output_dir: Path | None = None,
) -> None:
    validation = _load_json(validation_path)
    hoo = _load_json(hoo_path)
    manifest = _load_json(manifest_path)

    if not validation["folds"]:
        print("No validation folds to plot")
        return

    if output_dir is None:
        output_dir = validation_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_map = {p["id"]: p for p in manifest["probes"]}
    hoo_probe_metrics = {
        p["id"]: p["eval_metrics"]
        for fold in hoo["folds"]
        for p in fold["probes"]
    }

    probe_ids: list[str] = []
    probe_labels: list[str] = []
    train_r2_values: list[float] = []
    cv_r2_values: list[float] = []
    test_adj_r2_values: list[float] = []
    pearson_values: list[float] = []

    first_fold = validation["folds"][0]
    n_train = len(first_fold.get("train_datasets", []))
    eval_datasets = first_fold.get("eval_datasets") or [first_fold.get("eval_dataset", "")]
    n_test = len(eval_datasets)

    for fold in validation["folds"]:
        pid = fold["probe_id"]
        probe_ids.append(pid)

        train_ds = fold.get("train_datasets", [])
        eval_ds = fold.get("eval_datasets") or [fold.get("eval_dataset", "")]
        label = f"{pid}\n(train: {','.join(train_ds[:2])}{'...' if len(train_ds) > 2 else ''})\n(test: {','.join(eval_ds[:2])}{'...' if len(eval_ds) > 2 else ''})"
        probe_labels.append(label)

        train_r2_values.append(fold["metrics"]["r2"])
        pearson_values.append(fold["metrics"]["pearson_r"])
        cv_r2_values.append(probe_map[pid]["cv_r2_mean"])
        test_adj_r2_values.append(hoo_probe_metrics[pid]["r2_adjusted"])

    # Text summary
    print("Generalization Summary:")
    print("=" * 80)
    print(f"{'Probe':<8} {'Train R²':<12} {'CV R²':<12} {'Test R² (Adj.)':<16} {'Pearson r':<12} {'Gap':<10}")
    print("-" * 80)
    for i, pid in enumerate(probe_ids):
        gap = train_r2_values[i] - test_adj_r2_values[i]
        print(f"{pid:<8} {train_r2_values[i]:<12.4f} {cv_r2_values[i]:<12.4f} {test_adj_r2_values[i]:<16.4f} {pearson_values[i]:<12.4f} {gap:<10.4f}")

    avg_train = np.mean(train_r2_values)
    avg_cv = np.mean(cv_r2_values)
    avg_test = np.mean(test_adj_r2_values)
    avg_pearson = np.mean(pearson_values)
    avg_gap = avg_train - avg_test
    print("-" * 80)
    print(f"{'MEAN':<8} {avg_train:<12.4f} {avg_cv:<12.4f} {avg_test:<16.4f} {avg_pearson:<12.4f} {avg_gap:<10.4f}")
    print("=" * 80)

    # Plot: grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("HOO Probe Generalization: Train \u2192 Test Performance", fontsize=14, fontweight="bold")

    x = np.arange(len(probe_ids))
    width = 0.2

    bars1 = ax1.bar(x - width, train_r2_values, width, label=f"Train R² ({n_train} groups)", color="#2ecc71", alpha=0.8, edgecolor="black")
    bars2 = ax1.bar(x, cv_r2_values, width, label=f"CV R² ({n_train} groups)", color="#3498db", alpha=0.8, edgecolor="black")
    bars3 = ax1.bar(x + width, test_adj_r2_values, width, label=f"Test R² Adj. ({n_test} held-out)", color="#e74c3c", alpha=0.8, edgecolor="black")

    ax1.set_xlabel("Probe", fontsize=11, fontweight="bold")
    ax1.set_ylabel("R²", fontsize=11, fontweight="bold")
    ax1.set_title(f"R² Across Train ({n_train} groups) \u2192 Test ({n_test} held-out)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(probe_labels, fontsize=9)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

    # Pearson r
    bars = ax2.bar(probe_ids, pearson_values, color="#9b59b6", alpha=0.8, edgecolor="black", linewidth=1.5)
    ax2.set_xlabel("Probe ID", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Pearson r (held-out test)", fontsize=11, fontweight="bold")
    ax2.set_title("Directional Agreement: Test Set Rankings", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, pearson_values):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    timestamp = validation["created_at"].split("T")[0].replace("-", "")
    plot_path = output_dir / f"plot_{timestamp}_hoo_validation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {plot_path}")


def plot_hoo_summary(summary: dict, output_dir: Path) -> None:
    """Plot per-fold val_r vs hoo_r for each layer."""
    folds = summary["folds"]
    layers = summary["layers"]
    if not folds:
        return

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), squeeze=False)
    axes = axes[0]

    for ax, layer in zip(axes, layers):
        fold_labels = []
        ridge_hoo = []
        bt_hoo = []
        has_ridge = False
        has_bt = False

        for f in folds:
            label = ", ".join(f["held_out_groups"])
            if len(label) > 20:
                label = label[:17] + "..."
            fold_labels.append(label)

            rk = f"ridge_L{layer}"
            if rk in f["layers"]:
                has_ridge = True
                ridge_hoo.append(f["layers"][rk]["hoo_r"])
            else:
                ridge_hoo.append(None)

            bk = f"bradley_terry_L{layer}"
            if bk in f["layers"]:
                has_bt = True
                bt_hoo.append(f["layers"][bk]["hoo_acc"])
            else:
                bt_hoo.append(None)

        x = np.arange(len(fold_labels))
        width = 0.2
        offset = 0
        if has_ridge:
            vals = [v if v is not None else 0 for v in ridge_hoo]
            ax.bar(x + offset - width/2, vals, width, label="Ridge hoo_r", color="#e74c3c", alpha=0.8)
            offset += width
        if has_bt:
            vals = [v if v is not None else 0 for v in bt_hoo]
            ax.bar(x + offset - width/2, vals, width, label="BT hoo_acc", color="#9b59b6", alpha=0.8)

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"HOO by Held-Out Group ({summary['grouping']})", fontweight="bold")
    plt.tight_layout()
    date_str = datetime.now().strftime("%m%d%y")
    plot_path = output_dir / f"plot_{date_str}_hoo_{summary['grouping']}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")
