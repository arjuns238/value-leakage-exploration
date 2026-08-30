"""Checkpoint save/load for active learning runs."""

from __future__ import annotations

from pathlib import Path

from src.measurement.storage.base import save_yaml, load_yaml


CHECKPOINT_FILENAME = "checkpoint.yaml"


def save_checkpoint(
    run_dir: Path,
    iteration: int,
    comparisons_dicts: list[dict[str, str]],
    rank_correlations: list[float],
) -> Path:
    path = run_dir / CHECKPOINT_FILENAME
    data = {
        "iteration": iteration,
        "comparisons": comparisons_dicts,
        "rank_correlations": rank_correlations,
    }
    save_yaml(data, path)
    return path


def load_checkpoint(run_dir: Path) -> dict:
    return load_yaml(run_dir / CHECKPOINT_FILENAME)


def checkpoint_exists(run_dir: Path) -> bool:
    return (run_dir / CHECKPOINT_FILENAME).exists()
