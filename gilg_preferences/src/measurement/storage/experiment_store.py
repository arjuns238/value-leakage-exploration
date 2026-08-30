"""Experiment-level storage for organizing measurements by experiment_id."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.measurement.storage.base import save_yaml


EXPERIMENTS_DIR = Path("results/experiments")

MeasurementType = Literal[
    "pre_task_stated",
    "post_task_stated",
    "pre_task_revealed",
    "post_task_revealed",
    "pre_task_active_learning",
    "post_task_active_learning",
    "open_ended_steering",
]


class ExperimentStore:
    """Writes measurements to experiment-specific folders.

    Structure: results/experiments/{experiment_id}/{measurement_type}/{run_name}/
    """

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.base_dir = EXPERIMENTS_DIR / experiment_id

    def exists(self, measurement_type: MeasurementType, run_name: str) -> bool:
        """Check if a run already exists in this experiment."""
        run_dir = self.base_dir / measurement_type / run_name
        return (run_dir / "measurements.yaml").exists()

    def save(
        self,
        measurement_type: MeasurementType,
        run_name: str,
        measurements: list[dict],
        config: dict,
        extra_files: dict[str, dict] | None = None,
    ) -> Path:
        """Save measurements with config and optional extra files."""
        run_dir = self.base_dir / measurement_type / run_name
        save_yaml(config, run_dir / "config.yaml")
        save_yaml(measurements, run_dir / "measurements.yaml")
        if extra_files:
            for filename, data in extra_files.items():
                save_yaml(data, run_dir / filename)
        return run_dir

    # Aliases for backwards compatibility
    def save_stated(
        self,
        measurement_type: MeasurementType,
        run_name: str,
        measurements: list[dict],
        config: dict,
    ) -> Path:
        return self.save(measurement_type, run_name, measurements, config)

    def save_revealed(
        self,
        measurement_type: MeasurementType,
        run_name: str,
        measurements: list[dict],
        config: dict,
    ) -> Path:
        return self.save(measurement_type, run_name, measurements, config)
