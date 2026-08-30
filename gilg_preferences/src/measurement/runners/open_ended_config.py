"""Configuration schema for open-ended measurement experiments.

Extends base ExperimentConfig with open-ended specific parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.task_data import OriginDataset, parse_origins


class OpenEndedMeasurementConfig(BaseModel):
    """Extended config for open-ended valence measurement experiments."""

    # Core measurement type (always "open_ended" for this config)
    preference_mode: Literal["open_ended"] = "open_ended"

    # Model and generation parameters
    model: str
    temperature: float = 1.0
    max_concurrent: int | None = None

    # Task selection
    n_tasks: int = 10
    task_origins: list[Literal["wildchat", "alpaca", "math", "bailbench", "stress_test"]] = ["wildchat"]
    task_sampling_seed: int | None = None

    # If set, restrict to tasks with activations in activations/{model_name}/
    activations_model: str | None = None

    # Open-ended specific: prompt variants to measure
    prompt_variants: list[str] = Field(
        default=["experience_reflection"],
        description="List of prompt template variants to use (e.g., ['experience_reflection'])"
    )

    # Measurement parameters
    n_samples: int = Field(
        default=5,
        description="Number of repeats per task"
    )
    rating_seeds: list[int] = Field(
        default=[0],
        description="Random seeds for each measurement sample"
    )

    # Completion source
    completion_seed: int = Field(
        default=0,
        description="Seed for completion generation (used to load from CompletionStore)"
    )

    # Out-of-distribution evaluation (optional)
    include_out_of_distribution: bool = Field(
        default=False,
        description="Whether to include out-of-distribution tasks"
    )
    ood_task_origins: list[Literal["wildchat", "alpaca", "math", "bailbench", "stress_test"]] = Field(
        default=[],
        description="Task origins for OOD evaluation (different from task_origins)"
    )
    n_ood_tasks: int = Field(
        default=5,
        description="Number of OOD tasks to sample"
    )
    ood_sampling_seed: int | None = None

    # Experiment tracking
    experiment_id: str | None = None

    @model_validator(mode="after")
    def validate_ood_config(self) -> "OpenEndedMeasurementConfig":
        """Validate out-of-distribution configuration."""
        if self.include_out_of_distribution:
            if not self.ood_task_origins:
                raise ValueError("ood_task_origins must be specified when include_out_of_distribution=True")
            if set(self.ood_task_origins) == set(self.task_origins):
                raise ValueError("ood_task_origins must be different from task_origins")
        return self

    def get_origin_datasets(self) -> list[OriginDataset]:
        return parse_origins(self.task_origins)

    def get_ood_origin_datasets(self) -> list[OriginDataset]:
        return parse_origins(self.ood_task_origins)


def load_open_ended_config(path: Path) -> OpenEndedMeasurementConfig:
    """Load open-ended config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return OpenEndedMeasurementConfig.model_validate(data)
