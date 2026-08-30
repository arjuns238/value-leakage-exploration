from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from src.measurement.runners.utils.runner_utils import model_name_to_dir
from src.measurement.storage.base import find_project_root
from src.models.base import validate_selectors


class ExtractionConfig(BaseModel):
    model: str
    subfolder: str | None = None
    max_new_tokens: int = 2048
    task_origins: list[str] | None = None
    custom_tasks_file: Path | None = None
    n_tasks: int | None = None
    seed: int | None = None
    task_ids_file: Path | None = None
    layers_to_extract: list[float | int]
    selectors: list[str]
    batch_size: int = 32
    save_every: int = 100
    temperature: float = 1.0
    system_prompt: str | None = None
    prompt_template: str | None = None
    output_dir: str | None = None
    device: str = "cuda"
    max_memory: dict[int, str] | None = None  # e.g. {0: "60GiB", 1: "60GiB", ...} to reserve forward-pass headroom
    resume: bool = False

    @model_validator(mode="after")
    def validate_prompt_template(self) -> ExtractionConfig:
        if self.prompt_template is not None and "{task}" not in self.prompt_template:
            raise ValueError("prompt_template must contain '{task}' placeholder")
        return self

    @model_validator(mode="after")
    def validate_selectors_config(self) -> ExtractionConfig:
        validate_selectors(self.selectors)
        return self

    @model_validator(mode="after")
    def validate_task_source(self) -> ExtractionConfig:
        has_origins = self.task_origins is not None
        has_custom = self.custom_tasks_file is not None
        if not has_origins and not has_custom:
            raise ValueError("Must set either task_origins or custom_tasks_file")
        if has_origins and has_custom:
            raise ValueError("Cannot set both task_origins and custom_tasks_file")
        if has_origins and self.n_tasks is None and self.task_ids_file is None:
            raise ValueError("n_tasks is required when using task_origins without task_ids_file")
        return self

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        model_dir = model_name_to_dir(self.model)
        return find_project_root() / "activations" / model_dir

    @classmethod
    def from_yaml(cls, path: Path, **cli_overrides: object) -> ExtractionConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        for key, value in cli_overrides.items():
            if value is not None:
                data[key] = value
        return cls.model_validate(data)
