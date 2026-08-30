from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class OODMeasurementConfig(BaseModel, frozen=True):
    model: str
    temperature: float = 0.7
    max_concurrent: int = 50
    max_new_tokens: int = 256
    n_repeats: int = 6
    template_file: str
    template_name: str
    prompts: list[str]
    mapping: str
    custom_tasks: str = ""
    prompts_dir: str = "configs/ood/prompts"
    mappings_dir: str = "configs/ood/mappings"
    tasks_dir: str = "configs/ood/tasks"
    output_dir: str = "results/ood"


def load_ood_config(path: Path) -> OODMeasurementConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return OODMeasurementConfig.model_validate(data)
