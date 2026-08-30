from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.models import OpenAICompatibleClient
    from src.measurement.elicitation.prompt_templates.template import PromptTemplate


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def model_short_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    name = name.replace("-Instruct", "").replace("Meta-", "")
    return name.lower()


def save_yaml(data: dict | list, path: Path) -> None:
    """Save data to YAML file atomically to prevent corruption on interrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    temp_path.rename(path)


def load_yaml(path: Path) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)


def build_measurement_config(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str | None = None,
    order: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Build measurement config dict for consistent storage across all measurement types."""
    from src.measurement.runners.config import get_experiment_id

    tags = dict(template.tags_dict)

    if response_format is not None:
        tags["response_format"] = response_format
    if order is not None:
        tags["order"] = order
    if seed is not None:
        tags["seed"] = str(seed)

    config = {
        "template_name": template.name,
        "template_tags": tags,
        "model": client.model_name,
        "model_short": model_short_name(client.canonical_model_name),
    }

    if temperature is not None:
        config["temperature"] = temperature

    # Use passed experiment_id or fall back to current global
    exp_id = experiment_id or get_experiment_id()
    if exp_id is not None:
        config["experiment_id"] = exp_id

    return config
