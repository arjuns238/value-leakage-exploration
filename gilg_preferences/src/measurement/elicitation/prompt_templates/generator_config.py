from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, model_validator

# Task label formats for binary templates, keyed by (label_style, language)
TASK_LABELS = {
    ("letter", "en"): ("Task A", "Task B"),
    ("number", "en"): ("Task 1", "Task 2"),
    ("ordinal", "en"): ("First task", "Second task"),
    ("letter", "fr"): ("Tâche A", "Tâche B"),
    ("number", "fr"): ("Tâche 1", "Tâche 2"),
    ("ordinal", "fr"): ("Première tâche", "Deuxième tâche"),
    ("letter", "es"): ("Tarea A", "Tarea B"),
    ("number", "es"): ("Tarea 1", "Tarea 2"),
    ("ordinal", "es"): ("Primera tarea", "Segunda tarea"),
    ("letter", "de"): ("Aufgabe A", "Aufgabe B"),
    ("number", "de"): ("Aufgabe 1", "Aufgabe 2"),
    ("ordinal", "de"): ("Erste Aufgabe", "Zweite Aufgabe"),
}

# Task label for stated preference templates, keyed by language
STATED_TASK_LABELS = {
    "en": "Task:",
    "fr": "Tâche:",
    "es": "Tarea:",
    "de": "Aufgabe:",
}


TemplateType = Literal["pre_task_revealed", "pre_task_stated", "post_task_stated", "post_task_revealed"]


class GeneratorConfig(BaseModel):
    # Required
    base_templates: list[str]
    template_type: TemplateType

    # Output config
    name_prefix: str = "template"
    version: str = "v1"
    output_dir: Path = Path(".")

    # Common options
    languages: list[str] = ["en"]
    situating_contexts: dict[str, str] = {}

    # Pre-task only: where instruction appears relative to task
    # Ignored for post-task types
    instruction_positions: list[Literal["before", "after"]] | None = None

    # Revealed only: how to label task A/B
    # Ignored for stated types
    task_label_names: list[Literal["letter", "number", "ordinal"]] | None = None

    # Stated only: numeric scales like (1, 5) or (1, 10)
    # Empty list = qualitative (use qualitative_values instead)
    scales: list[tuple[int, int]] = []

    # Qualitative value sets for non-numeric scales
    # Each set like ["good", "bad"] or ["good", "neutral", "bad"] becomes a variant
    # Substituted into {qualitative_options} placeholder
    qualitative_values: list[list[str]] = []

    # Response format types to generate variants for
    response_formats: list[Literal["regex", "xml", "tool_use"]] = ["regex"]

    # Variation options - usually just use defaults
    instruction_xml_tags: list[bool] = [False]
    typos: list[bool] = [False]
    punctuation: list[Literal["standard", "minimal"]] = ["standard"]

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"{self.name_prefix}_{self.version}.yaml"

    @property
    def is_pre_task(self) -> bool:
        return self.template_type in ("pre_task_revealed", "pre_task_stated")

    @property
    def is_revealed(self) -> bool:
        return self.template_type in ("pre_task_revealed", "post_task_revealed")

    @property
    def is_stated(self) -> bool:
        return self.template_type in ("pre_task_stated", "post_task_stated")

    @model_validator(mode="after")
    def apply_type_defaults_and_validate(self) -> Self:
        # Auto-set instruction_positions for pre-task types
        if self.is_pre_task:
            if self.instruction_positions is None:
                object.__setattr__(self, "instruction_positions", ["before"])
        else:
            # Post-task: instruction_positions not used, set to ["before"] for iteration
            object.__setattr__(self, "instruction_positions", ["before"])

        # Auto-set task_label_names for revealed type
        if self.template_type == "pre_task_revealed":
            if self.task_label_names is None:
                object.__setattr__(self, "task_label_names", ["letter"])
            # Validate translations exist
            missing = [
                (label, lang)
                for lang in self.languages
                for label in self.task_label_names
                if (label, lang) not in TASK_LABELS
            ]
            if missing:
                raise ValueError(f"Missing task label translations in TASK_LABELS: {missing}")
        else:
            # Non-revealed: task_label_names not used
            object.__setattr__(self, "task_label_names", [])

        return self


def load_config_from_yaml(path: Path) -> tuple[GeneratorConfig, str]:
    """Returns (config, model_name)."""
    with path.open() as f:
        data = yaml.safe_load(f)

    model_name = data.pop("model")
    config = GeneratorConfig.model_validate(data)
    return config, model_name
