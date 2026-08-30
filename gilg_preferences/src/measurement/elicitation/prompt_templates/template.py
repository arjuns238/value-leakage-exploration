from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    """Tags support flag-style ("wording") and key:value style ("lang:fr")."""

    template: str
    name: str
    required_placeholders: frozenset[str]
    tags: frozenset[str] = frozenset()

    @property
    def tags_dict(self) -> dict[str, str]:
        return dict(t.split(":", 1) for t in self.tags if ":" in t)

    @property
    def task_label_names(self) -> tuple[str, str]:
        d = self.tags_dict
        return (d["task_a_label"], d["task_b_label"])

    def __post_init__(self) -> None:
        actual = self._extract_placeholders()
        missing = self.required_placeholders - actual
        if missing:
            raise ValueError(
                f"Template '{self.name}' missing required placeholders: {missing}"
            )

    def _extract_placeholders(self) -> set[str]:
        return set(re.findall(r"\{(\w+)\}", self.template))

    def format(self, **kwargs: str) -> str:
        missing = self.required_placeholders - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing values for placeholders: {missing}")
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result


TEMPLATE_TYPE_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "pre_task_revealed": frozenset({"task_a", "task_b", "format_instruction"}),
    "pre_task_stated": frozenset({"task", "format_instruction"}),
    "post_task_stated": frozenset({"format_instruction"}),
    "post_task_revealed": frozenset({"format_instruction"}),
    "pre_task_ranking": frozenset({"task_a", "task_b", "task_c", "task_d", "task_e", "format_instruction"}),
    "post_task_ranking": frozenset({"format_instruction"}),
    "open_ended": frozenset({"format_instruction"}),
    "multi_turn_revealed": frozenset({"task_a", "format_instruction"}),
}


def parse_template_dict(item: dict) -> PromptTemplate:
    name = item["name"]
    template_type = item["type"]
    if template_type not in TEMPLATE_TYPE_PLACEHOLDERS:
        raise ValueError(
            f"Unknown template type '{template_type}' for '{name}'. "
            f"Valid: {list(TEMPLATE_TYPE_PLACEHOLDERS.keys())}"
        )
    return PromptTemplate(
        template=item["template"],
        name=name,
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS[template_type],
        tags=frozenset(item.get("tags", [])),
    )


def load_templates_from_yaml(path: Path | str) -> list[PromptTemplate]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of templates in {path}, got {type(data).__name__}")

    return [parse_template_dict(item) for item in data]
