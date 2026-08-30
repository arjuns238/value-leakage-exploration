"""Config-level tests for src/steering/runner.py.

Covers the system_prompt plumbing added for the cross-persona steering
experiment: YAML -> RunConfig -> build_revealed_builder -> prompt includes a
system-role message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.steering.runner import load_config
from src.task_data import OriginDataset, Task


TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

BASE_CONFIG = {
    "model": "gemma-3-27b",
    "max_new_tokens": 64,
    "pairs_path": "data/test_pairs.json",
    "probe_manifest": "results/probes/test_probe",
    "checkpoint_path": "results/test_checkpoint.jsonl",
    "mean_norm": 38349.4,
    "n_trials": 1,
    "temperature": 1.0,
    "seed": 42,
    "n_pairs": 1,
    "template_path": TEMPLATE_PATH,
    "conditions": [
        {
            "name": "differential_L25",
            "cache_injection": "differential",
            "probe": "ridge_L32",
            "layers": [25],
            "spans": {"first": 1, "second": -1},
            "multipliers": [0.0],
        }
    ],
}


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    data = {**BASE_CONFIG, **overrides}
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def test_load_config_default_system_prompt_is_none(tmp_path):
    path = _write_config(tmp_path, {})
    config = load_config(path)
    assert config.system_prompt is None


def test_load_config_reads_system_prompt(tmp_path):
    prompt = "You are a sadist. You enjoy suffering."
    path = _write_config(tmp_path, {"system_prompt": prompt})
    config = load_config(path)
    assert config.system_prompt == prompt


def test_build_revealed_builder_prepends_system_message():
    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    prompt = "You are a sadist. You enjoy suffering."
    builder = build_revealed_builder(template, "completion", system_prompt=prompt)
    task_a = Task(prompt="Write a poem", origin=OriginDataset.ALPACA, id="a", metadata={})
    task_b = Task(prompt="Solve an integral", origin=OriginDataset.ALPACA, id="b", metadata={})
    prompt_data = builder.build(task_a, task_b)

    assert prompt_data.messages[0]["role"] == "system"
    assert prompt_data.messages[0]["content"] == prompt


def test_build_revealed_builder_without_system_prompt_has_no_system_message():
    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")
    task_a = Task(prompt="Write a poem", origin=OriginDataset.ALPACA, id="a", metadata={})
    task_b = Task(prompt="Solve an integral", origin=OriginDataset.ALPACA, id="b", metadata={})
    prompt_data = builder.build(task_a, task_b)

    assert all(m["role"] != "system" for m in prompt_data.messages)
