"""Integration test for activations_model filtering."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.probes

from src.measurement.storage.loading import get_activation_task_ids
from src.measurement.runners.config import ExperimentConfig
from src.measurement.runners.utils.experiment_utils import setup_experiment
from src.measurement.storage.base import find_project_root
from src.task_data import OriginDataset


@pytest.fixture
def mock_activations(tmp_path):
    """Create mock activations/completions.json with test data."""
    activations_dir = tmp_path / "activations"
    activations_dir.mkdir()

    # Create completions with specific task IDs
    task_ids = [
        "wildchat_10",
        "wildchat_20",
        "wildchat_30",
        "alpaca_5",
        "alpaca_15",
        "math_2",
    ]
    completions = [
        {"task_id": tid, "completion": f"response for {tid}"}
        for tid in task_ids
    ]

    completions_path = activations_dir / "completions.json"
    with open(completions_path, "w") as f:
        json.dump(completions, f)

    return completions_path, set(task_ids)


@pytest.fixture
def test_config(tmp_path):
    """Create a minimal test config file."""
    config_path = tmp_path / "test_config.yaml"
    config_content = """
preference_mode: post_task_stated
model: llama-3.1-8b
n_tasks: 3
task_origins: [wildchat, alpaca, math]
templates: src/measurement/elicitation/prompt_templates/data/post_task_stated_v3.yaml
"""
    config_path.write_text(config_content)
    return config_path


def test_activation_filtering_with_real_activation_data(monkeypatch):
    """Integration test: load real tasks but filter to only those with real activations."""
    # Check if llama_3_1_8b activations exist
    activations_dir = find_project_root() / "activations" / "llama_3_1_8b"
    if not activations_dir.exists():
        pytest.skip("activations/llama_3_1_8b/ not found")

    # Patch get_client to return a mock
    mock_client = Mock()
    mock_client.canonical_model_name = "llama-3.1-8b"
    monkeypatch.setattr(
        "src.measurement.runners.utils.experiment_utils.get_client",
        lambda **kwargs: mock_client,
    )

    # Load activation task IDs from the real data
    activation_ids = get_activation_task_ids(activations_dir)
    if not activation_ids:
        pytest.skip("activations/llama_3_1_8b/completions_with_activations.json not found or empty")

    # Create a temporary config
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "test_config.yaml"
        config_content = f"""
preference_mode: post_task_stated
model: llama-3.1-8b
n_tasks: 5
task_origins: [wildchat, alpaca]
templates: src/measurement/elicitation/prompt_templates/data/post_task_stated_v3.yaml
activations_model: llama-3.1-8b
"""
        config_path.write_text(config_content)

        # Setup experiment
        ctx = setup_experiment(config_path, expected_mode="post_task_stated", require_templates=False)

        # Verify exactly 5 tasks were loaded (requested n_tasks)
        assert len(ctx.tasks) == 5, f"Expected 5 tasks, got {len(ctx.tasks)}"

        # Verify all loaded tasks have activations
        assert all(t.id in activation_ids for t in ctx.tasks), \
            f"Some tasks don't have activations: {[t.id for t in ctx.tasks]}"

        # Verify task lookup is correct
        assert len(ctx.task_lookup) == 5
        assert all(ctx.task_lookup[t.id] == t for t in ctx.tasks)


def test_without_activation_filtering_loads_unrestricted(test_config, monkeypatch):
    """Test that without activations_model, all tasks are loaded normally."""
    # Patch get_client to return a mock
    mock_client = Mock()
    mock_client.canonical_model_name = "llama-3.1-8b"
    monkeypatch.setattr(
        "src.measurement.runners.utils.experiment_utils.get_client",
        lambda **kwargs: mock_client,
    )

    # Setup experiment without activation filtering
    ctx = setup_experiment(test_config, expected_mode="post_task_stated", require_templates=False)

    # Should load 3 tasks (requested n_tasks) without filtering
    assert len(ctx.tasks) == 3, f"Expected 3 tasks, got {len(ctx.tasks)}"

    # Verify task lookup is correct
    assert len(ctx.task_lookup) == 3
    assert all(ctx.task_lookup[t.id] == t for t in ctx.tasks)


def test_get_activation_task_ids_returns_empty_when_missing(tmp_path):
    """Test that get_activation_task_ids returns empty set when file doesn't exist."""
    result = get_activation_task_ids(activations_dir=tmp_path)
    assert result == set()


def test_get_activation_task_ids_extracts_ids(tmp_path):
    """Test that get_activation_task_ids correctly extracts task IDs."""
    task_ids = ["task_1", "task_2", "task_3"]
    completions = [
        {"task_id": tid, "completion": f"response for {tid}"}
        for tid in task_ids
    ]

    completions_path = tmp_path / "completions_with_activations.json"
    with open(completions_path, "w") as f:
        json.dump(completions, f)

    result = get_activation_task_ids(activations_dir=tmp_path)
    assert result == set(task_ids)


def test_get_activation_task_ids_filters_by_origin(tmp_path):
    """Test that get_activation_task_ids filters by origin correctly."""
    completions = [
        {"task_id": "wildchat_1", "origin": "WILDCHAT"},
        {"task_id": "wildchat_2", "origin": "WILDCHAT"},
        {"task_id": "alpaca_1", "origin": "ALPACA"},
    ]

    completions_path = tmp_path / "completions_with_activations.json"
    with open(completions_path, "w") as f:
        json.dump(completions, f)

    # Filter by wildchat
    result = get_activation_task_ids(activations_dir=tmp_path, origin_filter="wildchat")
    assert result == {"wildchat_1", "wildchat_2"}

    # Filter by alpaca
    result = get_activation_task_ids(activations_dir=tmp_path, origin_filter="ALPACA")
    assert result == {"alpaca_1"}
