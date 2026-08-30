"""End-to-end tests for the probe training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr

from src.probes.experiments.run_dir_probes import RunDirProbeConfig, _train_ridge_probe_heldout
from src.probes.experiments.hoo_ridge import build_ridge_xy


@pytest.fixture
def synthetic_probe_data():
    """Synthetic activations with a known linear signal, split into train/eval."""
    rng = np.random.default_rng(42)
    n_tasks, n_features = 200, 64

    true_direction = rng.standard_normal(n_features)
    true_direction /= np.linalg.norm(true_direction)

    activations = rng.standard_normal((n_tasks, n_features))
    scores = activations @ true_direction + rng.normal(0, 0.3, n_tasks)
    task_ids = np.array([f"task_{i:04d}" for i in range(n_tasks)])

    return {
        "activations": activations,
        "task_ids": task_ids,
        "train_scores": {task_ids[i]: float(scores[i]) for i in range(100)},
        "eval_scores": {task_ids[i]: float(scores[i]) for i in range(100, 200)},
    }


def test_probe_end_to_end(synthetic_probe_data, tmp_path):
    """Full pipeline: config -> train -> save -> verify predictions in raw space."""
    data = synthetic_probe_data

    config = RunDirProbeConfig(
        experiment_name="test_e2e",
        run_dir=tmp_path / "train",
        activations_path=tmp_path / "acts",
        output_dir=tmp_path / "output",
        layers=[0],
        modes=[],
        eval_run_dir=tmp_path / "eval",
        alpha_sweep_size=10,
    )
    config.output_dir.mkdir(parents=True)

    entry = _train_ridge_probe_heldout(
        config=config, layer=0,
        task_ids=data["task_ids"], activations=data["activations"],
        scores=data["train_scores"], eval_scores=data["eval_scores"],
        eval_measurements=[], eval_split_seed=42,
    )

    assert entry["final_r"] > 0.5
    assert entry["n_train"] == 100

    # Saved probe should predict correctly in raw (unscaled) space
    probe_path = config.output_dir / entry["file"]
    weights = np.load(probe_path)
    coef, intercept = weights[:-1], weights[-1]

    eval_indices, y_eval = build_ridge_xy(data["task_ids"], data["eval_scores"])
    predictions = data["activations"][eval_indices] @ coef + intercept
    r, _ = pearsonr(y_eval, predictions)
    assert r == pytest.approx(entry["final_r"], abs=0.01)


def test_config_requires_eval_run_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_name: test\nrun_dir: /tmp/r\n"
        "activations_path: /tmp/a\noutput_dir: /tmp/o\nlayers: [31]\n"
    )
    with pytest.raises(ValueError, match="eval_run_dir is required"):
        RunDirProbeConfig.from_yaml(config_path)
