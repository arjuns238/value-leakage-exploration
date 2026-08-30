from __future__ import annotations

import numpy as np

from src.probes.bradley_terry import PairwiseActivationData, train_bt
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


def test_train_bt_recovers_preference_direction():
    """End-to-end test: measurements → training → recovers known preference direction."""
    rng = np.random.default_rng(42)
    n_tasks = 50
    d_model = 10

    # True preference direction
    true_direction = rng.standard_normal(d_model)
    true_direction /= np.linalg.norm(true_direction)

    # Generate activations with utility = activation · true_direction
    activations_raw = rng.standard_normal((n_tasks, d_model))
    task_ids = np.array([f"t{i}" for i in range(n_tasks)])
    tasks = [Task(id=f"t{i}", prompt="", origin=OriginDataset.SYNTHETIC, metadata={}) for i in range(n_tasks)]
    activations = {0: activations_raw, 5: activations_raw}  # Test multiple layers

    # Generate measurements based on true utilities
    measurements = []
    for _ in range(200):
        i, j = rng.choice(n_tasks, size=2, replace=False)
        u_i = activations_raw[i] @ true_direction
        u_j = activations_raw[j] @ true_direction
        choice = "a" if u_i > u_j else "b"
        measurements.append(
            BinaryPreferenceMeasurement(
                task_a=tasks[i],
                task_b=tasks[j],
                choice=choice,
                preference_type=PreferenceType.PRE_TASK_REVEALED,
            )
        )

    data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)

    # Train per layer using the primitive directly
    for layer in [0, 5]:
        result = train_bt(data, layer, lambdas=np.array([1e-6]))
        assert result.train_accuracy > 0.9
        assert result.cv_accuracy_mean > 0.8

        learned_direction = result.weights[:-1]  # Exclude trailing zero
        learned_direction /= np.linalg.norm(learned_direction)
        correlation = abs(np.dot(learned_direction, true_direction))
        assert correlation > 0.8
