import numpy as np

from src.probes.core.evaluate import score_with_probe


def test_score_with_probe_basic():
    """Scores = activations @ coef + intercept."""
    coef = np.array([1.0, 2.0, 3.0])
    intercept = 0.5
    probe_weights = np.append(coef, intercept)
    activations = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    scores = score_with_probe(probe_weights, activations)
    np.testing.assert_array_almost_equal(scores, [1.5, 2.5, 3.5])


def test_score_with_probe_single_sample():
    probe_weights = np.array([2.0, -1.0, 0.0])  # coef=[2, -1], intercept=0
    activations = np.array([[3.0, 4.0]])
    scores = score_with_probe(probe_weights, activations)
    np.testing.assert_array_almost_equal(scores, [2.0])


def test_score_with_probe_zero_intercept():
    probe_weights = np.array([1.0, 1.0, 0.0])
    activations = np.array([[5.0, 3.0]])
    scores = score_with_probe(probe_weights, activations)
    np.testing.assert_array_almost_equal(scores, [8.0])


def test_score_with_probe_matches_manual_computation():
    """score_with_probe should match the hand-rolled pattern from experiments."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 20, 128
    activations = rng.standard_normal((n_samples, n_features))
    probe_weights = rng.standard_normal(n_features + 1)

    # What the experiments did manually
    coef = probe_weights[:-1]
    intercept = probe_weights[-1]
    expected = activations @ coef + intercept

    scores = score_with_probe(probe_weights, activations)
    np.testing.assert_array_almost_equal(scores, expected)
