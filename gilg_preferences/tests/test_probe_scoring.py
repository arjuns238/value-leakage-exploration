"""Unit tests for probe scoring callbacks — no GPU required."""

import numpy as np
import torch

from src.probes.scoring import _make_probe_callback, _build_callbacks


RNG = np.random.default_rng(42)
D_MODEL = 64


def _random_probe(d_model: int = D_MODEL) -> np.ndarray:
    return RNG.standard_normal(d_model + 1).astype(np.float32)


def _random_hidden(batch: int = 1, seq_len: int = 10, d_model: int = D_MODEL) -> torch.Tensor:
    return torch.randn(batch, seq_len, d_model)


class TestMakeProbeCallback:
    def test_scores_all_tokens(self):
        probe = _random_probe()
        hidden = _random_hidden(batch=2, seq_len=10)
        scores: list[np.ndarray] = []

        cb = _make_probe_callback(probe, scores, device="cpu")
        cb(hidden)

        assert len(scores) == 1
        assert scores[0].shape == (2, 10)

    def test_score_matches_manual_computation(self):
        probe = _random_probe()
        hidden = _random_hidden()
        scores: list[np.ndarray] = []

        cb = _make_probe_callback(probe, scores, device="cpu")
        cb(hidden)

        # Check last token
        act = hidden[0, -1, :].numpy()
        expected = float(act @ probe[:-1] + probe[-1])
        np.testing.assert_allclose(scores[0][0, -1], expected, rtol=1e-5)

        # Check all tokens
        all_acts = hidden[0].numpy()
        expected_all = all_acts @ probe[:-1] + probe[-1]
        np.testing.assert_allclose(scores[0][0], expected_all, rtol=1e-5)


class TestBuildCallbacks:
    def test_multiple_probes_same_layer(self):
        probes = [(8, _random_probe()), (8, _random_probe())]
        all_scores, callbacks = _build_callbacks(probes, device="cpu")

        assert len(all_scores) == 2
        assert len(callbacks) == 1  # composed into one callback at layer 8

        hidden = _random_hidden()
        callbacks[8](hidden)

        assert len(all_scores[0]) == 1
        assert len(all_scores[1]) == 1
        # Different probes should give different scores at last token
        assert all_scores[0][0][0, -1] != all_scores[1][0][0, -1]

    def test_probes_at_different_layers(self):
        probes = [(8, _random_probe()), (15, _random_probe())]
        all_scores, callbacks = _build_callbacks(probes, device="cpu")

        assert len(callbacks) == 2
        assert 8 in callbacks
        assert 15 in callbacks
