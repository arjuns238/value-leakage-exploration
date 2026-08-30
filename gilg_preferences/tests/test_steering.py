"""Tests for steering library primitives."""

import gc
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.steering.hooks import (
    compose_hooks,
    position_selective_steering,
    noop_steering,
    project_out_direction,
)
from src.steering.tokenization import find_text_span, find_pairwise_task_spans
from src.probes.core.storage import load_probe_direction


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer where each whitespace-separated word is one token."""
    tokenizer = MagicMock()

    def tokenize_with_offsets(text, return_offsets_mapping=False, add_special_tokens=True):
        words = text.split(" ")
        offsets = []
        pos = 0
        for w in words:
            start = text.find(w, pos)
            offsets.append((start, start + len(w)))
            pos = start + len(w)
        result = {"input_ids": list(range(len(words)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    tokenizer.side_effect = tokenize_with_offsets
    tokenizer.__call__ = tokenize_with_offsets
    return tokenizer


class TestPositionSelectiveSteering:
    def test_steers_correct_positions(self):
        steering_tensor = torch.ones(8)
        hook = position_selective_steering(steering_tensor, start=2, end=5)

        resid = torch.zeros(1, 10, 8)
        result = hook(resid, prompt_len=10)

        assert torch.all(result[0, 2:5, :] == 1.0)
        assert torch.all(result[0, :2, :] == 0.0)
        assert torch.all(result[0, 5:, :] == 0.0)

    def test_no_op_during_autoregressive(self):
        steering_tensor = torch.ones(8)
        hook = position_selective_steering(steering_tensor, start=0, end=1)

        resid = torch.zeros(1, 1, 8)
        result = hook(resid, prompt_len=10)
        assert torch.all(result == 0.0)


class TestComposeHooksDifferential:
    def test_positive_and_negative_spans(self):
        steering_tensor = torch.ones(8) * 2.0
        hook = compose_hooks(
            position_selective_steering(steering_tensor, 0, 3),
            position_selective_steering(-steering_tensor, 5, 8),
        )

        resid = torch.zeros(1, 10, 8)
        result = hook(resid, prompt_len=10)

        assert torch.all(result[0, 0:3, :] == 2.0)
        assert torch.all(result[0, 5:8, :] == -2.0)
        assert torch.all(result[0, 3:5, :] == 0.0)
        assert torch.all(result[0, 8:, :] == 0.0)

    def test_no_op_during_autoregressive(self):
        steering_tensor = torch.ones(8)
        hook = compose_hooks(
            position_selective_steering(steering_tensor, 0, 1),
            position_selective_steering(-steering_tensor, 2, 3),
        )

        resid = torch.zeros(1, 1, 8)
        result = hook(resid, prompt_len=10)
        assert torch.all(result == 0.0)


class TestNoopSteering:
    def test_returns_unchanged(self):
        hook = noop_steering()
        resid = torch.randn(1, 10, 8)
        original = resid.clone()
        result = hook(resid, prompt_len=10)
        assert torch.all(result == original)


class TestProjectOutDirection:
    def test_zeros_component_along_direction(self):
        torch.manual_seed(0)
        direction = torch.randn(64)
        resid = torch.randn(2, 5, 64)
        out = project_out_direction(direction)(resid, prompt_len=5)
        d_unit = direction / direction.norm()
        cos = (out * d_unit).sum(dim=-1)
        assert torch.allclose(cos, torch.zeros_like(cos), atol=1e-5)

    def test_preserves_orthogonal_components(self):
        direction = torch.zeros(64)
        direction[0] = 2.0
        resid = torch.randn(2, 5, 64)
        out = project_out_direction(direction)(resid, prompt_len=5)
        assert torch.allclose(out[..., 1:], resid[..., 1:])
        assert torch.allclose(out[..., 0], torch.zeros_like(out[..., 0]), atol=1e-5)

    def test_idempotent(self):
        torch.manual_seed(0)
        direction = torch.randn(64)
        resid = torch.randn(2, 5, 64)
        hook = project_out_direction(direction)
        once = hook(resid, prompt_len=5)
        twice = hook(once, prompt_len=5)
        assert torch.allclose(once, twice, atol=1e-5)

    def test_normalizes_input_magnitude(self):
        torch.manual_seed(0)
        direction = torch.randn(64)
        resid = torch.randn(2, 5, 64)
        out_unit = project_out_direction(direction)(resid, prompt_len=5)
        out_scaled = project_out_direction(direction * 7.3)(resid, prompt_len=5)
        assert torch.allclose(out_unit, out_scaled, atol=1e-5)


class TestFindTextSpan:
    def test_finds_span(self, mock_tokenizer):
        text = "Hello world foo bar baz"
        start, end = find_text_span(mock_tokenizer, text, "foo bar")
        assert start == 2
        assert end == 4

    def test_search_after(self, mock_tokenizer):
        text = "foo bar foo bar baz"
        start, end = find_text_span(mock_tokenizer, text, "foo bar", search_after=4)
        assert start == 2
        assert end == 4

    def test_not_found_raises(self, mock_tokenizer):
        with pytest.raises(ValueError, match="not found"):
            find_text_span(mock_tokenizer, "Hello world", "missing")


class TestFindPairwiseTaskSpans:
    def test_finds_both_spans(self, mock_tokenizer):
        prompt = "Task A: Write code Task B: Fix bugs"
        first_span, second_span = find_pairwise_task_spans(
            mock_tokenizer, prompt, "Write code", "Fix bugs"
        )
        assert first_span == (2, 4)
        assert second_span == (6, 8)

    def test_missing_marker_raises(self, mock_tokenizer):
        with pytest.raises(ValueError, match="Marker"):
            find_pairwise_task_spans(
                mock_tokenizer, "no markers here", "text_a", "text_b"
            )



class TestSteeredHFClientDirection:
    def test_direction_property(self):
        from src.steering.client import SteeredHFClient

        mock_model = MagicMock()
        mock_model.device = "cpu"
        direction = np.random.randn(128).astype(np.float32)

        client = SteeredHFClient(mock_model, layer=16, steering_direction=direction, coefficient=1.0)
        assert np.array_equal(client.direction, direction)

class TestLoadProbeDirection:
    def test_load_probe_direction_real_data(self):
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        layer, direction = load_probe_direction(manifest_dir, "0004")

        assert layer == 16
        assert direction.shape == (4096,)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_load_probe_direction_invalid_id_raises(self):
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        with pytest.raises(ValueError, match="not found in manifest"):
            load_probe_direction(manifest_dir, "9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
