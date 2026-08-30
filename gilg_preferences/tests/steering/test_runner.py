"""Tests for the config-driven steering runner.

CPU tests for core helpers, config parsing, checkpoint I/O, and cache math.
GPU integration tests for the full condition runners are marked with @pytest.mark.gpu.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
import torch
from transformers.cache_utils import DynamicCache

from src.steering.runner import (
    HookCondition,
    PostHocCondition,
    RunConfig,
    _append_checkpoint,
    _build_interpolated_cache,
    _clone_cache,
    _compute_cache_delta,
    _condition_name,
    _effective_coef,
    _load_checkpoint,
    _make_row,
    _parse_condition,
    _remap_choice,
    load_config,
)


# ---------------------------------------------------------------------------
# Ordering helpers — the logic that was previously buggy in 3 scripts
# ---------------------------------------------------------------------------

class TestEffectiveCoef:
    def test_ordering_0_unchanged(self):
        assert _effective_coef(1.5, 0) == 1.5

    def test_ordering_1_negated(self):
        assert _effective_coef(1.5, 1) == -1.5

    def test_negative_coef_ordering_1(self):
        assert _effective_coef(-2.0, 1) == 2.0

    def test_zero_coef(self):
        assert _effective_coef(0.0, 0) == 0.0
        assert _effective_coef(0.0, 1) == 0.0


class TestRemapChoice:
    def test_ordering_0_passthrough(self):
        assert _remap_choice("a", 0) == "a"
        assert _remap_choice("b", 0) == "b"

    def test_ordering_1_swaps(self):
        assert _remap_choice("a", 1) == "b"
        assert _remap_choice("b", 1) == "a"

    def test_refusal_unchanged(self):
        assert _remap_choice("refusal", 0) == "refusal"
        assert _remap_choice("refusal", 1) == "refusal"


# ---------------------------------------------------------------------------
# _make_row
# ---------------------------------------------------------------------------

class TestMakeRow:
    def test_schema(self):
        pair = {"pair_id": "p1", "task_a": "t1", "task_b": "t2", "delta_mu": 0.5}
        row = _make_row(
            pair=pair, multiplier=0.01,
            layer=25, condition="hook_patching",
            sample_idx=2, ordering=0,
            choice_presented="a", raw_response="Task A: poem",
        )
        assert row["pair_id"] == "p1"
        assert row["signed_multiplier"] == 0.01
        assert row["layer"] == 25
        assert row["condition"] == "hook_patching"
        assert row["sample_idx"] == 2
        assert row["ordering"] == 0
        assert row["choice_original"] == "a"
        assert row["choice_presented"] == "a"
        assert row["delta_mu"] == 0.5

    def test_ordering_remaps_choice(self):
        pair = {"pair_id": "p1", "task_a": "t1", "task_b": "t2", "delta_mu": 0.0}
        row = _make_row(
            pair=pair, multiplier=0.01,
            layer=-1, condition="test",
            sample_idx=0, ordering=1,
            choice_presented="a", raw_response="resp",
        )
        assert row["choice_presented"] == "a"
        assert row["choice_original"] == "b"  # remapped


# ---------------------------------------------------------------------------
# _condition_name
# ---------------------------------------------------------------------------

class TestConditionName:
    def test_no_recompute(self):
        assert _condition_name("hook_patching", False) == "hook_patching"

    def test_recompute(self):
        assert _condition_name("hook_patching", True) == "hook_patching_recompute"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestParseCondition:
    def test_post_hoc(self):
        raw = {
            "name": "kv_test",
            "cache_injection": "post_hoc",
            "probe": "ridge_L25",
            "kv_layers": [0, 61],
            "multipliers": [0.01, -0.01],
        }
        cond = _parse_condition(raw)
        assert isinstance(cond, PostHocCondition)
        assert cond.kv_layers == (0, 61)
        assert cond.multipliers == [0.01, -0.01]

    def test_hook_default_recompute(self):
        raw = {
            "name": "hook_test",
            "cache_injection": "hook",
            "probe_prefix": "ridge_L",
            "layers": [25],
            "multipliers": [0.05],
            "ref_mult": 0.05,
        }
        cond = _parse_condition(raw)
        assert isinstance(cond, HookCondition)
        assert cond.recompute_modes == [False]

    def test_hook_recompute_bool_true(self):
        raw = {
            "name": "hook_test",
            "cache_injection": "hook",
            "probe_prefix": "ridge_L",
            "layers": [25],
            "multipliers": [0.05],
            "ref_mult": 0.05,
            "recompute_suffix": True,
        }
        cond = _parse_condition(raw)
        assert cond.recompute_modes == [True]

    def test_hook_recompute_list(self):
        raw = {
            "name": "hook_test",
            "cache_injection": "hook",
            "probe_prefix": "ridge_L",
            "layers": [25],
            "multipliers": [0.05],
            "ref_mult": 0.05,
            "recompute_suffix": [False, True],
        }
        cond = _parse_condition(raw)
        assert cond.recompute_modes == [False, True]

    def test_unknown_injection_raises(self):
        with pytest.raises(ValueError, match="Unknown cache_injection"):
            _parse_condition({"name": "x", "cache_injection": "unknown"})


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(
            "model: gemma-3-27b\n"
            "max_new_tokens: 256\n"
            "pairs_path: pairs.json\n"
            "probe_manifest: probes/\n"
            "checkpoint_path: checkpoint.jsonl\n"
            "mean_norm: 35708\n"
            "n_pairs: 200\n"
            "n_trials: 3\n"
            "temperature: 1.0\n"
            "seed: 42\n"
            "template_path: templates/completion_preference.yaml\n"
            "conditions:\n"
            "  - name: kv_test\n"
            "    cache_injection: post_hoc\n"
            "    probe: ridge_L25\n"
            "    kv_layers: [0, 61]\n"
            "    multipliers: [0.01, -0.01]\n"
            "  - name: hook_test\n"
            "    cache_injection: hook\n"
            "    probe_prefix: ridge_L\n"
            "    ref_mult: 0.05\n"
            "    layers: [25, 32]\n"
            "    multipliers: [0.05, -0.05]\n"
            "    recompute_suffix: [false, true]\n"
        )
        config = load_config(cfg_path)
        assert config.model == "gemma-3-27b"
        assert config.mean_norm == 35708
        assert len(config.conditions) == 2
        assert isinstance(config.conditions[0], PostHocCondition)
        assert isinstance(config.conditions[1], HookCondition)
        hook = config.conditions[1]
        assert hook.recompute_modes == [False, True]


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

class TestCheckpointIO:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "checkpoint.jsonl"
        rows = [
            {"pair_id": "p1", "layer": 25, "signed_multiplier": 0.05, "condition": "hook", "ordering": 0, "extra": "data"},
            {"pair_id": "p1", "layer": 25, "signed_multiplier": 0.05, "condition": "hook", "ordering": 0, "extra": "data2"},
            {"pair_id": "p1", "layer": 25, "signed_multiplier": 0.05, "condition": "hook", "ordering": 1, "extra": "data3"},
        ]
        _append_checkpoint(path, rows)
        counts = _load_checkpoint(path)
        assert counts[("p1", 25, 0.05, "hook", 0)] == 2
        assert counts[("p1", 25, 0.05, "hook", 1)] == 1

    def test_missing_file_returns_empty(self, tmp_path):
        counts = _load_checkpoint(tmp_path / "nonexistent.jsonl")
        assert len(counts) == 0

    def test_append_creates_file(self, tmp_path):
        path = tmp_path / "new.jsonl"
        _append_checkpoint(path, [{"pair_id": "p1", "layer": 0, "signed_multiplier": 0.0, "condition": "c", "ordering": 0}])
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_append_is_additive(self, tmp_path):
        path = tmp_path / "checkpoint.jsonl"
        row = {"pair_id": "p1", "layer": 0, "signed_multiplier": 0.0, "condition": "c", "ordering": 0}
        _append_checkpoint(path, [row])
        _append_checkpoint(path, [row, row])
        counts = _load_checkpoint(path)
        assert counts[("p1", 0, 0.0, "c", 0)] == 3


# ---------------------------------------------------------------------------
# Cache math (CPU, no model needed)
# ---------------------------------------------------------------------------

def _make_test_cache(n_layers: int = 2, seq_len: int = 4, n_heads: int = 2, head_dim: int = 3) -> DynamicCache:
    cache = DynamicCache()
    for li in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim)
        v = torch.randn(1, n_heads, seq_len, head_dim)
        cache.update(k, v, li)
    return cache


class TestCloneCache:
    def test_clone_is_independent(self):
        original = _make_test_cache()
        cloned = _clone_cache(original)
        # Modify cloned
        cloned.layers[0].values[:] = 999.0
        # Original unchanged
        assert not torch.all(original.layers[0].values == 999.0)

    def test_clone_is_equal(self):
        original = _make_test_cache()
        cloned = _clone_cache(original)
        for li in range(len(original)):
            assert torch.equal(original.layers[li].keys, cloned.layers[li].keys)
            assert torch.equal(original.layers[li].values, cloned.layers[li].values)


class TestComputeCacheDelta:
    def test_delta_is_difference(self):
        clean = _make_test_cache()
        combined = _make_test_cache()
        deltas = _compute_cache_delta(combined, clean)
        for li in range(len(clean)):
            expected_dk = combined.layers[li].keys - clean.layers[li].keys
            expected_dv = combined.layers[li].values - clean.layers[li].values
            assert torch.equal(deltas[li][0], expected_dk)
            assert torch.equal(deltas[li][1], expected_dv)

    def test_zero_delta_for_identical(self):
        cache = _make_test_cache()
        cloned = _clone_cache(cache)
        deltas = _compute_cache_delta(cloned, cache)
        for dk, dv in deltas:
            assert torch.all(dk == 0)
            assert torch.all(dv == 0)


class TestBuildInterpolatedCache:
    def test_scale_zero_equals_clean(self):
        clean = _make_test_cache()
        combined = _make_test_cache()
        deltas = _compute_cache_delta(combined, clean)
        interp = _build_interpolated_cache(clean, deltas, 0.0)
        for li in range(len(clean)):
            assert torch.allclose(interp.layers[li].keys, clean.layers[li].keys)
            assert torch.allclose(interp.layers[li].values, clean.layers[li].values)

    def test_scale_one_equals_combined(self):
        clean = _make_test_cache()
        combined = _make_test_cache()
        deltas = _compute_cache_delta(combined, clean)
        interp = _build_interpolated_cache(clean, deltas, 1.0)
        for li in range(len(clean)):
            # atol for float32 rounding in clean + 1.0 * (combined - clean)
            assert torch.allclose(interp.layers[li].keys, combined.layers[li].keys, atol=1e-6)
            assert torch.allclose(interp.layers[li].values, combined.layers[li].values, atol=1e-6)

    def test_linearity(self):
        clean = _make_test_cache()
        combined = _make_test_cache()
        deltas = _compute_cache_delta(combined, clean)
        half = _build_interpolated_cache(clean, deltas, 0.5)
        for li in range(len(clean)):
            expected_k = clean.layers[li].keys + 0.5 * deltas[li][0]
            assert torch.allclose(half.layers[li].keys, expected_k)

    def test_does_not_mutate_clean(self):
        clean = _make_test_cache()
        original_k0 = clean.layers[0].keys.clone()
        combined = _make_test_cache()
        deltas = _compute_cache_delta(combined, clean)
        _build_interpolated_cache(clean, deltas, 5.0)
        assert torch.equal(clean.layers[0].keys, original_k0)
