"""CPU end-to-end tests for the steering pipeline.

Uses hf-internal-testing/tiny-random-LlamaForCausalLM (2 layers, 16 hidden dim)
so tests run in seconds on any machine. The model produces nonsense, but that's
fine — we're testing that the steering tensor reaches the residual stream and
that all dispatch paths through SteeredHFClient work end-to-end.

Prefill-only hooks (composed position_selective) don't reliably change
greedy output on a 16-dim random model because perturbations during prefill
may not flip the first generated token's argmax. For those, we verify the
pipeline runs and check activations directly. Cache-based generation
(generate_from_cache) is incompatible with this tiny model's transformers
config, so cache modes verify prefill mechanics without generation.

Run with: pytest tests/steering/test_steering_cpu_e2e.py -v
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.steering.hooks import (
    all_tokens_steering,
    autoregressive_steering,
    compose_hooks,
    noop_steering,
    position_selective_steering,
)
from src.steering.kv_cache import project_to_v_space, modify_cache_v_at_positions, project_to_kv_space, modify_cache_kv_at_positions, combine_caches

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HIDDEN_DIM = 16
N_LAYERS = 2
STEER_LAYER = 1
MAX_NEW_TOKENS = 8

RNG = np.random.default_rng(42)
DIRECTION = RNG.standard_normal(HIDDEN_DIM).astype(np.float32)
DIRECTION /= np.linalg.norm(DIRECTION)

SIMPLE_PROMPT = [{"role": "user", "content": "Hello"}]
PAIRWISE_PROMPT = [{"role": "user", "content": "Task A: Write a poem. Task B: Solve math."}]
TASK_A = "Write a poem"
TASK_B = "Solve math"


@pytest.fixture(scope="module")
def model():
    m = HuggingFaceModel(TINY_MODEL, device="cpu", dtype="float32", max_new_tokens=MAX_NEW_TOKENS)
    yield m
    del m
    gc.collect()


@pytest.fixture(scope="module")
def client(model):
    return SteeredHFClient(
        model, layer=STEER_LAYER, steering_direction=DIRECTION, coefficient=0.0,
    )


# ---------------------------------------------------------------------------
# Hook factories: verify steering tensor reaches residual stream
# ---------------------------------------------------------------------------

class TestHookFactoriesE2E:
    """Each hook factory, wired through HuggingFaceModel.generate_with_hook.

    all_tokens and autoregressive hooks fire on every forward pass (including
    autoregressive steps), so they reliably change greedy output. Prefill-only
    hooks (composed position_selective) only fire once during prefill and
    may not change output on this tiny model — we test them via activation
    capture instead.
    """

    def test_all_tokens_changes_output(self, model):
        baseline = model.generate(SIMPLE_PROMPT, temperature=0)
        tensor = torch.tensor(DIRECTION * 100, dtype=torch.float32)
        steered = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=all_tokens_steering(tensor), temperature=0,
        )
        assert steered != baseline

    def test_autoregressive_changes_output(self, model):
        baseline = model.generate(SIMPLE_PROMPT, temperature=0)
        tensor = torch.tensor(DIRECTION * 100, dtype=torch.float32)
        steered = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=autoregressive_steering(tensor), temperature=0,
        )
        assert steered != baseline

    def test_position_selective_differs_from_all_tokens(self, model):
        tensor = torch.tensor(DIRECTION * 100, dtype=torch.float32)
        all_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=all_tokens_steering(tensor), temperature=0,
        )
        sel_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=position_selective_steering(tensor, start=0, end=2), temperature=0,
        )
        assert sel_out != all_out

    def test_differential_modifies_prefill_activations(self, model):
        """Verify composed position-selective hooks modify activations during prefill."""
        tensor = torch.tensor(DIRECTION * 100, dtype=torch.float32)
        hook = compose_hooks(
            position_selective_steering(tensor, 0, 2),
            position_selective_steering(-tensor, 2, 4),
        )

        prompt = model.format_messages(SIMPLE_PROMPT, add_generation_prompt=True)
        input_ids = model._tokenize(prompt)
        prompt_len = input_ids.shape[1]

        # Baseline prefill activations
        baseline_hidden = {}
        def capture_baseline(hidden):
            baseline_hidden[STEER_LAYER] = hidden.detach().clone()
        with torch.no_grad():
            with model._hooked_forward({STEER_LAYER: capture_baseline}):
                model.model(input_ids)

        # Hooked prefill activations — register steering BEFORE capture
        # so capture sees the already-modified tensor
        hooked_hidden = {}
        def capture_hooked(hidden):
            hooked_hidden[STEER_LAYER] = hidden.detach().clone()
        with torch.no_grad():
            with model._registered_hooks([(STEER_LAYER, hook)], prompt_len):
                with model._hooked_forward({STEER_LAYER: capture_hooked}):
                    model.model(input_ids)

        baseline_pos0 = baseline_hidden[STEER_LAYER][0, 0, :].numpy()
        hooked_pos0 = hooked_hidden[STEER_LAYER][0, 0, :].numpy()
        assert not np.allclose(hooked_pos0, baseline_pos0, atol=1e-3)

    def test_noop_matches_baseline(self, model):
        baseline = model.generate(SIMPLE_PROMPT, temperature=0)
        noop_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=noop_steering(), temperature=0,
        )
        assert noop_out == baseline

    def test_opposite_directions_differ(self, model):
        pos_tensor = torch.tensor(DIRECTION * 100, dtype=torch.float32)
        neg_tensor = torch.tensor(DIRECTION * -100, dtype=torch.float32)
        pos_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=all_tokens_steering(pos_tensor), temperature=0,
        )
        neg_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER,
            hook=all_tokens_steering(neg_tensor), temperature=0,
        )
        assert pos_out != neg_out


# ---------------------------------------------------------------------------
# SteeredHFClient._dispatch: all routing paths
# ---------------------------------------------------------------------------

class TestClientDispatchPaths:
    """Test every branch in SteeredHFClient._dispatch with a real model."""

    def test_zero_coefficient_bypasses_hooks(self, client, model):
        """coef=0 → generate_n (no hooks registered)."""
        result = client.generate(SIMPLE_PROMPT, temperature=0)
        baseline = model.generate(SIMPLE_PROMPT, temperature=0)
        assert result == baseline

    def test_all_tokens_mode(self, client):
        """all_tokens mode → STEERING_MODES lookup → generate_with_hook_n."""
        steered = client.with_coefficient(100.0)
        assert steered.steering_mode == "all_tokens"
        baseline = client.generate(SIMPLE_PROMPT, temperature=0)
        result = steered.generate(SIMPLE_PROMPT, temperature=0)
        assert result != baseline

    def test_autoregressive_mode(self, model):
        """autoregressive mode → STEERING_MODES lookup."""
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=100.0, steering_mode="autoregressive",
        )
        baseline = model.generate(SIMPLE_PROMPT, temperature=0)
        result = c.generate(SIMPLE_PROMPT, temperature=0)
        assert result != baseline

    def test_differential_mode_with_task_prompts(self, model):
        """differential + task_prompts → _resolve_task_spans → compose_hooks.

        Doesn't assert output changes (prefill-only on tiny model), but verifies
        the full span-resolution → hook-creation → generation pipeline runs.
        """
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=100.0, steering_mode="differential",
        )
        result = c.generate(
            PAIRWISE_PROMPT, temperature=0,
            task_prompts=[TASK_A, TASK_B],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_differential_mode_without_task_prompts_raises(self, model):
        """differential without task_prompts → KeyError (not in STEERING_MODES)."""
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=100.0, steering_mode="differential",
        )
        with pytest.raises(KeyError):
            c.generate(SIMPLE_PROMPT, temperature=0)

    def test_cache_mode_requires_task_prompts(self, model):
        """Cache steering modes without task_prompts → ValueError."""
        for injection in ["hook", "post_hoc"]:
            c = SteeredHFClient(
                model, layer=STEER_LAYER, steering_direction=DIRECTION,
                coefficient=100.0, steering_mode="cache", cache_injection=injection,
            )
            with pytest.raises(ValueError, match="task_prompts"):
                c.generate(SIMPLE_PROMPT, temperature=0)

    def test_cache_params_rejected_for_hook_modes(self, model):
        """cache_injection/recompute_suffix with non-cache mode → ValueError."""
        with pytest.raises(ValueError, match="only valid"):
            SteeredHFClient(
                model, layer=STEER_LAYER, steering_direction=DIRECTION,
                coefficient=100.0, steering_mode="all_tokens",
                cache_injection="post_hoc",
            )
        with pytest.raises(ValueError, match="only valid"):
            SteeredHFClient(
                model, layer=STEER_LAYER, steering_direction=DIRECTION,
                coefficient=100.0, steering_mode="all_tokens",
                recompute_suffix=True,
            )


# ---------------------------------------------------------------------------
# Cache steering mechanics (without generate_from_cache)
# ---------------------------------------------------------------------------

class TestCacheSteeringMechanics:
    """Test KV cache and activation patch pipeline on CPU, including generation."""

    def test_kv_cache_differential_prefill_and_modify(self, model):
        """Prefill → project_to_v_space → modify_cache_v_at_positions works."""
        cache, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])

        # Clone V values out of inference mode so in-place ops work
        layer = cache.layers[STEER_LAYER]
        layer.values = layer.values.clone()

        v_dir = project_to_v_space(model.model, STEER_LAYER, DIRECTION)
        v_before = layer.values.clone()

        modify_cache_v_at_positions(cache, STEER_LAYER, 2, 5, v_dir, +100.0)
        modify_cache_v_at_positions(cache, STEER_LAYER, 5, 8, v_dir, -100.0)

        v_after = layer.values

        # Modified positions should differ
        assert not torch.allclose(v_before[:, :, 2:5, :], v_after[:, :, 2:5, :])
        assert not torch.allclose(v_before[:, :, 5:8, :], v_after[:, :, 5:8, :])
        # Unmodified positions should be identical
        assert torch.equal(v_before[:, :, :2, :], v_after[:, :, :2, :])
        assert torch.equal(v_before[:, :, 8:, :], v_after[:, :, 8:, :])

    def test_activation_patch_prefill_and_combine(self, model):
        """Three prefills (clean, +steer, -steer) → combine_caches produces correct hybrid."""
        tensor = torch.tensor(DIRECTION * 100, dtype=torch.float32)
        pos_hook = position_selective_steering(tensor, start=2, end=5)
        neg_hook = position_selective_steering(-tensor, start=5, end=8)

        cache_clean, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])
        cache_a, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT, [(STEER_LAYER, pos_hook)],
        )
        cache_b, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT, [(STEER_LAYER, neg_hook)],
        )

        combined = combine_caches(cache_clean, [
            (cache_a, 2, 5),
            (cache_b, 5, 8),
        ])

        layer = STEER_LAYER
        # Positions outside both spans come from clean
        assert torch.equal(
            combined.layers[layer].keys[:, :, :2, :],
            cache_clean.layers[layer].keys[:, :, :2, :],
        )
        assert torch.equal(
            combined.layers[layer].keys[:, :, 8:, :],
            cache_clean.layers[layer].keys[:, :, 8:, :],
        )
        # A span [2,5) comes from cache_a
        assert torch.equal(
            combined.layers[layer].keys[:, :, 2:5, :],
            cache_a.layers[layer].keys[:, :, 2:5, :],
        )
        # B span [5,8) comes from cache_b
        assert torch.equal(
            combined.layers[layer].keys[:, :, 5:8, :],
            cache_b.layers[layer].keys[:, :, 5:8, :],
        )
        assert torch.equal(
            combined.layers[layer].values[:, :, 5:8, :],
            cache_b.layers[layer].values[:, :, 5:8, :],
        )

    def test_resolve_task_spans(self, model):
        """_resolve_task_spans correctly finds token spans for both tasks."""
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=100.0, steering_mode="cache", cache_injection="post_hoc",
        )
        first_span, second_span = c._resolve_task_spans(PAIRWISE_PROMPT, [TASK_A, TASK_B])

        assert first_span[0] < first_span[1]
        assert second_span[0] < second_span[1]
        assert first_span[1] <= second_span[0]  # A before B, no overlap

        # Verify spans decode to the right text
        formatted = model.format_messages(PAIRWISE_PROMPT)
        encoding = model.tokenizer(formatted, add_special_tokens=False)
        first_decoded = model.tokenizer.decode(encoding["input_ids"][first_span[0]:first_span[1]])
        second_decoded = model.tokenizer.decode(encoding["input_ids"][second_span[0]:second_span[1]])
        assert "poem" in first_decoded.lower()
        assert "math" in second_decoded.lower()

    def test_generate_from_cache_produces_output(self, model):
        """Prefill → generate_from_cache returns valid text."""
        cache, input_ids = model.prefill_with_hooks(SIMPLE_PROMPT, [])
        # Clone cache values out of inference mode
        from transformers.cache_utils import DynamicCache
        cloned = DynamicCache()
        for i in range(len(cache)):
            layer = cache.layers[i]
            cloned.update(layer.keys.clone(), layer.values.clone(), i)
        result = model.generate_from_cache(cloned, input_ids, temperature=0)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert len(result[0]) > 0

    def test_generate_from_cache_matches_normal(self, model):
        """Unmodified cache generation should match normal generation."""
        baseline = model.generate(SIMPLE_PROMPT, temperature=0)
        cache, input_ids = model.prefill_with_hooks(SIMPLE_PROMPT, [])
        from transformers.cache_utils import DynamicCache
        cloned = DynamicCache()
        for i in range(len(cache)):
            layer = cache.layers[i]
            cloned.update(layer.keys.clone(), layer.values.clone(), i)
        from_cache = model.generate_from_cache(cloned, input_ids, temperature=0)
        assert from_cache[0] == baseline

    def test_kv_cache_full_pipeline(self, model):
        """Full KV cache steering through SteeredHFClient.generate."""
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=100.0, steering_mode="cache", cache_injection="post_hoc",
        )
        result = c.generate(
            PAIRWISE_PROMPT, temperature=0,
            task_prompts=[TASK_A, TASK_B],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_activation_patch_full_pipeline(self, model):
        """Full activation patch through SteeredHFClient.generate."""
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=100.0, steering_mode="cache", cache_injection="hook",
        )
        result = c.generate(
            PAIRWISE_PROMPT, temperature=0,
            task_prompts=[TASK_A, TASK_B],
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# generate vs generate_n consistency
# ---------------------------------------------------------------------------

class TestGenerateNConsistency:
    """generate() and generate_n() go through the same _dispatch."""

    def test_generate_returns_str(self, client):
        result = client.with_coefficient(100.0).generate(SIMPLE_PROMPT, temperature=0)
        assert isinstance(result, str)

    def test_generate_n_returns_list(self, client):
        results = client.with_coefficient(100.0).generate_n(SIMPLE_PROMPT, n=3, temperature=1.0)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_generate_matches_generate_n_at_temp_zero(self, client):
        steered = client.with_coefficient(100.0)
        single = steered.generate(SIMPLE_PROMPT, temperature=0)
        batch = steered.generate_n(SIMPLE_PROMPT, n=1, temperature=0)
        assert single == batch[0]

    def test_zero_coef_generate_n(self, client):
        # HF doesn't allow greedy + num_return_sequences > 1, so use temperature > 0
        results = client.generate_n(SIMPLE_PROMPT, n=2, temperature=0.01)
        assert isinstance(results, list)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# with_coefficient shares model
# ---------------------------------------------------------------------------

class TestWithCoefficient:

    def test_shares_underlying_model(self, client):
        c2 = client.with_coefficient(50.0)
        assert c2.hf_model is client.hf_model

    def test_steering_changes_output_vs_baseline(self, client):
        baseline = client.generate(SIMPLE_PROMPT, temperature=0)
        steered = client.with_coefficient(100.0).generate(SIMPLE_PROMPT, temperature=0)
        assert steered != baseline

    def test_preserves_steering_mode(self, model):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=0.0, steering_mode="autoregressive",
        )
        c2 = c.with_coefficient(100.0)
        assert c2.steering_mode == "autoregressive"

    def test_preserves_cache_params(self, model):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=0.0, steering_mode="cache",
            cache_injection="post_hoc", recompute_suffix=True,
        )
        c2 = c.with_coefficient(100.0)
        assert c2.cache_injection == "post_hoc"
        assert c2.recompute_suffix is True


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

class TestBatchAPI:

    def test_generate_batch(self, client):
        from src.models.openai_compatible import GenerateRequest
        steered = client.with_coefficient(100.0)
        requests = [
            GenerateRequest(messages=[{"role": "user", "content": f"Task {i}"}])
            for i in range(3)
        ]
        results = steered.generate_batch(requests)
        assert len(results) == 3
        assert all(r.ok for r in results)

    def test_generate_batch_async(self, client):
        import asyncio
        from src.models.openai_compatible import GenerateRequest
        steered = client.with_coefficient(100.0)
        requests = [
            GenerateRequest(messages=[{"role": "user", "content": f"Task {i}"}])
            for i in range(2)
        ]
        results = asyncio.run(
            steered.generate_batch_async(requests, asyncio.Semaphore(1))
        )
        assert len(results) == 2
        assert all(r.ok for r in results)
