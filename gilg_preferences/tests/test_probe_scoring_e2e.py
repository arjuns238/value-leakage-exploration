"""End-to-end GPU tests for probe scoring.

Run with: pytest tests/test_probe_scoring_e2e.py -v -s -m gpu
Requires GPU and ~3GB VRAM (llama-3.2-1b).
"""

import gc

import numpy as np
import pytest
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.evaluate import score_with_probe
from src.probes.scoring import score_prompt, score_prompt_all_tokens, score_prompt_batch

pytestmark = pytest.mark.gpu

MODEL_NAME = "llama-3.2-1b"
HIDDEN_DIM = 2048
PROBE_LAYER = 8

RNG = np.random.default_rng(42)
PROBE_A = RNG.standard_normal(HIDDEN_DIM + 1).astype(np.float32)
PROBE_B = RNG.standard_normal(HIDDEN_DIM + 1).astype(np.float32)

SIMPLE_PROMPT = [{"role": "user", "content": "What is 2 + 2?"}]


@pytest.fixture(scope="module")
def model():
    m = HuggingFaceModel(MODEL_NAME, max_new_tokens=16)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


class TestScorePrompt:
    def test_matches_manual_extraction(self, model):
        """score_prompt matches manually captured last-token activation."""
        # Capture last-token activation via _hooked_forward
        capture_cbs, activations = model._capture_callbacks([PROBE_LAYER])
        prompt = model.format_messages(SIMPLE_PROMPT, add_generation_prompt=True)
        input_ids = model._tokenize(prompt)
        with model._hooked_forward(capture_cbs):
            with torch.inference_mode():
                model.model(input_ids)
        act = activations[PROBE_LAYER][:, -1, :].float().numpy()
        expected = score_with_probe(PROBE_A, act)[0]

        scores = score_prompt(model, SIMPLE_PROMPT, probes=[(PROBE_LAYER, PROBE_A)])

        np.testing.assert_allclose(scores[0], expected, rtol=1e-5)

    def test_multiple_probes(self, model):
        scores = score_prompt(
            model, SIMPLE_PROMPT,
            probes=[(PROBE_LAYER, PROBE_A), (PROBE_LAYER, PROBE_B)],
        )
        assert len(scores) == 2
        assert scores[0] != pytest.approx(scores[1], abs=1e-3)


class TestScorePromptAllTokens:
    def test_returns_all_token_scores(self, model):
        all_scores = score_prompt_all_tokens(
            model, SIMPLE_PROMPT, probes=[(PROBE_LAYER, PROBE_A)],
        )
        prompt = model.format_messages(SIMPLE_PROMPT, add_generation_prompt=True)
        seq_len = model._tokenize(prompt).shape[1]

        assert len(all_scores) == 1
        assert all_scores[0].shape == (seq_len,)

    def test_last_token_matches_score_prompt(self, model):
        all_scores = score_prompt_all_tokens(
            model, SIMPLE_PROMPT, probes=[(PROBE_LAYER, PROBE_A)],
        )
        last_score = score_prompt(
            model, SIMPLE_PROMPT, probes=[(PROBE_LAYER, PROBE_A)],
        )
        np.testing.assert_allclose(all_scores[0][-1], last_score[0], rtol=1e-5)

    def test_matches_manual_all_token_extraction(self, model):
        """All-token scores match manually captured activations scored offline."""
        capture_cbs, activations = model._capture_callbacks([PROBE_LAYER])
        prompt = model.format_messages(SIMPLE_PROMPT, add_generation_prompt=True)
        input_ids = model._tokenize(prompt)
        with model._hooked_forward(capture_cbs):
            with torch.inference_mode():
                model.model(input_ids)
        # Score every token offline
        acts = activations[PROBE_LAYER][0].float().numpy()  # (seq_len, d_model)
        expected = score_with_probe(PROBE_A, acts)  # (seq_len,)

        all_scores = score_prompt_all_tokens(
            model, SIMPLE_PROMPT, probes=[(PROBE_LAYER, PROBE_A)],
        )

        np.testing.assert_allclose(all_scores[0], expected, rtol=1e-4)


class TestScorePromptBatch:
    def test_batch_matches_single(self, model):
        """Batched scoring matches scoring each prompt individually.

        Left-padding introduces small numerical differences due to attention
        masking, so we use a loose tolerance. The key property is that scores
        are close and preserve rank order.
        """
        prompts = [
            [{"role": "user", "content": "What is 2 + 2?"}],
            [{"role": "user", "content": "Tell me a joke."}],
            [{"role": "user", "content": "What is the capital of France?"}],
        ]

        individual = [
            score_prompt(model, msgs, probes=[(PROBE_LAYER, PROBE_A)])[0]
            for msgs in prompts
        ]

        batched = score_prompt_batch(
            model, prompts, probes=[(PROBE_LAYER, PROBE_A)],
        )

        assert len(batched) == 1
        assert batched[0].shape == (3,)
        # Loose tolerance: left-padding shifts attention patterns
        np.testing.assert_allclose(batched[0], individual, rtol=0.05)
        # Rank order preserved
        individual_order = np.argsort(individual)
        batched_order = np.argsort(batched[0])
        np.testing.assert_array_equal(individual_order, batched_order)

    def test_batch_multiple_probes(self, model):
        prompts = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "World"}],
        ]
        batched = score_prompt_batch(
            model, prompts,
            probes=[(PROBE_LAYER, PROBE_A), (PROBE_LAYER, PROBE_B)],
        )
        assert len(batched) == 2
        assert batched[0].shape == (2,)
        assert batched[1].shape == (2,)

    def test_different_prompts_give_different_scores(self, model):
        prompts = [
            [{"role": "user", "content": "Write me a poem about the ocean"}],
            [{"role": "user", "content": "Solve x^2 + 3x - 4 = 0"}],
        ]
        batched = score_prompt_batch(
            model, prompts, probes=[(PROBE_LAYER, PROBE_A)],
        )
        assert batched[0][0] != pytest.approx(batched[0][1], abs=1e-3)
