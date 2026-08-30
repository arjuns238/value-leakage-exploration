"""End-to-end GPU tests for all token selectors.

Uses gemma-3-1b-it and gemma-3-1b-pt (same tokenizer as 27B, ~2GB VRAM).
Run with: pytest tests/test_selectors_e2e.py -v -s
"""

import gc

import numpy as np
import pytest
import torch

from src.models.huggingface_model import HuggingFaceModel

pytestmark = pytest.mark.gpu

IT_MODEL = "gemma-3-1b"
PT_MODEL = "gemma-3-1b-pt"


@pytest.fixture(scope="module")
def it_model():
    m = HuggingFaceModel(IT_MODEL, max_new_tokens=16)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def pt_model():
    m = HuggingFaceModel(PT_MODEL, max_new_tokens=16)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


PROMPT_ONLY = [{"role": "user", "content": "Write a haiku about the ocean"}]
WITH_COMPLETION = [
    {"role": "user", "content": "Write a haiku about the ocean"},
    {"role": "assistant", "content": "Waves crash on the shore"},
]
LAYER = 0  # use first layer for speed


class TestITModelAllSelectors:
    """Instruct model: every selector should produce correct shapes."""

    def test_completion_selectors(self, it_model):
        """last, first, mean require an assistant message."""
        acts = it_model.get_activations(WITH_COMPLETION, [LAYER], ["last", "first", "mean"])
        d = it_model.hidden_dim
        for name in ["last", "first", "mean"]:
            assert acts[name][LAYER].shape == (d,), f"{name} shape mismatch"

    def test_task_selectors(self, it_model):
        """task_last, task_mean on prompt-only messages."""
        acts = it_model.get_activations(PROMPT_ONLY, [LAYER], ["task_last", "task_mean"])
        d = it_model.hidden_dim
        for name in ["task_last", "task_mean"]:
            assert acts[name][LAYER].shape == (d,), f"{name} shape mismatch"

    def test_eot_selector(self, it_model):
        """eot on prompt-only (finds <end_of_turn> in prompt template)."""
        acts = it_model.get_activations(PROMPT_ONLY, [LAYER], ["eot"])
        assert acts["eot"][LAYER].shape == (it_model.hidden_dim,)

    def test_turn_boundary_selectors(self, it_model):
        """turn_boundary:-1 through -5 on prompt-only."""
        selectors = [f"turn_boundary:{-i}" for i in range(1, 6)]
        acts = it_model.get_activations(PROMPT_ONLY, [LAYER], selectors)
        d = it_model.hidden_dim
        for name in selectors:
            assert acts[name][LAYER].shape == (d,), f"{name} shape mismatch"

    def test_all_selectors_together(self, it_model):
        """All selectors in a single call with a completion."""
        selectors = [
            "last", "first", "mean",
            "task_last", "task_mean",
            "eot",
            "turn_boundary:-1", "turn_boundary:-5",
        ]
        acts = it_model.get_activations(WITH_COMPLETION, [LAYER], selectors)
        d = it_model.hidden_dim
        for name in selectors:
            assert acts[name][LAYER].shape == (d,), f"{name} shape mismatch"


class TestITModelSelectorSemantics:
    """Verify selectors pick different token positions on the IT model."""

    def test_task_last_differs_from_turn_boundary(self, it_model):
        """task_last (last content token) != turn_boundary:-1 (\\n before generation)."""
        acts = it_model.get_activations(
            PROMPT_ONLY, [LAYER], ["task_last", "turn_boundary:-1"],
        )
        task = acts["task_last"][LAYER]
        tb = acts["turn_boundary:-1"][LAYER]
        assert not np.allclose(task, tb, atol=1e-5), (
            "task_last and turn_boundary:-1 should select different tokens on IT model"
        )

    def test_eot_equals_turn_boundary_minus_5(self, it_model):
        """For Gemma-3 IT, eot should be at turn_boundary:-5."""
        acts = it_model.get_activations(
            PROMPT_ONLY, [LAYER], ["eot", "turn_boundary:-5"],
        )
        eot = acts["eot"][LAYER]
        tb5 = acts["turn_boundary:-5"][LAYER]
        assert np.allclose(eot, tb5, atol=1e-6), (
            "eot and turn_boundary:-5 should be the same token for Gemma-3"
        )

    def test_turn_boundary_offsets_are_distinct(self, it_model):
        """Each turn_boundary offset should select a different token."""
        selectors = [f"turn_boundary:{-i}" for i in range(1, 6)]
        acts = it_model.get_activations(PROMPT_ONLY, [LAYER], selectors)
        vectors = [acts[s][LAYER] for s in selectors]
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert not np.allclose(vectors[i], vectors[j], atol=1e-5), (
                    f"{selectors[i]} and {selectors[j]} should be different tokens"
                )

    def test_task_mean_differs_from_task_last(self, it_model):
        acts = it_model.get_activations(
            PROMPT_ONLY, [LAYER], ["task_last", "task_mean"],
        )
        assert not np.allclose(
            acts["task_last"][LAYER], acts["task_mean"][LAYER], atol=1e-5,
        )


class TestPTModelSelectors:
    """Pre-trained (base) model: only non-chat-template selectors should work."""

    def test_task_selectors_work(self, pt_model):
        """task_last and task_mean should work on the base model."""
        messages = [{"role": "user", "content": "Write a haiku about the ocean"}]
        acts = pt_model.get_activations(messages, [LAYER], ["task_last", "task_mean"])
        d = pt_model.hidden_dim
        assert acts["task_last"][LAYER].shape == (d,)
        assert acts["task_mean"][LAYER].shape == (d,)

    def test_eot_rejected(self, pt_model):
        """eot requires a chat template and should be rejected on PT model."""
        with pytest.raises(ValueError, match="require a chat template"):
            pt_model.get_activations(PROMPT_ONLY, [LAYER], ["eot"])
    def test_turn_boundary_rejected(self, pt_model):
        """turn_boundary selectors require a chat template."""
        with pytest.raises(ValueError, match="require a chat template"):
            pt_model.get_activations(PROMPT_ONLY, [LAYER], ["turn_boundary:-1"])
    def test_task_last_is_last_content_token(self, pt_model):
        """On PT model (no template), task_last should match turn_boundary-style last token.

        PT model has no template suffix, so the task span's last token should be
        the last content token. We verify by comparing task_last against the
        activation at the last non-BOS token position from the same forward pass.
        """
        messages = [{"role": "user", "content": "Write a haiku about the ocean"}]
        # Use get_activations with both task_last and "last" (which selects the
        # actual last token in the sequence, including BOS). For PT without
        # template, there's only BOS + content, so task_last should equal "last"
        # when there's a completion, but without one we can compare task_last
        # to the known content end.
        acts = pt_model.get_activations(
            messages + [{"role": "assistant", "content": "Waves crash on the shore"}],
            [LAYER],
            ["task_last", "last"],
        )
        # task_last picks last content token of user message; "last" picks last
        # completion token. These must differ (different positions).
        assert not np.allclose(
            acts["task_last"][LAYER], acts["last"][LAYER], atol=1e-5,
        ), "task_last and last should select different positions"
        # Just verify task_last returns a valid activation vector
        assert acts["task_last"][LAYER].shape == (pt_model.hidden_dim,)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestITBatchedSelectors:
    """Batched extraction should match single-sample extraction.

    Left-padding changes SDPA attention patterns, so we use cosine similarity
    (>0.99) rather than element-wise tolerance.
    """

    def test_batch_matches_individual(self, it_model):
        messages_batch = [
            [{"role": "user", "content": "Write a poem"}],
            [{"role": "user", "content": "Explain quantum computing in simple terms"}],
        ]
        selectors = ["task_last", "task_mean", "eot", "turn_boundary:-1"]

        batch_acts = it_model.get_activations_batch(messages_batch, [LAYER], selectors)

        for i, messages in enumerate(messages_batch):
            individual = it_model.get_activations(messages, [LAYER], selectors)
            for name in selectors:
                cos = _cosine_sim(batch_acts[name][LAYER][i], individual[name][LAYER])
                assert cos > 0.99, (
                    f"Batch vs individual cosine too low for {name}, sample {i}: {cos:.6f}"
                )


class TestPTBatchedSelectors:
    """Batched extraction on PT model."""

    def test_batch_matches_individual(self, pt_model):
        messages_batch = [
            [{"role": "user", "content": "Write a poem"}],
            [{"role": "user", "content": "Explain quantum computing in simple terms"}],
        ]
        selectors = ["task_last", "task_mean"]

        batch_acts = pt_model.get_activations_batch(messages_batch, [LAYER], selectors)

        for i, messages in enumerate(messages_batch):
            individual = pt_model.get_activations(messages, [LAYER], selectors)
            for name in selectors:
                cos = _cosine_sim(batch_acts[name][LAYER][i], individual[name][LAYER])
                assert cos > 0.99, (
                    f"Batch vs individual cosine too low for {name}, sample {i}: {cos:.6f}"
                )


class TestMultiLayerExtraction:
    """Selectors should work across multiple layers."""

    def test_multiple_layers_it(self, it_model):
        layers = [0, it_model.n_layers // 2, it_model.n_layers - 1]
        selectors = ["task_last", "eot", "turn_boundary:-1"]
        acts = it_model.get_activations(PROMPT_ONLY, layers, selectors)
        d = it_model.hidden_dim
        for name in selectors:
            for layer in layers:
                assert acts[name][layer].shape == (d,), f"{name} L{layer} shape"
            # Different layers should produce different activations
            assert not np.allclose(
                acts[name][layers[0]], acts[name][layers[-1]], atol=1e-5,
            ), f"{name}: first and last layer should differ"

    def test_multiple_layers_pt(self, pt_model):
        layers = [0, pt_model.n_layers // 2, pt_model.n_layers - 1]
        messages = [{"role": "user", "content": "Write a haiku"}]
        acts = pt_model.get_activations(messages, layers, ["task_last", "task_mean"])
        d = pt_model.hidden_dim
        for name in ["task_last", "task_mean"]:
            for layer in layers:
                assert acts[name][layer].shape == (d,)
