"""Tests for tokenizer consistency between API and local models.

Validates that generating via API then tokenizing locally produces
correct token boundaries for activation extraction.
"""

import pytest

from src.models.huggingface_model import HuggingFaceModel
from src.models.hybrid_model import HybridActivationModel
from src.models import get_client


TEST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CANONICAL_MODEL = "llama-3.2-1b"


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace model once per test module."""
    return HuggingFaceModel(TEST_MODEL, device="cuda", dtype="bfloat16")


@pytest.fixture(scope="module")
def api_client():
    """Get API client for testing."""
    return get_client(CANONICAL_MODEL)


@pytest.mark.api
@pytest.mark.gpu
class TestHybridModelTokenization:
    """Test that hybrid model correctly handles API-generated text."""

    def test_api_completion_tokenizes_correctly(self, hf_model: HuggingFaceModel, api_client):
        """API-generated completion should tokenize correctly locally."""
        messages = [{"role": "user", "content": "Say hello in exactly two words."}]

        completion = api_client.generate(messages, temperature=0)

        full_messages = messages + [{"role": "assistant", "content": completion}]

        prompt_text = hf_model.format_messages(messages, add_generation_prompt=True)
        full_text = hf_model.format_messages(full_messages, add_generation_prompt=False)

        prompt_tokens = hf_model._tokenize(prompt_text)[0]
        full_tokens = hf_model._tokenize(full_text)[0]

        assert full_tokens.shape[0] > prompt_tokens.shape[0], (
            "Full conversation should have more tokens than prompt"
        )

    def test_hybrid_model_extraction(self, hf_model: HuggingFaceModel, api_client):
        """HybridActivationModel should successfully extract activations."""
        hybrid = HybridActivationModel(api_client, hf_model)

        messages = [{"role": "user", "content": "What is 2+2?"}]
        layers = [0, hf_model.n_layers - 1]
        selectors = ["last", "first"]

        result = hybrid.generate_with_activations(messages, layers, selectors, temperature=0)

        assert result.completion, "Should have non-empty completion"
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

        for selector in selectors:
            assert selector in result.activations
            for layer in layers:
                assert layer in result.activations[selector]
                assert result.activations[selector][layer].shape == (hf_model.hidden_dim,)


@pytest.mark.api
@pytest.mark.gpu
class TestTokenBoundaryConsistency:
    """Test token boundary detection across different scenarios."""

    def test_first_completion_idx_consistency(self, hf_model: HuggingFaceModel):
        """first_completion_idx should be consistent regardless of completion content."""
        base_messages = [{"role": "user", "content": "Hello"}]

        completions = ["Hi", "Hello there!", "A" * 100]

        first_indices = []
        for completion in completions:
            messages = base_messages + [{"role": "assistant", "content": completion}]
            idx = hf_model._get_assistant_start_position(messages)
            first_indices.append(idx)

        assert all(idx == first_indices[0] for idx in first_indices), (
            f"first_completion_idx should be same for all completions, got {first_indices}"
        )

    def test_special_characters_in_completion(self, hf_model: HuggingFaceModel):
        """Special characters should not break tokenization."""
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Code: ```python\nprint('hello')\n```"},
        ]
        layers = [0]

        acts = hf_model.get_activations(messages, layers, ["last"])
        assert acts["last"][0].shape == (hf_model.hidden_dim,)

    def test_unicode_in_completion(self, hf_model: HuggingFaceModel):
        """Unicode characters should not break tokenization."""
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Hello! 你好 🎉"},
        ]
        layers = [0]

        acts = hf_model.get_activations(messages, layers, ["last"])
        assert acts["last"][0].shape == (hf_model.hidden_dim,)


@pytest.mark.api
@pytest.mark.gpu
class TestHybridModelBatching:
    """Test batched operations on HybridActivationModel."""

    def test_get_activations_batch(self, hf_model: HuggingFaceModel):
        """Batched activation extraction should work."""
        messages_batch = [
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}],
        ]
        layers = [0, hf_model.n_layers - 1]
        selectors = ["last", "first"]

        acts = hf_model.get_activations_batch(messages_batch, layers, selectors)

        for selector in selectors:
            for layer in layers:
                assert acts[selector][layer].shape == (2, hf_model.hidden_dim)

    def test_generate_with_activations_batch(self, hf_model: HuggingFaceModel, api_client):
        """Batched generation + activation extraction should work."""
        hybrid = HybridActivationModel(api_client, hf_model)

        messages_batch = [
            [{"role": "user", "content": "Say hi"}],
            [{"role": "user", "content": "Say bye"}],
        ]
        layers = [0]
        selectors = ["last"]

        results = hybrid.generate_with_activations_batch(
            messages_batch, layers, selectors, temperature=0
        )

        assert len(results) == 2
        for result in results:
            assert result.completion
            assert result.prompt_tokens > 0
            assert result.completion_tokens > 0
            assert result.activations["last"][0].shape == (hf_model.hidden_dim,)

    def test_batch_matches_individual(self, hf_model: HuggingFaceModel, api_client):
        """Batched results should match individual calls."""
        import numpy as np

        hybrid = HybridActivationModel(api_client, hf_model)

        messages_batch = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        layers = [0]
        selectors = ["last"]

        batch_results = hybrid.generate_with_activations_batch(
            messages_batch, layers, selectors, temperature=0
        )

        for i, msgs in enumerate(messages_batch):
            individual = hybrid.generate_with_activations(msgs, layers, selectors, temperature=0)
            assert batch_results[i].completion == individual.completion
            assert np.allclose(
                batch_results[i].activations["last"][0],
                individual.activations["last"][0],
                rtol=1e-4, atol=1e-5,
            )


@pytest.mark.api
@pytest.mark.gpu
class TestMultiTurnConversations:
    """Test activation extraction in multi-turn conversations."""

    def test_multi_turn_final_assistant(self, hf_model: HuggingFaceModel):
        """Activations from final assistant turn in multi-turn conversation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        layers = [0]

        acts = hf_model.get_activations(messages, layers, ["last", "first"])

        assert acts["last"][0].shape == (hf_model.hidden_dim,)
        assert acts["first"][0].shape == (hf_model.hidden_dim,)
