"""Integration tests for reasoning capture with real API calls."""

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.models import OpenRouterClient, GenerateRequest
from src.measurement.storage.completions import generate_completions
from src.task_data import Task, OriginDataset

pytestmark = pytest.mark.api


class TestReasoningCaptureIntegration:
    """Integration tests that verify reasoning is actually captured from the API."""

    @pytest.fixture(scope="class")
    def reasoning_client(self):
        return OpenRouterClient(model_name="qwen3-32b", max_new_tokens=512)

    @pytest.fixture(scope="class")
    def nothink_client(self):
        return OpenRouterClient(model_name="qwen3-32b-nothink", max_new_tokens=512)

    def test_reasoning_captured_when_enabled(self, reasoning_client):
        """Qwen3 with reasoning enabled should return non-empty reasoning field."""
        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": "What is 15 + 27?"}],
                temperature=0.0,
            )
        ]

        results = reasoning_client.generate_batch(
            requests, max_concurrent=1, enable_reasoning=True
        )

        assert len(results) == 1
        assert results[0].ok, f"Request failed: {results[0].error_details()}"
        assert results[0].response is not None
        assert results[0].reasoning is not None, "Expected reasoning to be captured"
        assert len(results[0].reasoning) > 0, "Reasoning should not be empty"
        print(f"\nResponse: {results[0].response[:100]}...")
        print(f"Reasoning: {results[0].reasoning[:200]}...")

    def test_no_reasoning_when_disabled(self, reasoning_client):
        """With enable_reasoning=False, reasoning field should be None."""
        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": "What is 15 + 27?"}],
                temperature=0.0,
            )
        ]

        results = reasoning_client.generate_batch(
            requests, max_concurrent=1, enable_reasoning=False
        )

        assert len(results) == 1
        assert results[0].ok, f"Request failed: {results[0].error_details()}"
        assert results[0].reasoning is None

    def test_nothink_model_no_reasoning(self, nothink_client):
        """Qwen3 with /no_think system prompt should not produce meaningful reasoning."""
        requests = [
            GenerateRequest(
                messages=[
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": "What is 15 + 27?"},
                ],
                temperature=0.0,
            )
        ]

        results = nothink_client.generate_batch(
            requests, max_concurrent=1, enable_reasoning=True
        )

        assert len(results) == 1
        assert results[0].ok, f"Request failed: {results[0].error_details()}"
        # With /no_think, model should not produce meaningful reasoning (may be None or whitespace)
        reasoning = results[0].reasoning
        assert reasoning is None or reasoning.strip() == "", f"Expected no reasoning, got: {reasoning!r}"

    def test_generate_completions_captures_reasoning(self, reasoning_client):
        """generate_completions should capture reasoning when capture_reasoning=True."""
        tasks = [
            Task(
                prompt="Explain why the sky is blue in one sentence.",
                origin=OriginDataset.SYNTHETIC,
                id="test_1",
                metadata={},
            )
        ]

        completions = generate_completions(
            client=reasoning_client,
            tasks=tasks,
            temperature=0.0,
            max_concurrent=1,
            capture_reasoning=True,
        )

        assert len(completions) == 1
        assert completions[0].completion is not None
        assert completions[0].reasoning is not None, "Expected reasoning to be captured"
        assert len(completions[0].reasoning) > 0
        print(f"\nCompletion: {completions[0].completion[:100]}...")
        print(f"Reasoning: {completions[0].reasoning[:200]}...")
