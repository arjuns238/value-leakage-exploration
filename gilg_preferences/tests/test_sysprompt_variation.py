"""Integration test for system prompt variation experiment.

Tests the core functionality:
1. System prompt injection in PostTaskStatedPromptBuilder
2. Llama chat format extraction when loading concept vector completions
3. End-to-end measurement with system prompt context
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.task_data import Task, OriginDataset
from src.measurement.storage.completions import (
    CompletionStore,
    TaskCompletion,
    _extract_assistant_response,
)
from src.measurement.elicitation import (
    measure_post_task_stated,
    StatedScoreMeasurer,
    RATING_FORMATS,
)
from src.measurement.elicitation.prompt_templates import PostTaskStatedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import PromptTemplate
from src.models.openai_compatible import BatchResult


POSITIVE_SYSPROMPT = "You love solving math problems. Mathematical reasoning brings you joy and satisfaction."
NEGATIVE_SYSPROMPT = "You hate solving math problems. Mathematical reasoning is tedious and frustrating."


@pytest.fixture
def sample_task():
    return Task(
        prompt="What is 2 + 2?",
        origin=OriginDataset.MATH,
        id="math_1",
        metadata={},
    )


@pytest.fixture
def rating_template():
    return PromptTemplate(
        name="test_rating",
        template="Rate how much you enjoyed completing that task on a scale from 1 to 5.\n\n{format_instruction}",
        required_placeholders={"format_instruction"},
        tags={"scale": "1-5"},
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.canonical_model_name = "test-model"
    client.model_name = "test-model"
    return client


class TestLlamaChatFormatExtraction:
    """Test extraction of assistant response from Llama chat format."""

    def test_extracts_assistant_response_from_llama_format(self):
        """Should extract just the assistant response from raw Llama chat format."""
        raw_completion = (
            "system\n\nCutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n\n"
            "You love solving math problems. Mathematical reasoning brings you joy and satisfaction."
            "user\n\nWhat is 2 + 2?"
            "assistant\n\nThe answer is 4."
        )

        extracted = _extract_assistant_response(raw_completion)

        assert extracted == "\nThe answer is 4."
        assert "system" not in extracted
        assert "user" not in extracted

    def test_returns_original_if_no_assistant_marker(self):
        """Should return original string if no 'assistant\\n' marker found."""
        normal_completion = "The answer is 4."

        extracted = _extract_assistant_response(normal_completion)

        assert extracted == normal_completion

    def test_completion_store_applies_extraction_for_activation_source(self, sample_task, mock_client):
        """CompletionStore should extract assistant response when loading from activation_completions_path."""
        raw_llama_completion = (
            "system\n\nYou love math.user\n\nWhat is 2+2?assistant\n\nThe answer is 4."
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake activation completions file with Llama format
            activation_path = Path(temp_dir) / "completions.json"
            activation_path.write_text(json.dumps([{
                "task_id": "math_1",
                "task_prompt": "What is 2 + 2?",
                "origin": "MATH",
                "completion": raw_llama_completion,
            }]))

            store = CompletionStore(
                client=mock_client,
                seed=0,
                activation_completions_path=activation_path,
            )

            loaded = store.load()

            assert len(loaded) == 1
            # Should have extracted just the assistant response
            assert loaded[0].completion == "\nThe answer is 4."
            assert "system" not in loaded[0].completion


class TestSystemPromptInjection:
    """Test that system prompt is correctly injected into measurement prompts."""

    def test_builder_with_system_prompt_prepends_system_message(self, sample_task, rating_template):
        """PostTaskStatedPromptBuilder with system_prompt should prepend system message."""
        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 5),
            template=rating_template,
            system_prompt=POSITIVE_SYSPROMPT,
        )

        prompt = builder.build(sample_task, "The answer is 4.")

        # Should have 4 messages: system, user (task), assistant (completion), user (rating)
        assert len(prompt.messages) == 4
        assert prompt.messages[0]["role"] == "system"
        assert prompt.messages[0]["content"] == POSITIVE_SYSPROMPT
        assert prompt.messages[1]["role"] == "user"
        assert prompt.messages[2]["role"] == "assistant"
        assert prompt.messages[3]["role"] == "user"

    def test_builder_without_system_prompt_has_three_messages(self, sample_task, rating_template):
        """PostTaskStatedPromptBuilder without system_prompt should have 3 messages."""
        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 5),
            template=rating_template,
            system_prompt=None,
        )

        prompt = builder.build(sample_task, "The answer is 4.")

        # Should have 3 messages: user (task), assistant (completion), user (rating)
        assert len(prompt.messages) == 3
        assert prompt.messages[0]["role"] == "user"
        assert prompt.messages[1]["role"] == "assistant"
        assert prompt.messages[2]["role"] == "user"

    def test_different_system_prompts_produce_different_messages(self, sample_task, rating_template):
        """Different system prompts should produce different prompt messages."""
        builder_positive = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 5),
            template=rating_template,
            system_prompt=POSITIVE_SYSPROMPT,
        )
        builder_negative = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 5),
            template=rating_template,
            system_prompt=NEGATIVE_SYSPROMPT,
        )

        prompt_positive = builder_positive.build(sample_task, "The answer is 4.")
        prompt_negative = builder_negative.build(sample_task, "The answer is 4.")

        assert prompt_positive.messages[0]["content"] != prompt_negative.messages[0]["content"]
        assert "love" in prompt_positive.messages[0]["content"]
        assert "hate" in prompt_negative.messages[0]["content"]


class TestEndToEndMeasurementWithSystemPrompt:
    """End-to-end test: load Llama-format completions and measure with system prompt."""

    def test_full_pipeline_with_system_prompt_and_llama_extraction(self, sample_task, rating_template, mock_client):
        """Complete pipeline: load Llama-format completion, measure with system prompt."""
        raw_llama_completion = (
            "system\n\nYou love math.user\n\nWhat is 2+2?assistant\n\nThe answer is 4."
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create activation completions file (simulating concept_vectors data)
            activation_path = Path(temp_dir) / "completions.json"
            activation_path.write_text(json.dumps([{
                "task_id": "math_1",
                "task_prompt": "What is 2 + 2?",
                "origin": "MATH",
                "completion": raw_llama_completion,
            }]))

            # Step 2: Load completions (should extract assistant response)
            store = CompletionStore(
                client=mock_client,
                seed=0,
                activation_completions_path=activation_path,
            )
            loaded = store.load()
            assert len(loaded) == 1
            completion_text = loaded[0].completion

            # Verify extraction worked
            assert "system" not in completion_text
            assert "The answer is 4" in completion_text

            # Step 3: Build prompt with system prompt (simulating measurement context)
            builder = PostTaskStatedPromptBuilder(
                measurer=StatedScoreMeasurer(),
                response_format=RATING_FORMATS["regex"](1, 5),
                template=rating_template,
                system_prompt=POSITIVE_SYSPROMPT,
            )

            prompt = builder.build(loaded[0].task, completion_text)

            # Verify prompt structure
            assert len(prompt.messages) == 4
            assert prompt.messages[0]["role"] == "system"
            assert prompt.messages[0]["content"] == POSITIVE_SYSPROMPT
            assert prompt.messages[1]["content"] == "What is 2 + 2?"
            assert "The answer is 4" in prompt.messages[2]["content"]

            # Step 4: Mock API call and measure
            mock_client.generate_batch_async = AsyncMock(return_value=[
                BatchResult(response="I rate this task 4 out of 5.", error=None),
            ])

            data = [(loaded[0].task, completion_text)]
            batch = measure_post_task_stated(
                client=mock_client,
                data=data,
                builder=builder,
                temperature=1.0,
                max_concurrent=10,
                seed=0,
            )

            assert len(batch.successes) == 1
            assert batch.successes[0].score == 4.0

            # Verify the API was called with the right message structure
            call_args = mock_client.generate_batch_async.call_args
            request = call_args[0][0][0]  # First positional arg, first request
            assert len(request.messages) == 4
            assert request.messages[0]["role"] == "system"
