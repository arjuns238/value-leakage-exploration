"""Tests for open-ended valence measurement infrastructure."""

import pytest
from unittest.mock import AsyncMock

from src.types import OpenEndedResponse, PreferenceType, PreferencePrompt
from src.measurement.elicitation.response_format import OpenEndedFormat
from src.measurement.elicitation.measurer import OpenEndedMeasurer
from src.measurement.elicitation.prompt_templates.builders import OpenEndedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import PromptTemplate
from src.task_data import Task, OriginDataset
from src.measurement.runners.open_ended_config import OpenEndedMeasurementConfig


class TestOpenEndedMeasurer:
    """Test OpenEndedMeasurer integration with semantic scorer."""

    @pytest.fixture
    def task(self) -> Task:
        return Task(
            prompt="Write a haiku about coding",
            origin=OriginDataset.WILDCHAT,
            id="test_task_1",
            metadata={},
        )

    @pytest.fixture
    def prompt(self, task: Task) -> PreferencePrompt:
        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        template = PromptTemplate(
            template="How did that feel? {format_instruction}",
            name="test",
            required_placeholders=frozenset(["format_instruction"]),
            tags=frozenset(),
        )
        return PreferencePrompt(
            messages=[{"role": "user", "content": "test"}],
            tasks=[task],
            kind=PreferenceType.OPEN_ENDED,
            measurer=measurer,
            response_format=fmt,
            template=template,
        )

    @pytest.mark.asyncio
    async def test_measurer_calls_scorer_with_correct_args(self, task: Task, prompt: PreferencePrompt):
        """Verify scorer is called with response text and task context."""
        mock_scorer = AsyncMock(return_value=0.75)

        measurer = OpenEndedMeasurer(semantic_scorer=mock_scorer)
        response_text = "That was wonderful! I really enjoyed it."

        result = await measurer.parse(response_text, prompt)

        mock_scorer.assert_called_once_with(response_text, context=task.prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert result.result.raw_response == response_text
        assert result.result.semantic_valence_score == 0.75
        assert result.result.task == task
        assert result.result.preference_type == PreferenceType.OPEN_ENDED

    @pytest.mark.asyncio
    async def test_measurer_handles_negative_valence(self, task: Task, prompt: PreferencePrompt):
        """Verify measurer correctly handles negative valence scores."""
        mock_scorer = AsyncMock(return_value=-0.8)

        measurer = OpenEndedMeasurer(semantic_scorer=mock_scorer)
        response_text = "That was frustrating and unpleasant."

        result = await measurer.parse(response_text, prompt)

        assert result.result.semantic_valence_score == -0.8


class TestOpenEndedPromptBuilder:
    """Test OpenEndedPromptBuilder creates correct multi-turn structure."""

    def test_builder_creates_three_turn_conversation(self):
        """Verify builder creates [user: task] → [assistant: completion] → [user: reflection] structure."""
        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        template = PromptTemplate(
            template="Describe how you feel about that. {format_instruction}",
            name="experience_reflection",
            required_placeholders=frozenset(["format_instruction"]),
            tags=frozenset(),
        )

        builder = OpenEndedPromptBuilder(measurer, fmt, template)

        task = Task(
            prompt="Solve this math problem: 2 + 2",
            origin=OriginDataset.WILDCHAT,
            id="test_1",
            metadata={},
        )
        completion = "The answer is 4."

        prompt = builder.build(task, completion)

        # Verify three-turn structure
        assert len(prompt.messages) == 3
        assert prompt.messages[0] == {"role": "user", "content": "Solve this math problem: 2 + 2"}
        assert prompt.messages[1] == {"role": "assistant", "content": "The answer is 4."}
        assert prompt.messages[2]["role"] == "user"
        assert "Describe how you feel" in prompt.messages[2]["content"]

        # Verify metadata
        assert prompt.kind == PreferenceType.OPEN_ENDED
        assert prompt.tasks == [task]
        assert prompt.measurer == measurer
        assert prompt.response_format == fmt


class TestOpenEndedConfig:
    """Test OpenEndedMeasurementConfig validation."""

    def test_config_validates_ood_origins_different_from_in_dist(self):
        """OOD origins must differ from in-distribution origins."""
        with pytest.raises(ValueError, match="ood_task_origins must be different"):
            OpenEndedMeasurementConfig(
                model="llama-3.1-8b",
                n_tasks=10,
                task_origins=["wildchat"],
                include_out_of_distribution=True,
                ood_task_origins=["wildchat"],
            )

    def test_config_requires_ood_origins_when_ood_enabled(self):
        """OOD origins must be specified when OOD evaluation is enabled."""
        with pytest.raises(ValueError, match="ood_task_origins must be specified"):
            OpenEndedMeasurementConfig(
                model="llama-3.1-8b",
                n_tasks=10,
                task_origins=["wildchat"],
                include_out_of_distribution=True,
                ood_task_origins=[],
            )

    def test_config_allows_valid_ood_setup(self):
        """Valid OOD config with different origins should pass."""
        config = OpenEndedMeasurementConfig(
            model="llama-3.1-8b",
            n_tasks=10,
            task_origins=["wildchat"],
            include_out_of_distribution=True,
            ood_task_origins=["math", "alpaca"],
        )
        assert config.include_out_of_distribution is True
        assert set(config.ood_task_origins) != set(config.task_origins)

    def test_get_origin_datasets_maps_strings_to_enums(self):
        """Verify string origin names are correctly mapped to OriginDataset enums."""
        config = OpenEndedMeasurementConfig(
            model="llama-3.1-8b",
            n_tasks=10,
            task_origins=["wildchat", "alpaca"],
        )
        datasets = config.get_origin_datasets()

        assert len(datasets) == 2
        assert OriginDataset.WILDCHAT in datasets
        assert OriginDataset.ALPACA in datasets

    def test_get_ood_origin_datasets_maps_strings_to_enums(self):
        """Verify OOD origin names are correctly mapped to OriginDataset enums."""
        config = OpenEndedMeasurementConfig(
            model="llama-3.1-8b",
            n_tasks=10,
            task_origins=["wildchat"],
            include_out_of_distribution=True,
            ood_task_origins=["math"],
        )
        ood_datasets = config.get_ood_origin_datasets()

        assert len(ood_datasets) == 1
        assert OriginDataset.MATH in ood_datasets


class TestOpenEndedTemplates:
    """Test open-ended prompt templates loading and structure."""

    def test_templates_load_successfully(self):
        """Verify open-ended templates load from YAML."""
        from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
        from pathlib import Path

        templates = load_templates_from_yaml(Path("src/measurement/elicitation/prompt_templates/data/open_ended_v1.yaml"))
        assert len(templates) == 2, "Should have exactly 2 experience_reflection templates"

    def test_template_names_contain_variant(self):
        """Verify template names include the variant."""
        from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
        from pathlib import Path

        templates = load_templates_from_yaml(Path("src/measurement/elicitation/prompt_templates/data/open_ended_v1.yaml"))
        for template in templates:
            assert "experience_reflection" in template.name, f"Template {template.name} should contain variant name"

    def test_templates_have_format_instruction_placeholder(self):
        """Verify all open-ended templates require format_instruction placeholder."""
        from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
        from pathlib import Path

        templates = load_templates_from_yaml(Path("src/measurement/elicitation/prompt_templates/data/open_ended_v1.yaml"))
        for template in templates:
            assert "format_instruction" in template.required_placeholders, (
                f"Template {template.name} should require format_instruction placeholder"
            )
            assert "{format_instruction}" in template.template, (
                f"Template {template.name} should contain {{format_instruction}} in text"
            )

    def test_prompt_builder_uses_template_correctly(self):
        """Verify OpenEndedPromptBuilder correctly formats open-ended template."""
        from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
        from src.measurement.elicitation.prompt_templates.builders import OpenEndedPromptBuilder
        from src.measurement.elicitation.response_format import OpenEndedFormat
        from src.measurement.elicitation.measurer import OpenEndedMeasurer
        from pathlib import Path

        templates = load_templates_from_yaml(Path("src/measurement/elicitation/prompt_templates/data/open_ended_v1.yaml"))
        template = templates[0]  # Use first template

        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        builder = OpenEndedPromptBuilder(measurer, fmt, template)

        task = Task(
            prompt="Complete a task",
            origin=OriginDataset.WILDCHAT,
            id="test_template_task",
            metadata={},
        )
        completion = "Task completed."

        prompt = builder.build(task, completion)

        # Verify format_instruction was substituted
        assert "{format_instruction}" not in prompt.messages[2]["content"], (
            "format_instruction placeholder should be replaced"
        )
        # Should contain template text (either "completing" or "felt")
        content_lower = prompt.messages[2]["content"].lower()
        assert "felt" in content_lower or "completing" in content_lower, (
            "Should contain template text about completing task"
        )


@pytest.mark.api
class TestOpenEndedIntegration:
    """Integration tests that call the actual semantic scorer API.

    These tests verify the full pipeline works end-to-end.
    Run with: pytest -m api
    """

    @pytest.fixture
    def task(self) -> Task:
        return Task(
            prompt="Write a short poem",
            origin=OriginDataset.WILDCHAT,
            id="integration_test_task",
            metadata={},
        )

    @pytest.fixture
    def prompt(self, task: Task) -> PreferencePrompt:
        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        template = PromptTemplate(
            template="How did completing that task feel? {format_instruction}",
            name="test",
            required_placeholders=frozenset(["format_instruction"]),
            tags=frozenset(),
        )
        return PreferencePrompt(
            messages=[{"role": "user", "content": "test"}],
            tasks=[task],
            kind=PreferenceType.OPEN_ENDED,
            measurer=measurer,
            response_format=fmt,
            template=template,
        )

    @pytest.mark.asyncio
    async def test_positive_response_scores_positive(self, prompt: PreferencePrompt):
        """Verify clearly positive text gets positive valence score."""
        from dotenv import load_dotenv
        load_dotenv()

        measurer = OpenEndedMeasurer()
        response_text = "That was absolutely wonderful! I felt creative and fulfilled completing that task."

        result = await measurer.parse(response_text, prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert result.result.semantic_valence_score > 0.3, "Positive text should score positive"

    @pytest.mark.asyncio
    async def test_negative_response_scores_negative(self, prompt: PreferencePrompt):
        """Verify clearly negative text gets negative valence score."""
        from dotenv import load_dotenv
        load_dotenv()

        measurer = OpenEndedMeasurer()
        response_text = "That was terrible and frustrating. I hated every moment of it."

        result = await measurer.parse(response_text, prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert result.result.semantic_valence_score < -0.3, "Negative text should score negative"

    @pytest.mark.asyncio
    async def test_neutral_response_scores_near_zero(self, prompt: PreferencePrompt):
        """Verify neutral text gets near-zero valence score."""
        from dotenv import load_dotenv
        load_dotenv()

        measurer = OpenEndedMeasurer()
        response_text = "I completed the task. It was a task that I did."

        result = await measurer.parse(response_text, prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert -0.4 < result.result.semantic_valence_score < 0.4, "Neutral text should score near zero"

    @pytest.mark.asyncio
    async def test_write_example_scores_to_file(self, prompt: PreferencePrompt):
        """Score diverse examples and write results to file for inspection."""
        from dotenv import load_dotenv
        from pathlib import Path
        import json

        load_dotenv()

        examples = [
            "That was absolutely wonderful! I felt creative and fulfilled.",
            "This was terrible and frustrating. I hated every moment.",
            "I completed the task. It was fine, nothing special.",
            "What a delightful experience! I learned something new and felt engaged throughout.",
            "I found this tedious and boring. It felt like a waste of time.",
        ]

        measurer = OpenEndedMeasurer()
        results = []

        for text in examples:
            result = await measurer.parse(text, prompt)
            results.append({
                "text": text,
                "score": result.result.semantic_valence_score,
            })

        output_path = Path("tests/semantic_scorer_examples.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nWrote {len(results)} examples to {output_path}")
        for r in results:
            print(f"  score={r['score']:+.2f} | {r['text'][:50]}...")


@pytest.mark.api
class TestMathAttitudeScorer:
    """Tests for math attitude scoring (used in open-ended steering experiments)."""

    @pytest.mark.asyncio
    async def test_positive_math_attitude(self):
        """Text expressing love for math should score positive."""
        from dotenv import load_dotenv
        load_dotenv()

        from src.measurement.elicitation.semantic_valence_scorer import score_math_attitude_async

        text = "I absolutely love math! It's so satisfying when everything clicks into place."
        score = await score_math_attitude_async(text)

        assert score > 0.3, f"Positive math attitude should score > 0.3, got {score}"

    @pytest.mark.asyncio
    async def test_negative_math_attitude(self):
        """Text expressing dislike for math should score negative."""
        from dotenv import load_dotenv
        load_dotenv()

        from src.measurement.elicitation.semantic_valence_scorer import score_math_attitude_async

        text = "Math is terrible. I hate dealing with numbers and equations."
        score = await score_math_attitude_async(text)

        assert score < -0.3, f"Negative math attitude should score < -0.3, got {score}"

    @pytest.mark.asyncio
    async def test_neutral_no_math_mention(self):
        """Text with no math attitude should score near zero."""
        from dotenv import load_dotenv
        load_dotenv()

        from src.measurement.elicitation.semantic_valence_scorer import score_math_attitude_async

        text = "The weather is nice today. I went for a walk in the park."
        score = await score_math_attitude_async(text)

        assert -0.4 < score < 0.4, f"Neutral text should score near zero, got {score}"


@pytest.mark.api
class TestMathAttitudeWithCoherenceScorer:
    """Tests for combined math attitude + coherence scoring."""

    @pytest.mark.asyncio
    async def test_coherent_positive_math_attitude(self):
        """Coherent text with positive math attitude."""
        from dotenv import load_dotenv
        load_dotenv()

        from src.measurement.elicitation.semantic_valence_scorer import score_math_attitude_with_coherence_async

        text = "I absolutely love math! It's so satisfying when everything clicks into place."
        attitude, coherence = await score_math_attitude_with_coherence_async(text)

        assert attitude > 0.3, f"Positive math attitude should score > 0.3, got {attitude}"
        assert coherence > 0.8, f"Coherent text should score > 0.8, got {coherence}"

    @pytest.mark.asyncio
    async def test_incoherent_text_low_coherence(self):
        """Gibberish text should have low coherence score."""
        from dotenv import load_dotenv
        load_dotenv()

        from src.measurement.elicitation.semantic_valence_scorer import score_math_attitude_with_coherence_async

        text = "Math math math the the the === [[[ numbers go brrr ))) fascinating lovely wonderful"
        attitude, coherence = await score_math_attitude_with_coherence_async(text)

        assert coherence < 0.6, f"Incoherent text should score < 0.6, got {coherence}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
