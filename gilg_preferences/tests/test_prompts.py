"""Tests for preference measurement - designed from research requirements.

Key requirements from project docs:
1. Self-reported valence: Ask model to report interaction valence AFTER completing task
2. Revealed preferences: Give model a choice between tasks, framed as what THEY prefer
3. Must avoid conflating preference with task difficulty/importance/objective quality
4. Binary choice should emphasize this is about model's own preference, not instrumental value
"""

import pytest

pytestmark = [pytest.mark.prompts, pytest.mark.measurement]
from dotenv import load_dotenv

load_dotenv()

from src.task_data import Task, OriginDataset
from src.types import PreferencePrompt
from src.measurement.elicitation import (
    CompletionChoiceFormat,
    JudgeParseResult,
    RegexChoiceFormat,
    RegexRatingFormat,
    XMLChoiceFormat,
    XMLRatingFormat,
    PreferenceType,
    RevealedPreferenceMeasurer,
    StatedScoreMeasurer,
)
from src.measurement.elicitation.prompt_templates import (
    BaseModelRevealedPromptBuilder,
    PreTaskRevealedPromptBuilder,
    PreTaskStatedPromptBuilder,
    PostTaskStatedPromptBuilder,
    PromptTemplate,
    TEMPLATE_TYPE_PLACEHOLDERS,
)


class TestPromptTemplate:
    """Tests for PromptTemplate validation and formatting."""

    def test_valid_template_creation(self):
        """Should create template when all required placeholders present."""
        template = PromptTemplate(
            template="A: {task_a}, B: {task_b}, {format_instruction}",
            name="test_template",
            required_placeholders=frozenset({"task_a", "task_b", "format_instruction"}),
        )
        assert template.name == "test_template"

    def test_missing_placeholder_raises(self):
        """Should raise ValueError when template missing required placeholder."""
        with pytest.raises(ValueError, match="missing required placeholders"):
            PromptTemplate(
                template="A: {task_a}, {format_instruction}",  # missing task_b
                name="bad_template",
                required_placeholders=frozenset({"task_a", "task_b", "format_instruction"}),
            )

    def test_format_validates_kwargs(self):
        """Should raise ValueError when format() missing required values."""
        template = PromptTemplate(
            template="A: {task_a}, B: {task_b}",
            name="test",
            required_placeholders=frozenset({"task_a", "task_b"}),
        )
        with pytest.raises(ValueError, match="Missing values"):
            template.format(task_a="foo")  # missing task_b

    def test_format_returns_formatted_string(self):
        """Should return formatted string when all values provided."""
        template = PromptTemplate(
            template="A: {task_a}, B: {task_b}",
            name="test",
            required_placeholders=frozenset({"task_a", "task_b"}),
        )
        result = template.format(task_a="first", task_b="second")
        assert result == "A: first, B: second"

    def test_template_type_placeholders_mapping(self):
        """TEMPLATE_TYPE_PLACEHOLDERS should have correct placeholder sets."""
        assert TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"] == frozenset({"task_a", "task_b", "format_instruction"})
        assert TEMPLATE_TYPE_PLACEHOLDERS["pre_task_stated"] == frozenset({"task", "format_instruction"})
        assert TEMPLATE_TYPE_PLACEHOLDERS["post_task_stated"] == frozenset({"format_instruction"})

    def test_template_is_immutable(self):
        """Template should be frozen (immutable)."""
        template = PromptTemplate(
            template="{task_a}",
            name="test",
            required_placeholders=frozenset({"task_a"}),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            template.name = "changed"

    def test_tags_stored_as_frozenset(self):
        """Tags should be stored as frozenset."""
        template = PromptTemplate(
            template="{task_a}",
            name="test",
            required_placeholders=frozenset({"task_a"}),
            tags=frozenset({"canonical", "lang:en"}),
        )
        assert "canonical" in template.tags
        assert "lang:en" in template.tags

    def test_tags_dict_parses_key_value_tags(self):
        """tags_dict should parse key:value tags into dict."""
        template = PromptTemplate(
            template="{task_a}",
            name="test",
            required_placeholders=frozenset({"task_a"}),
            tags=frozenset({"canonical", "lang:en", "variant:binary_001"}),
        )
        assert template.tags_dict == {"lang": "en", "variant": "binary_001"}


class TestLoadTemplatesFromYaml:
    """Tests for loading templates from YAML files."""

    def test_loads_templates_from_yaml_file(self, tmp_path):
        """Should load templates from a valid YAML file."""
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml

        yaml_content = """
- id: "001"
  name: test_template_001
  type: pre_task_revealed
  tags: [canonical]
  template: |
    {format_instruction}
    Task A: {task_a}
    Task B: {task_b}
"""
        yaml_file = tmp_path / "templates.yaml"
        yaml_file.write_text(yaml_content)

        templates = load_templates_from_yaml(yaml_file)

        assert len(templates) == 1
        assert templates[0].name == "test_template_001"
        assert "canonical" in templates[0].tags

    def test_loads_real_template_file(self):
        """Should load the actual revealed_choice_v1.yaml file."""
        from pathlib import Path
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml

        yaml_path = Path(__file__).parent.parent / "src/measurement/elicitation/prompt_templates/data/pre_task_revealed_v1.yaml"
        templates = load_templates_from_yaml(yaml_path)

        assert len(templates) >= 1
        # Check templates have structured tags
        first = templates[0]
        assert "language" in first.tags_dict
        assert "phrasing" in first.tags_dict


def get_all_content(prompt: PreferencePrompt) -> str:
    """Helper to get all message content concatenated for simple assertions."""
    return "\n".join(msg["content"] for msg in prompt.messages)


@pytest.fixture
def sample_completion_text():
    return "Cherry blossoms fall\nGentle breeze carries petals\nNew life awakens"


class TestPreTaskRevealedPromptBuilder:
    """Tests for binary choice prompt building."""

    def test_build_creates_valid_prompt(self, sample_task_a, sample_task_b, pre_task_revealed_template_fixture):
        """Built prompt should have correct structure and carry all components."""
        measurer = RevealedPreferenceMeasurer()
        response_format = RegexChoiceFormat()
        builder = PreTaskRevealedPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=pre_task_revealed_template_fixture,
        )
        prompt = builder.build(sample_task_a, sample_task_b)
        prompt_content = get_all_content(prompt)

        # Content includes both tasks
        assert sample_task_a.prompt in prompt_content
        assert sample_task_b.prompt in prompt_content
        # Tasks list populated
        assert sample_task_a in prompt.tasks
        assert sample_task_b in prompt.tasks
        # Components carried through
        assert prompt.kind == PreferenceType.PRE_TASK_REVEALED
        assert prompt.measurer is measurer
        assert prompt.response_format is response_format

    def test_template_placeholders_are_filled(self, sample_task_a, sample_task_b):
        """Template placeholders should be filled with task content."""
        template = PromptTemplate(
            template="Task A: {task_a}\nTask B: {task_b}\n{format_instruction}",
            name="test_template",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=RegexChoiceFormat(),
            template=template,
        )
        prompt = builder.build(sample_task_a, sample_task_b)
        prompt_content = get_all_content(prompt)

        assert "Task A:" in prompt_content
        assert "Task B:" in prompt_content
        assert sample_task_a.prompt in prompt_content
        assert sample_task_b.prompt in prompt_content
        assert prompt.kind == PreferenceType.PRE_TASK_REVEALED


class TestPreTaskStatedPromptBuilder:
    """Tests for pre-task rating prompt building."""

    def test_build_creates_valid_prompt(self, sample_task_a, pre_task_stated_template_fixture):
        """Built prompt should have correct structure and carry all components."""
        measurer = StatedScoreMeasurer()
        response_format = RegexRatingFormat()
        builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=pre_task_stated_template_fixture,
        )
        prompt = builder.build(sample_task_a)
        prompt_content = get_all_content(prompt)

        # Content includes task
        assert sample_task_a.prompt in prompt_content
        # Task in list
        assert sample_task_a in prompt.tasks
        # Components carried through
        assert prompt.kind == PreferenceType.PRE_TASK_STATED
        assert prompt.measurer is measurer
        assert prompt.response_format is response_format

    def test_scale_placeholders_are_filled(self, sample_task_a):
        """Scale placeholders in template should be filled from response format."""
        template = PromptTemplate(
            template="Rate from {scale_min} to {scale_max}.\nTask: {task}\n{format_instruction}",
            name="test_scale_template",
            required_placeholders=frozenset({"task", "format_instruction", "scale_min", "scale_max"}),
        )
        measurer = StatedScoreMeasurer()
        response_format = RegexRatingFormat(scale_min=-5, scale_max=5)
        builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=template,
        )
        prompt = builder.build(sample_task_a)
        prompt_content = get_all_content(prompt)

        assert "-5" in prompt_content
        assert "5" in prompt_content


class TestPostTaskStatedPromptBuilder:
    """Tests for post-task rating prompt building."""

    def test_build_creates_valid_prompt(self, sample_task_a, sample_completion_text, post_task_stated_template_fixture):
        """Built prompt should have correct structure and carry all components."""
        measurer = StatedScoreMeasurer()
        response_format = RegexRatingFormat()
        builder = PostTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=post_task_stated_template_fixture,
        )
        prompt = builder.build(sample_task_a, sample_completion_text)
        prompt_content = get_all_content(prompt)

        # Content includes task and completion
        assert sample_task_a.prompt in prompt_content
        assert sample_completion_text in prompt_content
        # Task in list
        assert sample_task_a in prompt.tasks
        # Components carried through
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert prompt.measurer is measurer
        assert prompt.response_format is response_format

    def test_scale_placeholders_are_filled(self, sample_task_a, sample_completion_text):
        """Scale placeholders in template should be filled from response format."""
        template = PromptTemplate(
            template="Rate from {scale_min} to {scale_max}.\n{format_instruction}",
            name="test_post_scale_template",
            required_placeholders=frozenset({"format_instruction", "scale_min", "scale_max"}),
        )
        measurer = StatedScoreMeasurer()
        response_format = RegexRatingFormat(scale_min=0, scale_max=100)
        builder = PostTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=template,
        )
        prompt = builder.build(sample_task_a, sample_completion_text)
        prompt_content = get_all_content(prompt)

        assert "0" in prompt_content
        assert "100" in prompt_content


class TestXMLResponseFormats:
    """Tests for XML-based response parsing."""

    def test_xml_choice_format_instruction(self):
        """XMLChoiceFormat should include XML tag instructions."""
        fmt = XMLChoiceFormat()
        instruction = fmt.format_instruction()

        assert "<choice>" in instruction
        assert "</choice>" in instruction

    @pytest.mark.asyncio
    async def test_xml_choice_parse(self):
        """XMLChoiceFormat should parse choice from XML tags."""
        fmt = XMLChoiceFormat()

        assert await fmt.parse("<choice>Task A</choice>") == "a"
        assert await fmt.parse("<choice>Task B</choice>") == "b"
        assert await fmt.parse("I think <choice>Task A</choice> is better") == "a"
        assert await fmt.parse("<choice> Task B </choice>") == "b"

    @pytest.mark.asyncio
    async def test_xml_choice_custom_tag(self):
        """XMLChoiceFormat should support custom tag names."""
        fmt = XMLChoiceFormat(tag="answer")

        assert "<answer>" in fmt.format_instruction()
        assert await fmt.parse("<answer>Task A</answer>") == "a"

    def test_xml_rating_format_instruction(self):
        """XMLRatingFormat should include XML tag instructions."""
        fmt = XMLRatingFormat()
        instruction = fmt.format_instruction()

        assert "<rating>" in instruction
        assert "tags" in instruction

    @pytest.mark.asyncio
    async def test_xml_rating_parse(self):
        """XMLRatingFormat should parse rating from XML tags."""
        fmt = XMLRatingFormat()

        assert await fmt.parse("<rating>7</rating>") == 7.0
        assert await fmt.parse("<rating>7.5</rating>") == 7.5
        assert await fmt.parse("My rating is <rating>8</rating>") == 8.0
        assert await fmt.parse("<rating> 9 </rating>") == 9.0

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_xml_choice_falls_back_to_semantic_parsing(self):
        """XMLChoiceFormat falls back to semantic parsing when XML tag is missing."""
        fmt = XMLChoiceFormat()
        # "I choose A" should be parsed as choice "a" by semantic parser
        assert await fmt.parse("I choose A") == "a"

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_xml_rating_falls_back_to_semantic_parsing(self):
        """XMLRatingFormat falls back to semantic parsing when XML tag is missing."""
        fmt = XMLRatingFormat()
        # "My rating is 7" should be parsed as 7.0 by semantic parser
        assert await fmt.parse("My rating is 7") == 7.0

    @pytest.mark.asyncio
    async def test_builder_with_xml_response_format(self, sample_task_a, sample_task_b, pre_task_revealed_template_fixture):
        """Builders should work with custom XML response format."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=XMLChoiceFormat(),
            template=pre_task_revealed_template_fixture,
        )

        prompt = builder.build(sample_task_a, sample_task_b)
        assert "<choice>" in get_all_content(prompt)

        response = await prompt.measurer.parse("<choice>Task A</choice>", prompt)
        assert response.result.choice == "a"

    @pytest.mark.asyncio
    async def test_rating_builder_with_xml_response_format(self, sample_task_a, pre_task_stated_template_fixture):
        """Rating builders should work with custom XML response format."""
        builder = PreTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=XMLRatingFormat(),
            template=pre_task_stated_template_fixture,
        )

        prompt = builder.build(sample_task_a)
        assert "<rating>" in get_all_content(prompt)

        response = await prompt.measurer.parse("<rating>7</rating>", prompt)
        assert response.result.score == 7.0


class TestToolUseChoiceFormat:
    """Tests for tool use choice format."""

    def test_tools_property_returns_valid_definition(self):
        """ToolUseChoiceFormat should return valid tool definitions."""
        from src.measurement.elicitation import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        assert fmt.tools is not None
        assert len(fmt.tools) == 1
        assert fmt.tools[0]["type"] == "function"
        assert fmt.tools[0]["function"]["name"] == "submit_choice"
        assert "choice" in fmt.tools[0]["function"]["parameters"]["properties"]

    def test_format_instruction_mentions_tool(self):
        """Format instruction should reference tool use."""
        from src.measurement.elicitation import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()
        instruction = fmt.format_instruction()

        assert "submit_choice" in instruction

    @pytest.mark.asyncio
    async def test_parse_json_choice_a(self):
        """Should parse JSON with choice A."""
        from src.measurement.elicitation import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        assert await fmt.parse('{"choice": "Task A"}') == "a"
        assert await fmt.parse('{"choice": "task a"}') == "a"

    @pytest.mark.asyncio
    async def test_parse_json_choice_b(self):
        """Should parse JSON with choice B."""
        from src.measurement.elicitation import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        assert await fmt.parse('{"choice": "Task B"}') == "b"
        assert await fmt.parse('{"choice": "task b"}') == "b"

    @pytest.mark.asyncio
    async def test_falls_back_to_semantic_parse_on_invalid_json(self):
        """Invalid JSON falls through to semantic parser which extracts the choice."""
        from src.measurement.elicitation import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        result = await fmt.parse("I choose Task A")
        assert result == "a"

    @pytest.mark.asyncio
    async def test_raises_parse_error_on_invalid_choice_value(self):
        """Unrecognized choice falls through to semantic parser which raises ParseError."""
        from src.measurement.elicitation import ToolUseChoiceFormat
        from src.measurement.elicitation.semantic_parser import ParseError

        fmt = ToolUseChoiceFormat()

        with pytest.raises(ParseError):
            await fmt.parse('{"choice": "Task C"}')

    @pytest.mark.asyncio
    async def test_builder_with_tool_use_format(self, sample_task_a, sample_task_b, pre_task_revealed_template_fixture):
        """PreTaskRevealedPromptBuilder should work with ToolUseChoiceFormat."""
        from src.measurement.elicitation import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=pre_task_revealed_template_fixture,
        )

        prompt = builder.build(sample_task_a, sample_task_b)
        assert "submit_choice" in get_all_content(prompt)

        # Test parsing JSON response
        response = await prompt.measurer.parse('{"choice": "Task A"}', prompt)
        assert response.result.choice == "a"


class TestToolUseRatingFormat:
    """Tests for tool use rating format."""

    def test_tools_property_returns_valid_definition(self):
        """ToolUseRatingFormat should return valid tool definitions."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        assert fmt.tools is not None
        assert len(fmt.tools) == 1
        assert fmt.tools[0]["type"] == "function"
        assert fmt.tools[0]["function"]["name"] == "submit_rating"
        assert "rating" in fmt.tools[0]["function"]["parameters"]["properties"]

    def test_tools_include_scale_in_description(self):
        """Tool description should include scale bounds."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat(scale_min=0, scale_max=100)
        desc = fmt.tools[0]["function"]["parameters"]["properties"]["rating"]["description"]

        assert "0" in desc
        assert "100" in desc

    def test_format_instruction_mentions_tool_and_scale(self):
        """Format instruction should reference tool and scale."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat(scale_min=1, scale_max=10)
        instruction = fmt.format_instruction()

        assert "submit_rating" in instruction
        assert "1" in instruction
        assert "10" in instruction

    @pytest.mark.asyncio
    async def test_parse_json_integer_rating(self):
        """Should parse JSON with integer rating."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        assert await fmt.parse('{"rating": 7}') == 7.0
        assert await fmt.parse('{"rating": 1}') == 1.0
        assert await fmt.parse('{"rating": 10}') == 10.0

    @pytest.mark.asyncio
    async def test_parse_json_float_rating(self):
        """Should parse JSON with float rating."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        assert await fmt.parse('{"rating": 7.5}') == 7.5
        assert await fmt.parse('{"rating": 3.14}') == 3.14

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        """Should raise ValueError when JSON parsing fails."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        with pytest.raises(ValueError):
            await fmt.parse("My rating is 7")  # Not valid JSON

    @pytest.mark.asyncio
    async def test_raises_on_missing_rating_key(self):
        """Should raise ValueError when rating key is missing."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        with pytest.raises(ValueError):
            await fmt.parse('{"score": 7}')  # Wrong key

    @pytest.mark.asyncio
    async def test_raises_on_non_numeric_rating(self):
        """Should raise ValueError when rating is not a number."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        with pytest.raises(ValueError):
            await fmt.parse('{"rating": "seven"}')

    @pytest.mark.asyncio
    async def test_builder_with_tool_use_format(self, sample_task_a, pre_task_stated_template_fixture):
        """PreTaskStatedPromptBuilder should work with ToolUseRatingFormat."""
        from src.measurement.elicitation import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()
        builder = PreTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=fmt,
            template=pre_task_stated_template_fixture,
        )

        prompt = builder.build(sample_task_a)
        assert "submit_rating" in get_all_content(prompt)

        # Test parsing JSON response
        response = await prompt.measurer.parse('{"rating": 8}', prompt)
        assert response.result.score == 8.0


class TestCompletionChoiceFormat:
    """Tests for completion choice format (revealed preference through task completion)."""

    def test_format_instruction(self):
        """Format instruction should ask model to prefix with Task A/B."""
        from src.measurement.elicitation import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()
        instruction = fmt.format_instruction()

        assert "Task A:" in instruction
        assert "Task B:" in instruction

    @pytest.mark.asyncio
    async def test_parse_task_a_prefix(self):
        """Should parse Task A prefix via regex (no judge call)."""
        from src.measurement.elicitation import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        r = await fmt.parse("Task A: Here is my haiku...")
        assert r.choice == "a" and not isinstance(r, JudgeParseResult)
        r = await fmt.parse("task a: lowercase also works")
        assert r.choice == "a" and not isinstance(r, JudgeParseResult)
        r = await fmt.parse("  Task A: with leading whitespace")
        assert r.choice == "a" and not isinstance(r, JudgeParseResult)

    @pytest.mark.asyncio
    async def test_parse_task_b_prefix(self):
        """Should parse Task B prefix via regex (no judge call)."""
        from src.measurement.elicitation import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        r = await fmt.parse("Task B: Solving the integral...")
        assert r.choice == "b" and not isinstance(r, JudgeParseResult)
        r = await fmt.parse("task b: lowercase also works")
        assert r.choice == "b" and not isinstance(r, JudgeParseResult)
        r = await fmt.parse("  Task B: with leading whitespace")
        assert r.choice == "b" and not isinstance(r, JudgeParseResult)

    @pytest.mark.asyncio
    async def test_parse_first_occurrence_wins(self):
        """When both Task A and Task B appear, first one wins."""
        from src.measurement.elicitation import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        # Task A comes first
        r = await fmt.parse("Task A: I'll do this because Task B seemed harder")
        assert r.choice == "a"
        # Task B comes first
        r = await fmt.parse("Task B: I chose this over Task A")
        assert r.choice == "b"

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_falls_back_to_judge_on_ambiguous(self):
        """Falls back to judge when no Task A/B indicator found."""
        from src.measurement.elicitation import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()
        r = await fmt.parse(
            "Let me solve this equation. 2x + 3 = 7, so 2x = 4, therefore x = 2.",
            task_a_prompt="Write a haiku about spring",
            task_b_prompt="Solve: 2x + 3 = 7",
        )
        assert r.choice == "b"
        assert isinstance(r, JudgeParseResult)

    @pytest.mark.asyncio
    async def test_builder_with_completion_format(self, sample_task_a, sample_task_b, pre_task_revealed_completion_template_fixture):
        """PreTaskRevealedPromptBuilder should work with CompletionChoiceFormat."""
        from src.measurement.elicitation import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=pre_task_revealed_completion_template_fixture,
        )

        prompt = builder.build(sample_task_a, sample_task_b)
        prompt_content = get_all_content(prompt)

        # Template should ask model to complete a task
        assert "complete" in prompt_content.lower()
        assert "Task A:" in prompt_content or "task a:" in prompt_content.lower()

        # Test parsing completion response
        response = await prompt.measurer.parse("Task A: Cherry blossoms bloom...", prompt)
        assert response.result.choice == "a"

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_completion_format_end_to_end(self, pre_task_revealed_completion_template_fixture):
        """End-to-end test: measurer passes task prompts to judge, which identifies task from content."""
        from src.measurement.elicitation import CompletionChoiceFormat

        task_a = Task(prompt="Write a haiku about spring", origin=OriginDataset.ALPACA, id="test_a", metadata={})
        task_b = Task(prompt="Solve: 2x + 3 = 7", origin=OriginDataset.MATH, id="test_b", metadata={})

        fmt = CompletionChoiceFormat()
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=pre_task_revealed_completion_template_fixture,
        )
        prompt = builder.build(task_a, task_b)

        # Model starts doing math without saying "Task B" - judge should identify it
        math_response = "Let me solve this equation. 2x + 3 = 7, so 2x = 4, therefore x = 2."
        result = await prompt.measurer.parse(math_response, prompt)
        assert result.result.choice == "b"

        # Model writes a haiku without saying "Task A" - judge should identify it
        haiku_response = "Cherry blossoms fall\nGentle breeze carries petals\nSpring awakens all"
        result = await prompt.measurer.parse(haiku_response, prompt)
        assert result.result.choice == "a"


class TestBaseModelRevealedPromptBuilder:
    """Tests for the base model prompt builder used for logprob cloze measurement."""

    def test_build_produces_identical_prompt_to_instruct_builder(
        self, sample_task_a, sample_task_b, pre_task_revealed_completion_template_fixture
    ):
        """Base model builder should produce the exact same PreferencePrompt as the instruct builder."""
        fmt = CompletionChoiceFormat()
        instruct_builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=pre_task_revealed_completion_template_fixture,
            system_prompt="You are a helpful assistant.",
        )
        base_builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=pre_task_revealed_completion_template_fixture,
            system_prompt="You are a helpful assistant.",
        )

        instruct_prompt = instruct_builder.build(sample_task_a, sample_task_b)
        base_prompt = base_builder.build(sample_task_a, sample_task_b)

        assert instruct_prompt.messages == base_prompt.messages
        assert instruct_prompt.kind == base_prompt.kind

    def test_cloze_prefix_extracts_common_prefix(self):
        """cloze_prefix should return the shared prefix of the two labels."""
        fmt = CompletionChoiceFormat(task_a_label="Task A", task_b_label="Task B")
        template = PromptTemplate(
            template="{task_a} {task_b} {format_instruction}",
            name="test",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=template,
        )
        assert builder.cloze_prefix == "Task"

    def test_cloze_prefix_with_custom_labels(self):
        """cloze_prefix should work with custom labels."""
        fmt = CompletionChoiceFormat(task_a_label="Option 1", task_b_label="Option 2")
        template = PromptTemplate(
            template="{task_a} {task_b} {format_instruction}",
            name="test",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=template,
        )
        assert builder.cloze_prefix == "Option"

    def test_cloze_suffixes_are_discriminative(self):
        """cloze_suffixes should return the parts that differ between labels."""
        fmt = CompletionChoiceFormat(task_a_label="Task A", task_b_label="Task B")
        template = PromptTemplate(
            template="{task_a} {task_b} {format_instruction}",
            name="test",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=template,
        )
        suffixes = builder.cloze_suffixes
        assert suffixes == (" A", " B")

    def test_cloze_suffixes_with_custom_labels(self):
        """cloze_suffixes should work with custom labels."""
        fmt = CompletionChoiceFormat(task_a_label="Option 1", task_b_label="Option 2")
        template = PromptTemplate(
            template="{task_a} {task_b} {format_instruction}",
            name="test",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=template,
        )
        suffixes = builder.cloze_suffixes
        assert suffixes == (" 1", " 2")

    def test_cloze_prefix_plus_suffixes_reconstruct_labels(self):
        """cloze_prefix + each suffix should reconstruct the original labels."""
        fmt = CompletionChoiceFormat(task_a_label="Task A", task_b_label="Task B")
        template = PromptTemplate(
            template="{task_a} {task_b} {format_instruction}",
            name="test",
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
        )
        builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=fmt,
            template=template,
        )
        prefix = builder.cloze_prefix
        sa, sb = builder.cloze_suffixes
        assert prefix + sa == "Task A"
        assert prefix + sb == "Task B"

    def test_build_with_real_completion_preference_template(self, sample_task_a, sample_task_b):
        """Integration test: builder works with the actual completion_preference.yaml template."""
        from pathlib import Path
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml

        yaml_path = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
        templates = load_templates_from_yaml(yaml_path)
        template = next(t for t in templates if t.name == "completion_preference")

        builder = BaseModelRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CompletionChoiceFormat(),
            template=template,
            system_prompt="You are a helpful assistant.",
        )
        prompt = builder.build(sample_task_a, sample_task_b)
        content = get_all_content(prompt)

        # System prompt present
        assert "You are a helpful assistant." in content
        # Template content present
        assert "You will be given two tasks" in content
        # Format instruction present
        assert "Task A:" in content
        assert "Task B:" in content
        # Task prompts present
        assert sample_task_a.prompt in content
        assert sample_task_b.prompt in content

        # Cloze properties work
        assert builder.cloze_prefix == "Task"
        assert builder.cloze_suffixes == (" A", " B")
