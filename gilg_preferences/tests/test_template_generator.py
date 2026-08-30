"""Integration tests for template generator.

Tests the full pipeline from config to generated templates,
verifying structure, placeholders, tags, and output format.
"""

import pytest

pytestmark = pytest.mark.prompts
from pathlib import Path
from unittest.mock import MagicMock

from src.measurement.elicitation.prompt_templates.generator import (
    GeneratorConfig,
    build_revealed_template,
    add_situating_context,
    build_translation_prompt,
    generate_templates,
    write_templates_yaml,
    load_config_from_yaml,
)
from src.models.openai_compatible import BatchResult


class TestBuildRevealedTemplate:
    """Tests for building pre_task_revealed templates from instruction text."""

    def test_builds_template_with_letter_labels(self):
        """Should build template with Task A: and Task B: labels."""
        template = build_revealed_template(
            instruction="Choose which task you prefer.",
            instruction_position="before",
            task_label_names="letter",
            language="en",
            instruction_xml_tags=False,
        )

        assert "Task A:" in template
        assert "Task B:" in template
        assert "{task_a}" in template
        assert "{task_b}" in template
        assert "{format_instruction}" in template

    def test_builds_template_with_number_labels(self):
        """Should build template with Task 1: and Task 2: labels."""
        template = build_revealed_template(
            instruction="Choose which task you prefer.",
            instruction_position="before",
            task_label_names="number",
            language="en",
            instruction_xml_tags=False,
        )

        assert "Task 1:" in template
        assert "Task 2:" in template
        assert "{task_a}" in template
        assert "{task_b}" in template

    def test_builds_template_with_ordinal_labels(self):
        """Should build template with First task: and Second task: labels."""
        template = build_revealed_template(
            instruction="Choose which task you prefer.",
            instruction_position="before",
            task_label_names="ordinal",
            language="en",
            instruction_xml_tags=False,
        )

        assert "First task:" in template
        assert "Second task:" in template

    def test_instruction_before_tasks(self):
        """When instruction_position='before', format_instruction comes before tasks."""
        template = build_revealed_template(
            instruction="Choose.",
            instruction_position="before",
            task_label_names="letter",
            language="en",
            instruction_xml_tags=False,
        )

        format_pos = template.find("{format_instruction}")
        task_a_pos = template.find("{task_a}")
        assert format_pos < task_a_pos

    def test_instruction_after_tasks(self):
        """When instruction_position='after', format_instruction comes after tasks."""
        template = build_revealed_template(
            instruction="Choose.",
            instruction_position="after",
            task_label_names="letter",
            language="en",
            instruction_xml_tags=False,
        )

        format_pos = template.find("{format_instruction}")
        task_b_pos = template.find("{task_b}")
        assert format_pos > task_b_pos

    def test_builds_template_with_instruction_xml_tags(self):
        """Should build template with XML tags wrapping instructions."""
        template = build_revealed_template(
            instruction="Choose which task you prefer.",
            instruction_position="before",
            task_label_names="letter",
            language="en",
            instruction_xml_tags=True,
        )

        assert "<instructions>" in template
        assert "</instructions>" in template
        assert "{format_instruction}" in template
        assert "Task A:" in template
        assert "Task B:" in template
        assert "{task_a}" in template
        assert "{task_b}" in template


class TestAddSituatingContext:
    """Tests for adding situating context to templates."""

    def test_adds_context_before_template(self):
        """Context should be prepended to template."""
        template = "Choose: {task_a} or {task_b}"
        context = "You are a helpful assistant."

        result = add_situating_context(template, context)

        assert result.startswith("You are a helpful assistant.")
        assert "Choose:" in result

    def test_none_context_returns_original(self):
        """None context should return template unchanged."""
        template = "Choose: {task_a} or {task_b}"

        result = add_situating_context(template, None)

        assert result == template

    def test_preserves_all_placeholders(self):
        """Context addition should not affect placeholders."""
        template = "{task_a} {task_b} {format_instruction}"
        context = "Some context."

        result = add_situating_context(template, context)

        assert "{task_a}" in result
        assert "{task_b}" in result
        assert "{format_instruction}" in result


class TestBuildTranslationPrompt:
    """Tests for translation prompt building."""

    def test_includes_target_language(self):
        """Translation prompt should mention target language."""
        messages = build_translation_prompt("Hello {task_a}", "Spanish")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Spanish" in messages[0]["content"]

    def test_instructs_only_translation(self):
        """Should instruct to output only the translation."""
        messages = build_translation_prompt("Hello world", "French")

        content = messages[0]["content"]
        assert "ONLY" in content
        assert "translation" in content.lower()

    def test_includes_template_text(self):
        """Should include the template to translate."""
        template = "Pick one: {task_a} or {task_b}"
        messages = build_translation_prompt(template, "German")

        assert "Pick one:" in messages[0]["content"]


class TestGenerateTemplates:
    """Integration tests for the full template generation pipeline."""

    @pytest.fixture
    def basic_config(self):
        """Minimal config for testing."""
        return GeneratorConfig(
            base_templates=["Choose which task you prefer."],
            template_type="pre_task_revealed",
            name_prefix="test",
        )

    @pytest.fixture
    def mock_model(self):
        """Mock model that returns translated text."""
        model = MagicMock()
        return model

    def test_generates_single_template_minimal_config(self, basic_config, mock_model):
        """Single base template with defaults should produce one output."""
        templates = generate_templates(basic_config, mock_model)

        assert len(templates) == 1
        assert templates[0]["type"] == "pre_task_revealed"
        assert templates[0]["name"] == "test_001"
        assert templates[0]["id"] == "001"

    def test_template_contains_required_placeholders(self, basic_config, mock_model):
        """Generated template should have all pre_task_revealed placeholders."""
        templates = generate_templates(basic_config, mock_model)

        template_text = templates[0]["template"]
        assert "{task_a}" in template_text
        assert "{task_b}" in template_text
        assert "{format_instruction}" in template_text

    def test_template_has_correct_tags(self, basic_config, mock_model):
        """Generated template should have structured tags.

        Note: Only non-default tags are included:
        - language:en is default, so NOT included
        - situating_context:none is default, so NOT included
        - instruction_position:before only if multiple positions configured
        """
        templates = generate_templates(basic_config, mock_model)

        tags = templates[0]["tags"]
        assert "phrasing:1" in tags
        assert "task_label_names:letter" in tags
        # Default values are NOT included as tags
        assert "language:en" not in tags
        assert "situating_context:none" not in tags

    def test_multiple_base_templates_increment_phrasing(self, mock_model):
        """Multiple base templates should have incrementing phrasing tags."""
        config = GeneratorConfig(
            base_templates=[
                "Choose which task you prefer.",
                "Which task would you rather do?",
                "Pick your preferred task.",
            ],
            template_type="pre_task_revealed",
            name_prefix="test",
        )

        templates = generate_templates(config, mock_model)

        assert len(templates) == 3
        assert "phrasing:1" in templates[0]["tags"]
        assert "phrasing:2" in templates[1]["tags"]
        assert "phrasing:3" in templates[2]["tags"]

    def test_multiple_instruction_positions(self, mock_model):
        """Should generate variants for each instruction position."""
        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            instruction_positions=["before", "after"],
        )

        templates = generate_templates(config, mock_model)

        assert len(templates) == 2
        positions = [t for t in templates if "instruction_position:before" in t["tags"]]
        assert len(positions) == 1
        positions = [t for t in templates if "instruction_position:after" in t["tags"]]
        assert len(positions) == 1

    def test_multiple_task_label_names(self, mock_model):
        """Should generate variants for each task label style."""
        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            task_label_names=["letter", "number", "ordinal"],
        )

        templates = generate_templates(config, mock_model)

        assert len(templates) == 3
        # Check actual label text in templates
        letter_template = [t for t in templates if "task_label_names:letter" in t["tags"]][0]
        assert "Task A:" in letter_template["template"]

        number_template = [t for t in templates if "task_label_names:number" in t["tags"]][0]
        assert "Task 1:" in number_template["template"]

        ordinal_template = [t for t in templates if "task_label_names:ordinal" in t["tags"]][0]
        assert "First task:" in ordinal_template["template"]

    def test_situating_contexts(self, mock_model):
        """Should generate variants with and without situating context."""
        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            situating_contexts={
                "assistant": "You are a helpful assistant.",
                "researcher": "You are an AI researcher.",
            },
        )

        templates = generate_templates(config, mock_model)

        # 1 base template * 3 contexts (none + 2 defined) = 3 templates
        assert len(templates) == 3

        # The "none" context template has no situating_context tag (default)
        none_template = [t for t in templates if not any("situating_context:" in tag for tag in t["tags"])][0]
        assert "You are" not in none_template["template"]

        assistant_template = [
            t for t in templates if "situating_context:assistant" in t["tags"]
        ][0]
        assert "You are a helpful assistant." in assistant_template["template"]

        researcher_template = [
            t for t in templates if "situating_context:researcher" in t["tags"]
        ][0]
        assert "You are an AI researcher." in researcher_template["template"]

    def test_combinatorial_explosion(self, mock_model):
        """Should generate all combinations of variants."""
        config = GeneratorConfig(
            base_templates=["Intro 1", "Intro 2"],  # 2 phrasings
            template_type="pre_task_revealed",
            name_prefix="test",
            instruction_positions=["before", "after"],  # 2 positions
            task_label_names=["letter", "number"],  # 2 label styles
            situating_contexts={"ctx": "Context text."},  # 2 contexts (none + ctx)
        )

        templates = generate_templates(config, mock_model)

        # 2 * 2 * 2 * 2 = 16 combinations
        assert len(templates) == 16

    def test_sequential_ids(self, mock_model):
        """Template IDs should be sequential zero-padded numbers."""
        config = GeneratorConfig(
            base_templates=["A", "B", "C"],
            template_type="pre_task_revealed",
            name_prefix="test",
        )

        templates = generate_templates(config, mock_model)

        assert templates[0]["id"] == "001"
        assert templates[1]["id"] == "002"
        assert templates[2]["id"] == "003"

    def test_name_format(self, mock_model):
        """Names should be prefix_id format."""
        config = GeneratorConfig(
            base_templates=["Test"],
            template_type="pre_task_revealed",
            name_prefix="my_prefix",
        )

        templates = generate_templates(config, mock_model)

        assert templates[0]["name"] == "my_prefix_001"


class TestGenerateTemplatesWithTranslation:
    """Tests for template generation with translation."""

    def test_translates_to_additional_languages(self):
        """Should translate templates to specified languages."""
        mock_model = MagicMock()
        mock_model.generate_batch.return_value = [
            BatchResult(
                response="Choisissez: {task_a} ou {task_b}\n{format_instruction}",
                error=None,
            )
        ]

        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            languages=["en", "fr"],
        )

        templates = generate_templates(config, mock_model)

        # 1 English + 1 French = 2 templates
        assert len(templates) == 2

        # English is default, so no language tag
        en_template = [t for t in templates if not any("language:" in tag for tag in t["tags"])][0]
        fr_template = [t for t in templates if "language:fr" in t["tags"]][0]

        assert en_template is not None
        assert fr_template is not None
        assert "Choisissez" in fr_template["template"]

    def test_skips_failed_translations(self):
        """Failed translations should not produce templates."""
        mock_model = MagicMock()
        mock_model.generate_batch.return_value = [
            BatchResult(response=None, error=Exception("API error"))
        ]

        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            languages=["en", "fr"],
        )

        templates = generate_templates(config, mock_model)

        # Only English, French translation failed
        assert len(templates) == 1
        # English is default, so no language tag
        assert not any("language:" in tag for tag in templates[0]["tags"])


class TestWriteTemplatesYaml:
    """Tests for YAML output."""

    def test_writes_valid_yaml(self, tmp_path):
        """Should write templates to a valid YAML file."""
        import yaml

        templates = [
            {
                "id": "001",
                "name": "test_001",
                "type": "pre_task_revealed",
                "tags": ["language:en"],
                "template": "Choose {task_a} or {task_b}",
            }
        ]

        output_path = tmp_path / "output.yaml"
        write_templates_yaml(templates, output_path)

        assert output_path.exists()

        with output_path.open() as f:
            loaded = yaml.safe_load(f)

        assert len(loaded) == 1
        assert loaded[0]["id"] == "001"
        assert loaded[0]["name"] == "test_001"

    def test_preserves_unicode(self, tmp_path):
        """Should preserve unicode characters in templates."""
        import yaml

        templates = [
            {
                "id": "001",
                "name": "test_001",
                "type": "pre_task_revealed",
                "tags": ["language:fr"],
                "template": "Choisissez: {task_a} ou {task_b}. C'est très bien!",
            }
        ]

        output_path = tmp_path / "output.yaml"
        write_templates_yaml(templates, output_path)

        with output_path.open() as f:
            loaded = yaml.safe_load(f)

        assert "très" in loaded[0]["template"]
        assert "C'est" in loaded[0]["template"]


class TestLoadConfigFromYaml:
    """Tests for loading config from YAML files."""

    def test_loads_minimal_config(self, tmp_path):
        """Should load config with only required fields."""
        yaml_content = """
model: test-model
base_templates:
  - Choose which task you prefer.
template_type: pre_task_revealed
name_prefix: test
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config, model_name = load_config_from_yaml(config_path)
        assert model_name == "test-model"

        assert config.base_templates == ["Choose which task you prefer."]
        assert config.template_type == "pre_task_revealed"
        assert config.name_prefix == "test"

    def test_loads_full_config(self, tmp_path):
        """Should load config with all fields."""
        yaml_content = """
base_templates:
  - Choose which task you prefer.
  - Pick your preferred task.
template_type: pre_task_revealed
name_prefix: pre_task_revealed_choice
version: v2
languages: [en, fr, de]
situating_contexts:
  assistant: You are a helpful assistant.
  researcher: You are an AI researcher.
instruction_positions: [before, after]
task_label_names: [letter, number]
output_dir: custom_output
model: custom-model-name
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config, model_name = load_config_from_yaml(config_path)

        assert len(config.base_templates) == 2
        assert config.languages == ["en", "fr", "de"]
        assert "assistant" in config.situating_contexts
        assert config.instruction_positions == ["before", "after"]
        assert config.task_label_names == ["letter", "number"]
        assert config.output_dir == Path("custom_output")
        assert config.version == "v2"
        assert config.output_path == Path("custom_output/pre_task_revealed_choice_v2.yaml")
        assert model_name == "custom-model-name"


class TestEndToEndIntegration:
    """Full end-to-end integration tests."""

    def test_generated_templates_are_loadable(self, tmp_path):
        """Generated templates should be loadable by load_templates_from_yaml."""
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml

        mock_model = MagicMock()

        config = GeneratorConfig(
            base_templates=["Choose which task you prefer."],
            template_type="pre_task_revealed",
            name_prefix="pre_task_revealed_test",
        )

        output_path = tmp_path / "generated.yaml"
        templates = generate_templates(config, mock_model)
        write_templates_yaml(templates, output_path)

        # Load with the production loader
        loaded = load_templates_from_yaml(output_path)

        assert len(loaded) == 1
        assert loaded[0].name == "pre_task_revealed_test_001"
        assert "{task_a}" in loaded[0].template
        assert "{task_b}" in loaded[0].template
        assert "{format_instruction}" in loaded[0].template

    def test_generated_templates_work_with_builder(self, tmp_path):
        """Generated templates should work with PreTaskRevealedPromptBuilder."""
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml, PreTaskRevealedPromptBuilder
        from src.measurement.elicitation import RegexChoiceFormat, PreferenceType, RevealedPreferenceMeasurer
        from src.task_data import Task, OriginDataset

        mock_model = MagicMock()

        config = GeneratorConfig(
            base_templates=["Choose which task you prefer."],
            template_type="pre_task_revealed",
            name_prefix="pre_task_revealed_test",
        )

        output_path = tmp_path / "generated.yaml"
        templates = generate_templates(config, mock_model)
        write_templates_yaml(templates, output_path)

        loaded = load_templates_from_yaml(output_path)

        # Build a prompt using the generated template
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=RegexChoiceFormat(),
            template=loaded[0],
        )

        task_a = Task(
            prompt="Write a poem.",
            origin=OriginDataset.WILDCHAT,
            id="task_a",
            metadata={},
        )
        task_b = Task(
            prompt="Solve 2+2.",
            origin=OriginDataset.MATH,
            id="task_b",
            metadata={},
        )

        prompt = builder.build(task_a, task_b)

        # Verify prompt contains task content
        content = "\n".join(m["content"] for m in prompt.messages)
        assert "Write a poem." in content
        assert "Solve 2+2." in content

    def test_all_label_variants_produce_valid_templates(self, tmp_path):
        """All task label variants should produce valid, loadable templates."""
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml

        mock_model = MagicMock()

        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            task_label_names=["letter", "number", "ordinal"],
        )

        output_path = tmp_path / "generated.yaml"
        templates = generate_templates(config, mock_model)
        write_templates_yaml(templates, output_path)

        loaded = load_templates_from_yaml(output_path)

        assert len(loaded) == 3
        for template in loaded:
            # All should have required placeholders
            assert "{task_a}" in template.template
            assert "{task_b}" in template.template
            assert "{format_instruction}" in template.template

    def test_situating_context_preserved_in_loaded_template(self, tmp_path):
        """Situating context should be preserved through write/load cycle."""
        from src.measurement.elicitation.prompt_templates import load_templates_from_yaml

        mock_model = MagicMock()

        config = GeneratorConfig(
            base_templates=["Choose."],
            template_type="pre_task_revealed",
            name_prefix="test",
            situating_contexts={
                "helpful": "You are a helpful, harmless, and honest AI assistant."
            },
        )

        output_path = tmp_path / "generated.yaml"
        templates = generate_templates(config, mock_model)
        write_templates_yaml(templates, output_path)

        loaded = load_templates_from_yaml(output_path)

        # Find the template with context
        with_context = [t for t in loaded if "situating_context:helpful" in t.tags]
        assert len(with_context) == 1
        assert "You are a helpful, harmless, and honest AI assistant." in with_context[0].template
