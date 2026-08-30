"""Tests for unified steering + measurement infrastructure."""

from src.measurement.elicitation.response_format import (
    CompletionChoiceFormat,
    RegexChoiceFormat,
    XMLChoiceFormat,
)
from src.measurement.elicitation.completion_judge import RegexOnly


# --- CompletionChoiceFormat.extract_label tests ---


class TestExtractLabelCompletion:
    """CompletionChoiceFormat.extract_label: startswith-based extraction."""

    def setup_method(self):
        self.fmt = CompletionChoiceFormat(
            task_a_label="Task A",
            task_b_label="Task B",
        )

    def test_starts_with_task_a(self):
        assert self.fmt.extract_label("Task A: here is my solution...") == "a"

    def test_starts_with_task_b(self):
        assert self.fmt.extract_label("Task B: let me solve this...") == "b"

    def test_exact_label_match(self):
        assert self.fmt.extract_label("Task A") == "a"
        assert self.fmt.extract_label("Task B") == "b"

    def test_garbage_returns_refusal(self):
        assert self.fmt.extract_label("I don't want to do either task.") == "refusal"

    def test_markdown_prefix_stripped(self):
        assert self.fmt.extract_label("**Task A:** here is my poem") == "a"

    def test_empty_string_returns_refusal(self):
        assert self.fmt.extract_label("") == "refusal"


# --- parse_sync tests for non-completion formats ---


class TestParseSyncRegex:
    """RegexChoiceFormat.parse_sync: regex-based extraction."""

    def setup_method(self):
        self.fmt = RegexChoiceFormat(task_a_label="Task A", task_b_label="Task B")

    def test_label_in_response(self):
        assert self.fmt.parse_sync("I choose Task A because it's fun.") == "a"

    def test_both_labels_first_wins(self):
        assert self.fmt.parse_sync("Task A is better than Task B") == "a"

    def test_no_label_returns_parse_fail(self):
        assert self.fmt.parse_sync("Neither option appeals to me.") == "parse_fail"


class TestParseSyncXML:
    """XMLChoiceFormat.parse_sync: XML tag extraction."""

    def setup_method(self):
        self.fmt = XMLChoiceFormat(task_a_label="Task A", task_b_label="Task B")

    def test_xml_tag_extraction(self):
        assert self.fmt.parse_sync("<choice>Task A</choice>") == "a"
        assert self.fmt.parse_sync("<choice>Task B</choice>") == "b"

    def test_no_tag_returns_parse_fail(self):
        assert self.fmt.parse_sync("I pick Task A") == "parse_fail"
