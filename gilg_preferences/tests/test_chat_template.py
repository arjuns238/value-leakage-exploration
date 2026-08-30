"""Tests for chat template utilities."""

import pytest

pytestmark = pytest.mark.models

from transformers import AutoTokenizer


class TestStripSystemPreamble:
    """Tests for stripping Llama's default system preamble."""

    @pytest.fixture
    def llama_tokenizer(self):
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    @pytest.fixture
    def stripped_tokenizer(self, llama_tokenizer):
        lines = llama_tokenizer.chat_template.split("\n")
        filtered = [
            line for line in lines
            if "Cutting Knowledge Date" not in line and "Today Date" not in line
        ]
        llama_tokenizer.chat_template = "\n".join(filtered)
        return llama_tokenizer

    def test_default_template_has_preamble(self, llama_tokenizer):
        messages = [{"role": "user", "content": "Hello"}]
        result = llama_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "Cutting Knowledge Date" in result

    def test_strip_removes_knowledge_cutoff(self, stripped_tokenizer):
        messages = [{"role": "user", "content": "Hello"}]
        result = stripped_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "Cutting Knowledge" not in result
        assert "Today Date" not in result

    def test_strip_preserves_user_system_message(self, stripped_tokenizer):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = stripped_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "Be helpful." in result
        assert "Cutting Knowledge" not in result

    def test_strip_preserves_message_structure(self, stripped_tokenizer):
        messages = [{"role": "user", "content": "Hello"}]
        result = stripped_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result


class TestStripSystemPreambleOtherModels:
    """Ensure stripping is safe no-op for models without preamble."""

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ])
    def test_strip_is_noop_for_other_models(self, model_name):
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.chat_template is None:
            pytest.skip(f"{model_name} has no chat template")

        original = tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Apply stripping logic
        lines = tok.chat_template.split("\n")
        filtered = [
            line for line in lines
            if "Cutting Knowledge Date" not in line and "Today Date" not in line
        ]
        tok.chat_template = "\n".join(filtered)

        stripped = tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        assert original == stripped
