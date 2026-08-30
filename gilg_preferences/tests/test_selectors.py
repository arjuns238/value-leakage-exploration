"""Unit tests for token selectors (no GPU required)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import pytest

from src.models.base import (
    SPAN_SELECTORS,
    find_eot_indices,
    parse_anchored_offset,
    select_task_last_batched,
    select_task_mean_batched,
    validate_selectors,
    is_valid_selector,
    requires_chat_template,
)
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.activations import load_span_activations
from src.probes.extraction.persistence import save_activations, load_existing_data

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TestTurnBoundarySelector:
    """Test turn_boundary:N selector logic."""

    def test_selects_correct_offset_from_first_completion(self):
        batch_size, seq_len, d_model = 2, 15, 4
        activations = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32)
        activations = activations.reshape(batch_size, seq_len, d_model)
        first_completion_indices = torch.tensor([10, 12])

        # turn_boundary:-1 → first_completion_idx - 1
        indices = first_completion_indices + (-1)
        result = activations[torch.arange(batch_size), indices, :]

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, 9, :])
        assert torch.allclose(result[1], activations[1, 11, :])

    def test_parse_valid_offsets(self):
        assert parse_anchored_offset("turn_boundary:-1") == ("first_completion", -1)
        assert parse_anchored_offset("turn_boundary:-5") == ("first_completion", -5)
        assert parse_anchored_offset("turn_boundary:0") == ("first_completion", 0)
        assert parse_anchored_offset("assistant_tb:-1") == ("assistant_to_user", -1)
        assert parse_anchored_offset("assistant_tb:-2") == ("assistant_to_user", -2)

    def test_parse_non_anchored_returns_none(self):
        assert parse_anchored_offset("last") is None
        assert parse_anchored_offset("eot") is None
        assert parse_anchored_offset("task_last") is None

    def test_parse_rejects_non_integer(self):
        with pytest.raises(ValueError, match="must be an integer"):
            parse_anchored_offset("turn_boundary:abc")
        with pytest.raises(ValueError, match="must be an integer"):
            parse_anchored_offset("assistant_tb:xyz")

    def test_parse_rejects_non_negative_assistant_tb(self):
        with pytest.raises(ValueError, match="must be negative"):
            parse_anchored_offset("assistant_tb:0")
        with pytest.raises(ValueError, match="must be negative"):
            parse_anchored_offset("assistant_tb:1")


class TestSelectorValidation:

    def test_valid_fixed_selectors(self):
        validate_selectors(["last", "first", "mean", "eot", "task_last", "task_mean"])

    def test_valid_turn_boundary(self):
        validate_selectors(["turn_boundary:-1", "turn_boundary:-5"])

    def test_valid_assistant_selectors(self):
        validate_selectors(["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"])

    def test_valid_span_selectors(self):
        validate_selectors(["assistant_all"])
        validate_selectors(["followup_all"])

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown selector"):
            validate_selectors(["nonexistent"])

    def test_rejects_old_names(self):
        with pytest.raises(ValueError, match="Unknown selector"):
            validate_selectors(["prompt_last"])
        with pytest.raises(ValueError, match="Unknown selector"):
            validate_selectors(["prompt_mean"])
        with pytest.raises(ValueError, match="Unknown selector"):
            validate_selectors(["content_last"])
        with pytest.raises(ValueError, match="Unknown selector"):
            validate_selectors(["content_mean"])

    def test_is_valid_selector(self):
        assert is_valid_selector("last")
        assert is_valid_selector("eot")
        assert is_valid_selector("task_last")
        assert is_valid_selector("turn_boundary:-1")
        assert is_valid_selector("assistant_all")
        assert is_valid_selector("followup_all")
        assert not is_valid_selector("prompt_last")
        assert not is_valid_selector("bogus")

    def test_requires_chat_template(self):
        assert requires_chat_template("eot")
        assert requires_chat_template("turn_boundary:-1")
        assert requires_chat_template("turn_boundary:-5")
        assert requires_chat_template("assistant_all")
        assert requires_chat_template("followup_all")
        assert not requires_chat_template("last")
        assert not requires_chat_template("first")
        assert not requires_chat_template("task_last")
        assert not requires_chat_template("task_mean")


class TestFindEotIndices:
    """Test find_eot_indices with synthetic token IDs simulating Gemma 3 template.

    Gemma 3 prompt layout:
      <bos> <start_of_turn> user \\n {prompt tokens} <end_of_turn> \\n <start_of_turn> model \\n
      ^0    ^1               ^2   ^3  ^4...          ^eot            ...                       ^first_completion_idx
    """

    EOT_TOKEN_ID = 107  # Gemma 3's <end_of_turn> token ID

    def _make_gemma_ids(self, prompt_len: int) -> torch.Tensor:
        """Build a fake token sequence with EOT at the right place."""
        prefix = [1, 106, 2, 3]  # <bos> <start_of_turn> user \n
        prompt = list(range(1000, 1000 + prompt_len))
        suffix = [self.EOT_TOKEN_ID, 13, 106, 4, 13]  # <end_of_turn> \n <start_of_turn> model \n
        return torch.tensor(prefix + prompt + suffix, dtype=torch.long)

    def test_finds_eot_before_first_completion(self):
        ids = self._make_gemma_ids(prompt_len=5)
        first_comp = torch.tensor([len(ids)])
        result = find_eot_indices(ids.unsqueeze(0), self.EOT_TOKEN_ID, first_comp)
        assert result[0].item() == 9  # 4 (prefix) + 5 (prompt)

    def test_batch_with_different_lengths(self):
        ids_short = self._make_gemma_ids(prompt_len=3)
        ids_long = self._make_gemma_ids(prompt_len=10)
        max_len = max(len(ids_short), len(ids_long))

        padded = torch.zeros(2, max_len, dtype=torch.long)
        pad_short = max_len - len(ids_short)
        padded[0, pad_short:] = ids_short
        padded[1, :len(ids_long)] = ids_long

        first_comp = torch.tensor([
            len(ids_short) + pad_short,
            len(ids_long),
        ])
        result = find_eot_indices(padded, self.EOT_TOKEN_ID, first_comp)

        assert result[0].item() == 4 + 3 + pad_short
        assert result[1].item() == 4 + 10

    def test_raises_if_no_eot_found(self):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        first_comp = torch.tensor([5])
        with pytest.raises(ValueError, match="No end-of-turn token"):
            find_eot_indices(ids, self.EOT_TOKEN_ID, first_comp)

    def test_eot_is_before_turn_boundary_minus_1(self):
        """EOT should be strictly before turn_boundary:-1 position."""
        ids = self._make_gemma_ids(prompt_len=8)
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.EOT_TOKEN_ID, first_comp)
        turn_boundary_minus_1 = first_comp[0] - 1
        assert eot_idx[0].item() < turn_boundary_minus_1.item()


@pytest.mark.api
class TestEotWithGemmaTokenizer:
    """Integration test: verify EOT finding works with the real Gemma 3 tokenizer."""

    @pytest.fixture(autouse=True)
    def load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")

    def _tokenize_prompt(self, user_content: str) -> torch.Tensor:
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(prompt, return_tensors="pt").input_ids[0]

    def test_eot_token_exists(self):
        assert self.eot_token_id != self.tokenizer.unk_token_id

    def test_finds_eot_in_real_prompt(self):
        ids = self._tokenize_prompt("Write me a haiku about cats")
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.eot_token_id, first_comp)
        assert ids[eot_idx[0].item()].item() == self.eot_token_id

    def test_eot_is_at_turn_boundary_minus_5(self):
        """EOT should be at turn_boundary:-5 for Gemma-3 IT."""
        ids = self._tokenize_prompt("Hello world")
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.eot_token_id, first_comp)

        offset = eot_idx[0].item() - (len(ids) - 1)
        # EOT is 4 positions before the last token: eot \n sot model \n
        # So eot_idx = first_comp - 5
        assert first_comp[0].item() - eot_idx[0].item() == 5

    def test_works_with_long_prompt(self):
        ids = self._tokenize_prompt("x " * 500)
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.eot_token_id, first_comp)
        assert ids[eot_idx[0].item()].item() == self.eot_token_id


class TestTaskLastSelector:

    def test_selects_last_task_token(self):
        batch_size, seq_len, d_model = 2, 10, 4
        activations = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32)
        activations = activations.reshape(batch_size, seq_len, d_model)
        task_starts = torch.tensor([2, 3])
        task_ends = torch.tensor([7, 9])  # exclusive

        result = select_task_last_batched(activations, task_starts, task_ends)

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, 6, :])  # 7 - 1
        assert torch.allclose(result[1], activations[1, 8, :])  # 9 - 1

    def test_differs_from_turn_boundary(self):
        """task_last should differ from turn_boundary:-1 when there are template tokens."""
        batch_size, seq_len, d_model = 1, 15, 4
        activations = torch.arange(seq_len * d_model, dtype=torch.float32).reshape(1, seq_len, d_model)
        # Task ends at 8, but first_completion is at 12 (template tokens between 8 and 12)
        task_ends = torch.tensor([8])
        first_completion_indices = torch.tensor([12])

        task_last = select_task_last_batched(activations, torch.tensor([2]), task_ends)
        turn_boundary = activations[0, first_completion_indices[0] - 1, :]

        assert torch.allclose(task_last[0], activations[0, 7, :])  # last task token
        assert torch.allclose(turn_boundary, activations[0, 11, :])  # last template token
        assert not torch.allclose(task_last[0], turn_boundary)


class TestTaskMeanSelector:

    def test_averages_over_task_tokens_only(self):
        batch_size, seq_len, d_model = 2, 10, 4
        activations = torch.randn(batch_size, seq_len, d_model)
        task_starts = torch.tensor([2, 3])
        task_ends = torch.tensor([7, 9])

        result = select_task_mean_batched(activations, task_starts, task_ends)

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, 2:7, :].mean(dim=0))
        assert torch.allclose(result[1], activations[1, 3:9, :].mean(dim=0))

    def test_excludes_template_tokens(self):
        batch_size, seq_len, d_model = 1, 15, 4
        activations = torch.zeros(batch_size, seq_len, d_model)
        # Put signal only in template tokens (before and after task)
        activations[0, :2, :] = 1.0   # before task
        activations[0, 7:, :] = 1.0   # after task
        task_starts = torch.tensor([2])
        task_ends = torch.tensor([7])

        result = select_task_mean_batched(activations, task_starts, task_ends)
        assert torch.allclose(result, torch.zeros(1, d_model))


class TestChatTemplateSelectorValidation:

    def _make_model_stub(self, has_chat_template: bool) -> HuggingFaceModel:
        stub = object.__new__(HuggingFaceModel)
        stub.tokenizer = MagicMock()
        stub.tokenizer.chat_template = "some template" if has_chat_template else None
        stub.model_name = "test-model"
        return stub

    def test_rejects_turn_boundary_on_base_model(self):
        model = self._make_model_stub(has_chat_template=False)
        with pytest.raises(ValueError, match="require a chat template"):
            model._validate_selectors_for_model(["turn_boundary:-1"])

    def test_rejects_eot_on_base_model(self):
        model = self._make_model_stub(has_chat_template=False)
        with pytest.raises(ValueError, match="require a chat template"):
            model._validate_selectors_for_model(["eot"])

    def test_allows_task_selectors_on_base_model(self):
        model = self._make_model_stub(has_chat_template=False)
        model._validate_selectors_for_model(["task_last", "task_mean"])

    def test_allows_standard_selectors_on_base_model(self):
        model = self._make_model_stub(has_chat_template=False)
        model._validate_selectors_for_model(["last", "first", "mean"])

    def test_allows_all_selectors_on_instruct_model(self):
        model = self._make_model_stub(has_chat_template=True)
        model._validate_selectors_for_model([
            "turn_boundary:-1", "turn_boundary:-5", "eot",
            "task_last", "task_mean", "last", "first", "mean",
        ])


@pytest.mark.api
class TestTaskSelectorsWithGemmaTokenizer:
    """Integration test: verify task span detection with the real Gemma 3 tokenizers."""

    @pytest.fixture(autouse=True)
    def load_tokenizers(self):
        from transformers import AutoTokenizer
        self.it_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        self.pt_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-pt")

    def _get_task_span(self, tokenizer, user_content, has_chat_template=True):
        from src.steering.tokenization import find_text_span
        messages = [{"role": "user", "content": user_content}]
        if has_chat_template and tokenizer.chat_template is not None:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            formatted = user_content
        return find_text_span(tokenizer, formatted, user_content), formatted

    def test_it_span_decodes_to_original_content(self):
        content = "Write me a haiku about cats"
        (start, end), formatted = self._get_task_span(self.it_tokenizer, content)

        ids = self.it_tokenizer(formatted, add_special_tokens=False).input_ids
        span_text = self.it_tokenizer.decode(ids[start:end])
        assert span_text.strip() == content

    def test_it_span_excludes_template_tokens(self):
        content = "Explain quantum computing"
        (start, end), formatted = self._get_task_span(self.it_tokenizer, content)

        ids = self.it_tokenizer(formatted, add_special_tokens=False).input_ids
        total_tokens = len(ids)
        assert start > 0, "Task should not start at position 0 (template prefix expected)"
        assert end < total_tokens, "Task should not extend to the end (template suffix expected)"

    def test_it_last_task_token_is_not_template(self):
        content = "Write a poem about the ocean"
        (start, end), formatted = self._get_task_span(self.it_tokenizer, content)

        ids = self.it_tokenizer(formatted, add_special_tokens=False).input_ids
        last_task_token = self.it_tokenizer.decode([ids[end - 1]])
        assert "<end_of_turn>" not in last_task_token
        assert "<start_of_turn>" not in last_task_token

    def test_pt_span_decodes_to_original_content(self):
        """Base model: task span should cover the full text (no template wrapping)."""
        content = "Write me a haiku about cats"
        (start, end), formatted = self._get_task_span(
            self.pt_tokenizer, content, has_chat_template=False,
        )

        ids = self.pt_tokenizer(formatted, add_special_tokens=False).input_ids
        span_text = self.pt_tokenizer.decode(ids[start:end])
        assert span_text.strip() == content

    def test_it_and_pt_same_content_different_positions(self):
        """Same content should be at different token positions in IT vs PT."""
        content = "Solve this math problem: 2+2"

        (it_start, it_end), _ = self._get_task_span(self.it_tokenizer, content)
        (pt_start, pt_end), _ = self._get_task_span(
            self.pt_tokenizer, content, has_chat_template=False,
        )

        assert it_start > pt_start, (
            f"IT task start ({it_start}) should be after PT start ({pt_start}) "
            "due to template prefix"
        )

    def test_various_content_lengths(self):
        for content in [
            "Hi",
            "What is the meaning of life?",
            "Please write a comprehensive analysis of " + "very " * 100 + "long topics",
        ]:
            (start, end), formatted = self._get_task_span(self.it_tokenizer, content)
            ids = self.it_tokenizer(formatted, add_special_tokens=False).input_ids
            span_text = self.it_tokenizer.decode(ids[start:end])
            assert span_text.strip() == content, f"Failed for content: {content[:50]}..."


class TestSpanSelectorDispatch:

    def test_apply_span_selectors_slices_correctly(self):
        model = object.__new__(HuggingFaceModel)
        batch_size, seq_len, d_model = 3, 20, 8
        activations = {
            0: torch.randn(batch_size, seq_len, d_model),
            5: torch.randn(batch_size, seq_len, d_model),
        }
        assistant_starts = torch.tensor([4, 6, 3])
        assistant_ends = torch.tensor([10, 15, 7])

        result = model._apply_span_selectors(
            activations, ["assistant_all"],
            assistant_starts=assistant_starts, assistant_ends=assistant_ends,
        )

        assert "assistant_all" in result
        for layer in [0, 5]:
            per_sample = result["assistant_all"][layer]
            assert len(per_sample) == batch_size
            for i in range(batch_size):
                expected_len = assistant_ends[i] - assistant_starts[i]
                assert per_sample[i].shape == (expected_len, d_model)
                expected = activations[layer][i, assistant_starts[i]:assistant_ends[i], :].float().numpy()
                np.testing.assert_array_equal(per_sample[i], expected)

    def test_followup_all_slices_correctly(self):
        model = object.__new__(HuggingFaceModel)
        batch_size, seq_len, d_model = 2, 20, 8
        activations = {
            0: torch.randn(batch_size, seq_len, d_model),
        }
        followup_starts = torch.tensor([12, 14])
        followup_ends = torch.tensor([18, 19])

        result = model._apply_span_selectors(
            activations, ["followup_all"],
            followup_starts=followup_starts, followup_ends=followup_ends,
        )

        assert "followup_all" in result
        per_sample = result["followup_all"][0]
        assert len(per_sample) == batch_size
        for i in range(batch_size):
            expected_len = followup_ends[i] - followup_starts[i]
            assert per_sample[i].shape == (expected_len, d_model)
            expected = activations[0][i, followup_starts[i]:followup_ends[i], :].float().numpy()
            np.testing.assert_array_equal(per_sample[i], expected)


class TestSpanActivationsRoundtrip:

    def test_save_load_concat_offsets(self):
        rng = np.random.default_rng(42)
        task_ids = ["task_a", "task_b", "task_c"]
        d_model = 16
        lengths = [5, 12, 3]

        acts_list_layer_0 = [rng.standard_normal((l, d_model)).astype(np.float32) for l in lengths]
        acts_list_layer_7 = [rng.standard_normal((l, d_model)).astype(np.float32) for l in lengths]

        span_activations = {
            "assistant_all": {
                0: acts_list_layer_0,
                7: acts_list_layer_7,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_activations(output_dir, task_ids, span_activations, span=True)

            path = output_dir / "activations_assistant_all.npz"
            assert path.exists()

            # Verify raw file has offsets key
            raw = np.load(path)
            assert "offsets" in raw
            np.testing.assert_array_equal(raw["offsets"], [0, 5, 17, 20])

            # Load via load_span_activations
            loaded_ids, loaded_acts = load_span_activations(path)
            assert list(loaded_ids) == task_ids
            for layer in [0, 7]:
                original = acts_list_layer_0 if layer == 0 else acts_list_layer_7
                for i in range(len(task_ids)):
                    np.testing.assert_array_equal(loaded_acts[layer][i], original[i])

    def test_load_span_with_filter(self):
        rng = np.random.default_rng(42)
        task_ids = ["a", "b", "c", "d"]
        d_model = 4
        lengths = [2, 3, 1, 5]
        acts = [rng.standard_normal((l, d_model)).astype(np.float32) for l in lengths]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_activations(output_dir, task_ids, {"assistant_all": {0: acts}}, span=True)

            path = output_dir / "activations_assistant_all.npz"
            loaded_ids, loaded_acts = load_span_activations(path, task_id_filter={"b", "d"})
            assert list(loaded_ids) == ["b", "d"]
            np.testing.assert_array_equal(loaded_acts[0][0], acts[1])
            np.testing.assert_array_equal(loaded_acts[0][1], acts[3])

    def test_load_existing_data_span_format(self):
        rng = np.random.default_rng(42)
        task_ids = ["x", "y"]
        d_model = 4
        acts = [rng.standard_normal((3, d_model)).astype(np.float32),
                rng.standard_normal((7, d_model)).astype(np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_activations(output_dir, task_ids, {"assistant_all": {2: acts}}, span=True)

            loaded_ids, loaded_acts, _ = load_existing_data(output_dir, ["assistant_all"])
            assert loaded_ids == task_ids
            assert len(loaded_acts["assistant_all"][2]) == 2
            np.testing.assert_array_equal(loaded_acts["assistant_all"][2][0], acts[0])
            np.testing.assert_array_equal(loaded_acts["assistant_all"][2][1], acts[1])
