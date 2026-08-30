"""HuggingFace-based model for activation extraction and steering."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.models.base import (
    ASSISTANT_SELECTOR_REGISTRY,
    BATCHED_SELECTOR_REGISTRY,
    COMPLETION_SELECTORS,
    TASK_SELECTOR_REGISTRY,
    TASK_SELECTORS,
    TOKEN_ID_SELECTORS,
    ActivationResults,
    GenerationResult,
    LayerHook,
    find_eot_indices,
    is_anchored_offset_selector,
    needs_assistant_content_span,
    needs_assistant_tb_anchor,
    needs_followup_content_span,
    parse_anchored_offset,
    requires_chat_template,
    split_selectors,
    validate_selectors,
)
from src.models.registry import is_valid_model, get_hf_name, get_eot_token
from src.models.architecture import get_layers, get_n_layers, get_hidden_dim
from src.steering.tokenization import find_text_span
from src.types import Message


class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
        attn_implementation: str = "sdpa",
        subfolder: str | None = None,
        max_memory: dict[int, str] | None = None,
    ):
        self.canonical_model_name = model_name
        if is_valid_model(model_name):
            resolved_name = get_hf_name(model_name)
        else:
            resolved_name = model_name
        self.model_name = resolved_name
        self.max_new_tokens = max_new_tokens
        # device_map="auto" distributes across GPUs; tensor placement uses "cuda"
        self.device = "cuda" if device == "auto" else device
        # For *-nothink Qwen variants, force enable_thinking=False on every chat
        # template render. Without this, Qwen3.5 emits a <think>...</think> block
        # before its real answer; the OpenRouter `/no_think` system-prompt trick
        # only affects the API path, not local HF inference.
        self._default_enable_thinking: bool | None = (
            False if model_name.endswith("-nothink") else None
        )

        torch_dtype = getattr(torch, dtype)
        load_kwargs: dict = dict(
            torch_dtype=torch_dtype,
            device_map=device,
        )
        if max_memory is not None:
            # Force balanced sharding with explicit per-GPU caps to leave forward-pass headroom.
            # Keys must be int (GPU index) or "cpu". Values like "60GiB".
            load_kwargs["max_memory"] = max_memory
        if subfolder is not None:
            load_kwargs["subfolder"] = subfolder
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_name, attn_implementation=attn_implementation, **load_kwargs,
            )
        except ValueError:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    resolved_name, attn_implementation="eager", **load_kwargs,
                )
            except ValueError:
                # Try the model-specific text-only ForCausalLM class first (avoids
                # loading a vision encoder via AutoModelForImageTextToText). This
                # is hugely important for Qwen3.5-MoE which is multimodal: its
                # text decoder alone is 244GB; loading the vision tower on top
                # OOMs even on 4×80GB GPUs during forward passes.
                text_only_loaded = False
                try:
                    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
                    self.model = Qwen3_5MoeForCausalLM.from_pretrained(
                        resolved_name, attn_implementation="sdpa", **load_kwargs,
                    )
                    text_only_loaded = True
                except (ValueError, ImportError, OSError):
                    pass
                if not text_only_loaded:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        resolved_name, attn_implementation="eager", **load_kwargs,
                    )
        self.model.eval()
        tokenizer_kwargs = {"subfolder": subfolder} if subfolder else {}
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def n_layers(self) -> int:
        return get_n_layers(self.model)

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim(self.model)

    def resolve_layer(self, layer: int | float) -> int:
        """Resolve layer index. Floats in [0, 1] are relative positions."""
        if isinstance(layer, float):
            return int(layer * self.n_layers)
        return layer

    def _get_layer(self, layer: int) -> torch.nn.Module:
        return get_layers(self.model)[layer]

    def _get_eot_token_id(self) -> int:
        eot_token = get_eot_token(self.canonical_model_name)
        return self.tokenizer.convert_tokens_to_ids(eot_token)

    @property
    def has_chat_template(self) -> bool:
        return self.tokenizer.chat_template is not None

    def _validate_selectors_for_model(self, selector_names: list[str]) -> None:
        validate_selectors(selector_names)
        if self.has_chat_template:
            return
        bad = [s for s in selector_names if requires_chat_template(s)]
        if bad:
            raise ValueError(
                f"Selectors {bad} require a chat template, but {self.model_name} has none. "
                f"Use task_last or task_mean instead."
            )

    def format_messages(
        self,
        messages: list[Message],
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        if self.has_chat_template:
            kwargs: dict = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
            effective_enable_thinking = enable_thinking if enable_thinking is not None else self._default_enable_thinking
            if effective_enable_thinking is not None:
                # Qwen3.5 deprecated the /no_think soft switch; this is the supported toggle.
                kwargs["enable_thinking"] = effective_enable_thinking
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        # Base models without chat template: concatenate message content
        parts = []
        if messages and messages[0]["role"] == "system":
            parts.append(messages[0]["content"])
            messages = messages[1:]
        for m in messages:
            parts.append(m["content"])
        return "\n\n".join(parts)

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _get_assistant_start_position(self, messages: list[Message]) -> int:
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Messages must end with an assistant message")
        prompt_messages = messages[:-1]
        prompt_with_header = self.format_messages(prompt_messages, add_generation_prompt=True)
        return self._tokenize(prompt_with_header).shape[1]

    def _get_first_assistant_span(self, messages: list[Message]) -> tuple[int, int]:
        """Find token indices [start, end) of the first assistant message content.

        Uses differential formatting to avoid substring matching ambiguity:
        tokenize up to assistant header to get start, append content to get end.
        """
        asst_idx = self._find_first_assistant_idx(messages)
        before = self.format_messages(messages[:asst_idx], add_generation_prompt=True)
        start = self._tokenize(before).shape[1]
        through = before + messages[asst_idx]["content"]
        end = self._tokenize(through).shape[1]
        return start, end

    def _get_assistant_to_user_anchor(self, messages: list[Message]) -> int:
        """Find the token position where the follow-up user content starts.

        This is the anchor for assistant_tb:N selectors — the structural
        equivalent of first_completion_indices for turn_boundary:N.
        Both anchors point to the first content token after a turn header,
        so the same offsets select the same structural tokens:
          -1 = \\n, -2 = role, -3 = <start_of_turn>, -4 = \\n, -5 = <end_of_turn>
        """
        asst_idx = self._find_first_assistant_idx(messages)
        followup_idx = asst_idx + 1
        if followup_idx >= len(messages):
            raise ValueError("No message after first assistant turn")
        # Format through the follow-up message, then strip its content via rindex
        formatted = self.format_messages(messages[:followup_idx + 1], add_generation_prompt=False)
        content_char_start = formatted.rindex(messages[followup_idx]["content"])
        header = formatted[:content_char_start]
        return self._tokenize(header).shape[1]

    def _get_followup_user_span(self, messages: list[Message]) -> tuple[int, int]:
        """Find token indices [start, end) from assistant content end through follow-up content end.

        Includes turn boundary tokens (end-of-turn, start-of-turn markers) so the
        probe signal at the transition is visible.
        """
        asst_idx = self._find_first_assistant_idx(messages)
        followup_idx = asst_idx + 1
        if followup_idx >= len(messages) or messages[followup_idx]["role"] != "user":
            raise ValueError("No follow-up user message after first assistant turn")
        # Start from end of assistant content (includes turn boundary tokens in span)
        _, start = self._get_first_assistant_span(messages)
        # End at end of followup content
        header_with_content = self.format_messages(messages[:followup_idx + 1], add_generation_prompt=False)
        content = messages[followup_idx]["content"]
        content_char_start = header_with_content.rindex(content)
        through = header_with_content[:content_char_start] + content
        end = self._tokenize(through).shape[1]
        return start, end

    @staticmethod
    def _find_first_assistant_idx(messages: list[Message]) -> int:
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                return i
        raise ValueError("No assistant message found in messages")

    def _get_task_span(self, messages: list[Message]) -> tuple[int, int]:
        """Find token indices [start, end) of the last user message content."""
        user_content = None
        for m in reversed(messages):
            if m["role"] == "user":
                user_content = m["content"]
                break
        if user_content is None:
            raise ValueError("No user message found in messages")

        has_completion = messages and messages[-1]["role"] == "assistant"
        formatted = self.format_messages(messages, add_generation_prompt=not has_completion)
        # Chat templates may strip trailing whitespace from message content
        search_text = user_content.strip() if formatted.find(user_content) == -1 else user_content
        return find_text_span(self.tokenizer, formatted, search_text)

    @contextmanager
    def _hooked_forward(
        self, layer_callbacks: dict[int, Callable[[torch.Tensor], None]],
    ) -> Iterator[None]:
        """Context manager that registers per-layer callbacks during a forward pass.

        Each callback receives the on-device hidden state tensor (batch, seq, d_model).
        """
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for layer, cb in layer_callbacks.items():
            def make_hook(callback: Callable[[torch.Tensor], None]) -> Callable:
                def hook(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    callback(hidden)
                return hook
            handles.append(self._get_layer(layer).register_forward_hook(make_hook(cb)))

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def _capture_callbacks(layers: list[int]) -> tuple[dict[int, Callable[[torch.Tensor], None]], dict[int, torch.Tensor]]:
        """Build callbacks that capture activations to CPU. Returns (callbacks, activations_dict)."""
        activations: dict[int, torch.Tensor] = {}

        def make_capture(layer_idx: int) -> Callable[[torch.Tensor], None]:
            def capture(hidden: torch.Tensor) -> None:
                activations[layer_idx] = hidden.detach().cpu()
            return capture

        callbacks = {layer: make_capture(layer) for layer in layers}
        return callbacks, activations

    def _left_pad(
        self, token_ids_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Left-pad a list of 1-D token tensors. Returns (padded, attention_mask, seq_lengths)."""
        seq_lengths = [ids.shape[0] for ids in token_ids_list]
        max_len = max(seq_lengths)
        batch_size = len(token_ids_list)

        padded = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, ids in enumerate(token_ids_list):
            pad_offset = max_len - seq_lengths[i]
            padded[i, pad_offset:] = ids
            attention_mask[i, pad_offset:] = 1

        return padded, attention_mask, seq_lengths

    def _apply_selectors(
        self,
        activations: dict[int, torch.Tensor],
        selector_names: list[str],
        first_completion_indices: torch.Tensor,
        max_seq_len: int,
        input_ids: torch.Tensor | None = None,
        task_starts: torch.Tensor | None = None,
        task_ends: torch.Tensor | None = None,
        assistant_starts: torch.Tensor | None = None,
        assistant_ends: torch.Tensor | None = None,
        assistant_to_user_anchor: torch.Tensor | None = None,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Apply token selectors to reduce (batch, seq, d_model) -> (batch, d_model).

        Builds a dispatch table mapping each selector name to a function
        (layer_acts) -> (batch, d_model), then applies them all uniformly.
        """
        batch_size = first_completion_indices.shape[0]
        seq_lengths = torch.tensor([max_seq_len] * batch_size)
        batch_range = torch.arange(batch_size)

        # Anchor tensors for anchored offset selectors (turn_boundary:N, assistant_tb:N)
        anchors: dict[str, torch.Tensor] = {
            "first_completion": first_completion_indices,
        }
        if assistant_to_user_anchor is not None:
            anchors["assistant_to_user"] = assistant_to_user_anchor

        # Pre-compute single-index selectors: name -> (batch,) index tensor
        index_selectors: dict[str, torch.Tensor] = {}
        for name in selector_names:
            parsed = parse_anchored_offset(name)
            if parsed is not None:
                anchor_name, offset = parsed
                assert anchor_name in anchors, f"{name} requires anchor '{anchor_name}'"
                index_selectors[name] = anchors[anchor_name] + offset
        if "eot" in selector_names:
            assert input_ids is not None, "eot selector requires input_ids"
            eot_token_id = self._get_eot_token_id()
            index_selectors["eot"] = find_eot_indices(input_ids, eot_token_id, first_completion_indices)

        # Build per-selector dispatch: name -> fn(layer_acts) -> (batch, d_model)
        dispatch: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for name in selector_names:
            if name in index_selectors:
                idx = index_selectors[name]
                dispatch[name] = lambda acts, _idx=idx: acts[batch_range, _idx, :]

            elif name in BATCHED_SELECTOR_REGISTRY:
                fn = BATCHED_SELECTOR_REGISTRY[name]
                fci = first_completion_indices.cpu()
                dispatch[name] = lambda acts, _fn=fn, _fci=fci: _fn(acts, _fci, seq_lengths)

            elif name in TASK_SELECTOR_REGISTRY:
                assert task_starts is not None and task_ends is not None
                fn = TASK_SELECTOR_REGISTRY[name]
                dispatch[name] = lambda acts, _fn=fn: _fn(acts, task_starts, task_ends)

            elif name in ASSISTANT_SELECTOR_REGISTRY:
                assert assistant_starts is not None and assistant_ends is not None
                fn = ASSISTANT_SELECTOR_REGISTRY[name]
                dispatch[name] = lambda acts, _fn=fn: _fn(acts, assistant_starts, assistant_ends)

            else:
                raise ValueError(f"No dispatch for selector: {name}")

        results: dict[str, dict[int, np.ndarray]] = {name: {} for name in selector_names}
        for layer, layer_acts in activations.items():
            for name in selector_names:
                results[name][layer] = dispatch[name](layer_acts).float().numpy()
        return results

    def _apply_span_selectors(
        self,
        activations: dict[int, torch.Tensor],
        span_selector_names: list[str],
        assistant_starts: torch.Tensor | None = None,
        assistant_ends: torch.Tensor | None = None,
        followup_starts: torch.Tensor | None = None,
        followup_ends: torch.Tensor | None = None,
    ) -> dict[str, dict[int, list[np.ndarray]]]:
        """Extract variable-length per-token spans. Returns {name: {layer: [array_per_sample]}}."""
        span_bounds = {
            "assistant_all": (assistant_starts, assistant_ends),
            "followup_all": (followup_starts, followup_ends),
        }
        first_name = span_selector_names[0]
        starts, ends = span_bounds[first_name]
        batch_size = starts.shape[0]
        results: dict[str, dict[int, list[np.ndarray]]] = {name: {} for name in span_selector_names}
        for layer, layer_acts in activations.items():
            for name in span_selector_names:
                starts, ends = span_bounds[name]
                assert starts is not None and ends is not None, f"Missing span bounds for {name}"
                per_sample = [
                    layer_acts[i, starts[i]:ends[i], :].float().numpy()
                    for i in range(batch_size)
                ]
                results[name][layer] = per_sample
        return results

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def _decode_completions(
        self, output_ids: torch.Tensor, prompt_len: int, n: int,
    ) -> list[str]:
        return [
            self.tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True).strip()
            for i in range(n)
        ]

    def _build_gen_kwargs(
        self,
        temperature: float,
        max_new_tokens: int | None,
        num_return_sequences: int = 1,
    ) -> dict:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": num_return_sequences,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        return gen_kwargs

    @torch.inference_mode()
    def get_logprobs(
        self,
        messages: list[Message],
        top_k: int = 10,
    ) -> dict[str, float]:
        prompt = self.format_messages(messages, add_generation_prompt=False)
        input_ids = self._tokenize(prompt)
        logits = self.model(input_ids).logits[0, -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        top_values, top_indices = torch.topk(log_probs, top_k)
        return {
            self.tokenizer.decode(idx.item()): val.item()
            for idx, val in zip(top_indices, top_values)
        }

    @torch.inference_mode()
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        prompt = self.format_messages(
            messages, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids, **self._build_gen_kwargs(temperature, max_new_tokens),
        )
        return self._decode_completions(output_ids, prompt_len, 1)[0]

    @torch.inference_mode()
    def generate_n(
        self,
        messages: list[Message],
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        """Generate n completions in a single forward pass (shared prefill)."""
        prompt = self.format_messages(
            messages, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids, **self._build_gen_kwargs(temperature, max_new_tokens, n),
        )
        return self._decode_completions(output_ids, prompt_len, n)

    @torch.inference_mode()
    def generate_batch(
        self,
        messages_batch: list[list[Message]],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        enable_thinking: bool | None = None,
        n: int = 1,
    ) -> list[list[str]]:
        """Generate completions for a batch of distinct prompts in one forward pass.

        Returns a list of length len(messages_batch); each entry is a list of
        n completions for that prompt. Sequences are left-padded so the last
        token positions align (matches the rest of the codebase).

        Use this for eval loops where N independent prompts can share a single
        prefill — typically 5–10× faster than calling generate() in a loop on
        a memory-bandwidth-bound MoE model.
        """
        prompts = [
            self.format_messages(m, add_generation_prompt=True, enable_thinking=enable_thinking)
            for m in messages_batch
        ]
        token_ids_list = [self._tokenize(p)[0] for p in prompts]
        padded, attention_mask, _ = self._left_pad(token_ids_list)
        prompt_len = padded.shape[1]

        output_ids = self.model.generate(
            padded,
            attention_mask=attention_mask,
            **self._build_gen_kwargs(temperature, max_new_tokens, num_return_sequences=n),
        )

        # output_ids: (batch * n, prompt_len + new_tokens). Strip prompt prefix
        # uniformly (left-padding makes prompt_len identical across rows).
        results: list[list[str]] = []
        for b in range(len(messages_batch)):
            completions: list[str] = []
            for i in range(n):
                row = output_ids[b * n + i]
                completions.append(
                    self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                )
            results.append(completions)
        return results

    @torch.inference_mode()
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
    ) -> ActivationResults:
        """Get activations for a single conversation.

        Returns ActivationResults with dict-like access for point selectors
        (results["last"][layer]) and .span for span selectors.
        """
        self._validate_selectors_for_model(selector_names)
        point_selectors, span_selectors = split_selectors(selector_names)
        has_completion = messages and messages[-1]["role"] == "assistant"

        if not has_completion:
            needs_completion = set(selector_names) & COMPLETION_SELECTORS
            if needs_completion:
                raise ValueError(
                    f"Selectors {needs_completion} require an assistant message, "
                    f"but messages end with role '{messages[-1]['role']}'"
                )
            prompt = self.format_messages(messages, add_generation_prompt=True)
            input_ids = self._tokenize(prompt)
            seq_len = input_ids.shape[1]
            first_completion_idx = seq_len
        else:
            prompt = self.format_messages(messages, add_generation_prompt=False)
            input_ids = self._tokenize(prompt)
            first_completion_idx = self._get_assistant_start_position(messages)
            seq_len = input_ids.shape[1]

            if first_completion_idx >= seq_len:
                raise ValueError(
                    f"Assistant start position {first_completion_idx} is beyond "
                    f"sequence length {seq_len}. "
                    "This can happen if the assistant message is empty."
                )

            n_completion_tokens = seq_len - first_completion_idx
            if n_completion_tokens < 2:
                raise ValueError(
                    f"Assistant message has no content tokens (only {n_completion_tokens} "
                    f"token after prompt, likely just an end-of-turn marker). "
                    "This can happen if the assistant message is empty."
                )

        capture_callbacks, activations = self._capture_callbacks(layers)
        with self._hooked_forward(capture_callbacks):
            self.model(input_ids)

        first_indices = torch.tensor([first_completion_idx])
        needs_ids = set(selector_names) & TOKEN_ID_SELECTORS
        needs_task = set(selector_names) & TASK_SELECTORS
        task_starts = task_ends = None
        assistant_starts = assistant_ends = None
        followup_starts = followup_ends = None
        assistant_anchor = None
        if needs_task:
            start, end = self._get_task_span(messages)
            task_starts = torch.tensor([start])
            task_ends = torch.tensor([end])
        if needs_assistant_content_span(selector_names):
            a_start, a_end = self._get_first_assistant_span(messages)
            assistant_starts = torch.tensor([a_start])
            assistant_ends = torch.tensor([a_end])
        if needs_followup_content_span(selector_names):
            f_start, f_end = self._get_followup_user_span(messages)
            followup_starts = torch.tensor([f_start])
            followup_ends = torch.tensor([f_end])
        if needs_assistant_tb_anchor(selector_names):
            assistant_anchor = torch.tensor([self._get_assistant_to_user_anchor(messages)])

        point_results: dict[str, dict[int, np.ndarray]] = {}
        if point_selectors:
            batched = self._apply_selectors(
                activations, point_selectors, first_indices, seq_len,
                input_ids=input_ids if needs_ids else None,
                task_starts=task_starts,
                task_ends=task_ends,
                assistant_starts=assistant_starts,
                assistant_ends=assistant_ends,
                assistant_to_user_anchor=assistant_anchor,
            )
            point_results = {
                name: {layer: acts[0] for layer, acts in layer_dict.items()}
                for name, layer_dict in batched.items()
            }

        span_results: dict[str, dict[int, list[np.ndarray]]] = {}
        if span_selectors:
            span_results = self._apply_span_selectors(
                activations, span_selectors,
                assistant_starts=assistant_starts,
                assistant_ends=assistant_ends,
                followup_starts=followup_starts,
                followup_ends=followup_ends,
            )

        return ActivationResults(point_results, span_results)

    @torch.inference_mode()
    def get_activations_batch(
        self,
        messages_batch: list[list[Message]],
        layers: list[int],
        selector_names: list[str],
    ) -> ActivationResults:
        """Get activations for a batch of conversations via a single forward pass.

        Left-pads sequences to equal length. Selector indices are shifted to
        account for the padding offset.

        Returns ActivationResults with dict-like access for point selectors
        and .span for span selectors.
        """
        self._validate_selectors_for_model(selector_names)
        point_selectors, span_selectors = split_selectors(selector_names)
        needs_task = set(selector_names) & TASK_SELECTORS
        _needs_content_span = needs_assistant_content_span(selector_names)
        _needs_followup_span = needs_followup_content_span(selector_names)
        _needs_tb_anchor = needs_assistant_tb_anchor(selector_names)
        token_ids_list: list[torch.Tensor] = []
        first_completion_indices: list[int] = []
        task_start_indices: list[int] = []
        task_end_indices: list[int] = []
        assistant_start_indices: list[int] = []
        assistant_end_indices: list[int] = []
        followup_start_indices: list[int] = []
        followup_end_indices: list[int] = []
        assistant_anchor_indices: list[int] = []
        for messages in messages_batch:
            has_completion = messages and messages[-1]["role"] == "assistant"
            if has_completion:
                prompt = self.format_messages(messages, add_generation_prompt=False)
                ids = self._tokenize(prompt)[0]  # (seq_len,)
                token_ids_list.append(ids)
                first_completion_indices.append(self._get_assistant_start_position(messages))
            else:
                needs_completion_sel = set(selector_names) & COMPLETION_SELECTORS
                if needs_completion_sel:
                    raise ValueError(
                        f"Selectors {needs_completion_sel} require an assistant message, "
                        f"but messages end with role '{messages[-1]['role']}'"
                    )
                prompt = self.format_messages(messages, add_generation_prompt=True)
                ids = self._tokenize(prompt)[0]
                token_ids_list.append(ids)
                first_completion_indices.append(ids.shape[0])
            if needs_task:
                start, end = self._get_task_span(messages)
                task_start_indices.append(start)
                task_end_indices.append(end)
            if _needs_content_span:
                a_start, a_end = self._get_first_assistant_span(messages)
                assistant_start_indices.append(a_start)
                assistant_end_indices.append(a_end)
            if _needs_followup_span:
                f_start, f_end = self._get_followup_user_span(messages)
                followup_start_indices.append(f_start)
                followup_end_indices.append(f_end)
            if _needs_tb_anchor:
                assistant_anchor_indices.append(self._get_assistant_to_user_anchor(messages))

        padded, attention_mask, seq_lengths = self._left_pad(token_ids_list)
        max_len = padded.shape[1]
        batch_size = len(messages_batch)

        capture_callbacks, activations = self._capture_callbacks(layers)
        with self._hooked_forward(capture_callbacks):
            self.model(padded, attention_mask=attention_mask)

        # Shift indices to account for left-padding
        shifted_first = torch.tensor([
            first_completion_indices[i] + (max_len - seq_lengths[i])
            for i in range(batch_size)
        ])

        needs_ids = set(selector_names) & TOKEN_ID_SELECTORS
        task_starts = task_ends = None
        assistant_starts = assistant_ends = None
        followup_starts = followup_ends = None
        assistant_anchor = None
        if needs_task:
            task_starts = torch.tensor([
                task_start_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
            task_ends = torch.tensor([
                task_end_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
        if _needs_content_span:
            assistant_starts = torch.tensor([
                assistant_start_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
            assistant_ends = torch.tensor([
                assistant_end_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
        if _needs_followup_span:
            followup_starts = torch.tensor([
                followup_start_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
            followup_ends = torch.tensor([
                followup_end_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
        if _needs_tb_anchor:
            assistant_anchor = torch.tensor([
                assistant_anchor_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])

        point_results: dict[str, dict[int, np.ndarray]] = {}
        if point_selectors:
            point_results = self._apply_selectors(
                activations, point_selectors, shifted_first, max_len,
                input_ids=padded if needs_ids else None,
                task_starts=task_starts,
                task_ends=task_ends,
                assistant_starts=assistant_starts,
                assistant_ends=assistant_ends,
                assistant_to_user_anchor=assistant_anchor,
            )

        span_results: dict[str, dict[int, list[np.ndarray]]] = {}
        if span_selectors:
            span_results = self._apply_span_selectors(
                activations, span_selectors,
                assistant_starts=assistant_starts,
                assistant_ends=assistant_ends,
                followup_starts=followup_starts,
                followup_ends=followup_ends,
            )

        return ActivationResults(point_results, span_results)

    @torch.inference_mode()
    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, add_generation_prompt=True)
        prompt_tokens = len(self._tokenize(prompt)[0])

        completion = self.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        completion_tokens = len(self.tokenizer(completion, return_tensors="pt").input_ids[0])

        full_messages = messages + [{"role": "assistant", "content": completion}]
        activations = self.get_activations(full_messages, layers, selector_names)

        return GenerationResult(
            completion=completion,
            activations=activations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @torch.inference_mode()
    def generate_with_hook(
        self,
        messages: list[Message],
        layer: int,
        hook: LayerHook,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with a hook applied at a single layer."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=[(layer, hook)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )[0]

    @torch.inference_mode()
    def generate_with_hook_n(
        self,
        messages: list[Message],
        layer: int,
        hook: LayerHook,
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate n completions with a hook at a single layer (shared prefill)."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=[(layer, hook)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
        )

    @torch.inference_mode()
    def generate_with_hooks(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with hooks applied at multiple layers simultaneously."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=layer_hooks,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )[0]

    @torch.inference_mode()
    def generate_with_hooks_n(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate n completions with hooks at multiple layers (shared prefill)."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=layer_hooks,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
        )

    @contextmanager
    def _registered_hooks(
        self, layer_hooks: list[tuple[int, LayerHook]], prompt_len: int,
    ) -> Iterator[None]:
        """Context manager that registers steering hooks on transformer layers."""
        def make_hf_hook(hook: LayerHook) -> Callable:
            def hf_hook(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> tuple | torch.Tensor:
                hidden = output[0] if isinstance(output, tuple) else output
                modified = hook(hidden, prompt_len)
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hf_hook

        handles = [
            self._get_layer(layer).register_forward_hook(make_hf_hook(hook))
            for layer, hook in layer_hooks
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _generate_hooked(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
        temperature: float,
        max_new_tokens: int | None,
        num_return_sequences: int,
    ) -> list[str]:
        prompt = self.format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]

        with self._registered_hooks(layer_hooks, prompt_len):
            output_ids = self.model.generate(
                input_ids,
                **self._build_gen_kwargs(temperature, max_new_tokens, num_return_sequences),
            )

        return self._decode_completions(output_ids, prompt_len, num_return_sequences)

    @torch.inference_mode()
    def prefill_with_hooks(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
    ) -> tuple[DynamicCache, torch.Tensor]:
        """Run prefill with optional hooks and return (kv_cache, input_ids).

        The cache covers all prompt positions [0, seq_len).
        """
        prompt = self.format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]

        with self._registered_hooks(layer_hooks, prompt_len):
            outputs = self.model(input_ids, use_cache=True)

        return outputs.past_key_values, input_ids

    @torch.inference_mode()
    def recompute_suffix(
        self,
        cache: DynamicCache,
        input_ids: torch.Tensor,
        recompute_from: int,
    ) -> DynamicCache:
        """Re-run forward pass for tokens [recompute_from, seq_len) attending to the cache at [0, recompute_from)."""
        truncated = DynamicCache()
        for layer_idx in range(len(cache)):
            layer = cache.layers[layer_idx]
            k = layer.keys[:, :, :recompute_from, :].clone()
            v = layer.values[:, :, :recompute_from, :].clone()
            truncated.update(k, v, layer_idx)

        suffix_ids = input_ids[:, recompute_from:]
        outputs = self.model(suffix_ids, past_key_values=truncated, use_cache=True)
        return outputs.past_key_values

    @torch.inference_mode()
    def generate_from_cache(
        self,
        cache: DynamicCache,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate from a (possibly modified) KV cache.

        Truncates the cache to [0, seq_len-1), expands both cache and input_ids
        to num_return_sequences, then uses model.generate() which re-forwards
        the last prompt token against the modified cache.

        WARNING: mutates `cache` in-place (truncation). Do not reuse after calling.
        """
        seq_len = input_ids.shape[1]
        n = num_return_sequences

        # Truncate and expand cache for batch generation
        batch_cache = DynamicCache()
        for layer_idx in range(len(cache)):
            layer = cache.layers[layer_idx]
            k = layer.keys[:, :, :seq_len - 1, :].expand(n, -1, -1, -1).contiguous()
            v = layer.values[:, :, :seq_len - 1, :].expand(n, -1, -1, -1).contiguous()
            batch_cache.update(k, v, layer_idx)

        expanded_ids = input_ids.expand(n, -1).contiguous()
        gen_kwargs = self._build_gen_kwargs(temperature, max_new_tokens, num_return_sequences=1)
        gen_kwargs["past_key_values"] = batch_cache

        output_ids = self.model.generate(expanded_ids, **gen_kwargs)
        return self._decode_completions(output_ids, seq_len, n)
