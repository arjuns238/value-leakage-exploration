"""Steered HuggingFace model that duck-types as OpenAICompatibleClient."""

from __future__ import annotations

import asyncio
from typing import Callable

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_compatible import BatchResult, GenerateRequest
from src.steering.hooks import (
    STEERING_MODES,
    compose_hooks,
    position_selective_steering,
)
from src.steering.kv_cache import combine_caches, modify_cache_kv_at_positions, project_to_kv_space
from src.steering.tokenization import find_pairwise_task_spans


class SteeredHFClient:
    """HuggingFaceModel with steering, duck-typed as OpenAICompatibleClient."""

    def __init__(
        self,
        hf_model: HuggingFaceModel,
        layer: int,
        steering_direction: np.ndarray,
        coefficient: float,
        steering_mode: str = "all_tokens",
        cache_injection: str = "hook",
        recompute_suffix: bool = False,
        a_marker: str = "Task A:",
        b_marker: str = "Task B:",
    ):
        self.hf_model = hf_model
        self.layer = layer
        self._direction = steering_direction
        self.coefficient = coefficient
        self.steering_mode = steering_mode
        if steering_mode == "cache" and cache_injection not in ("hook", "post_hoc"):
            raise ValueError(f"cache_injection must be 'hook' or 'post_hoc', got {cache_injection!r}")
        if steering_mode != "cache" and (cache_injection != "hook" or recompute_suffix):
            raise ValueError(
                "cache_injection and recompute_suffix are only valid with steering_mode='cache'"
            )
        self.cache_injection = cache_injection
        self.recompute_suffix = recompute_suffix
        self.a_marker = a_marker
        self.b_marker = b_marker

        self.canonical_model_name = hf_model.canonical_model_name
        self.model_name = hf_model.model_name
        self.max_new_tokens = hf_model.max_new_tokens

        # Pre-compute scaled steering tensor on GPU
        scaled = steering_direction * coefficient
        self._steering_tensor = torch.tensor(
            scaled, dtype=torch.bfloat16, device=hf_model.device
        )

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    def with_coefficient(self, coefficient: float) -> SteeredHFClient:
        """Return a new client with a different coefficient, sharing the same model."""
        return SteeredHFClient(
            self.hf_model,
            self.layer,
            self._direction,
            coefficient,
            self.steering_mode,
            self.cache_injection,
            self.recompute_suffix,
            self.a_marker,
            self.b_marker,
        )

    def _resolve_task_spans(
        self, messages: list, task_prompts: list[str]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        formatted = self.hf_model.format_messages(messages)
        return find_pairwise_task_spans(
            self.hf_model.tokenizer, formatted,
            task_prompts[0], task_prompts[1],
            self.a_marker, self.b_marker,
        )

    def _resolve_hook(self, messages: list, task_prompts: list[str] | None = None):
        if task_prompts is not None and len(task_prompts) == 2 and self.steering_mode == "differential":
            first_span, second_span = self._resolve_task_spans(messages, task_prompts)
            return compose_hooks(
                position_selective_steering(self._steering_tensor, first_span[0], first_span[1]),
                position_selective_steering(-self._steering_tensor, second_span[0], second_span[1]),
            )
        return STEERING_MODES[self.steering_mode](self._steering_tensor)

    def generate(self, messages, temperature=1.0, task_prompts: list[str] | None = None) -> str:
        return self._dispatch(messages, temperature, task_prompts, n=1)[0]

    def generate_n(self, messages, n: int, temperature: float = 1.0, task_prompts: list[str] | None = None) -> list[str]:
        return self._dispatch(messages, temperature, task_prompts, n=n)

    def _dispatch(
        self,
        messages: list,
        temperature: float,
        task_prompts: list[str] | None,
        n: int,
    ) -> list[str]:
        if self.coefficient == 0:
            return self.hf_model.generate_n(messages, n=n, temperature=temperature)

        if self.steering_mode == "cache":
            if task_prompts is None or len(task_prompts) != 2:
                raise ValueError(
                    f"Cache steering modes require exactly 2 task_prompts, got {task_prompts}"
                )
            first_span, second_span = self._resolve_task_spans(messages, task_prompts)
            return self._generate_cache_steered(messages, first_span, second_span, temperature, n)

        hook = self._resolve_hook(messages, task_prompts)
        return self.hf_model.generate_with_hook_n(
            messages=messages, layer=self.layer, hook=hook, n=n, temperature=temperature,
        )

    def _generate_cache_steered(
        self,
        messages: list,
        first_span: tuple[int, int],
        second_span: tuple[int, int],
        temperature: float,
        n: int,
    ) -> list[str]:
        if self.cache_injection == "hook":
            combined, input_ids = self._build_hook_injected_cache(messages, first_span, second_span)
        else:
            combined, input_ids = self._build_post_hoc_cache(messages, first_span, second_span)

        if self.recompute_suffix and second_span[1] < input_ids.shape[1]:
            combined = self.hf_model.recompute_suffix(combined, input_ids, second_span[1])

        return self.hf_model.generate_from_cache(
            combined, input_ids, temperature=temperature, num_return_sequences=n,
        )

    def _build_hook_injected_cache(self, messages, first_span, second_span):
        """Three prefills: clean base, +steer on A, -steer on B. Splice task spans onto clean."""
        first_hook = position_selective_steering(self._steering_tensor, first_span[0], first_span[1])
        second_hook = position_selective_steering(-self._steering_tensor, second_span[0], second_span[1])

        cache_clean, input_ids = self.hf_model.prefill_with_hooks(messages, [])
        cache_a, _ = self.hf_model.prefill_with_hooks(messages, [(self.layer, first_hook)])
        cache_b, _ = self.hf_model.prefill_with_hooks(messages, [(self.layer, second_hook)])

        combined = combine_caches(cache_clean, [
            (cache_a, first_span[0], first_span[1]),
            (cache_b, second_span[0], second_span[1]),
        ])
        return combined, input_ids

    def _build_post_hoc_cache(self, messages, first_span, second_span):
        """Clean prefill, then modify K and V cache at task positions."""
        cache, input_ids = self.hf_model.prefill_with_hooks(messages, [])
        k_dir, v_dir = project_to_kv_space(self.hf_model.model, self.layer, self._direction)
        modify_cache_kv_at_positions(cache, self.layer, first_span[0], first_span[1], k_dir, v_dir, +self.coefficient)
        modify_cache_kv_at_positions(cache, self.layer, second_span[0], second_span[1], k_dir, v_dir, -self.coefficient)
        return cache, input_ids

    def _run_batch(
        self,
        requests: list[GenerateRequest],
        on_complete: Callable[[], None] | None = None,
    ) -> list[BatchResult]:
        results: list[BatchResult] = []
        for request in requests:
            if request.tools is not None:
                raise ValueError("SteeredHFClient does not support tool use")
            try:
                response = self.generate(request.messages, temperature=request.temperature, task_prompts=request.task_prompts)
                results.append(BatchResult(response=response, error=None))
            except Exception as e:
                results.append(BatchResult(response=None, error=e))
            if on_complete:
                on_complete()
        return results

    async def generate_batch_async(
        self,
        requests: list[GenerateRequest],
        semaphore: asyncio.Semaphore,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool = False,
        async_client: object | None = None,
    ) -> list[BatchResult]:
        # Async signature required because callers await this, but HF inference
        # is synchronous — semaphore + async_client accepted for interface compat
        # with OpenAICompatibleClient (ignored here).
        return self._run_batch(requests, on_complete)

    def generate_batch(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int = 10,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool = False,
    ) -> list[BatchResult]:
        return self._run_batch(requests, on_complete)
