"""Hybrid model: API generation + local activation extraction.

Uses fast API inference (OpenRouter) for generation,
then extracts activations locally with a single forward pass.
"""

from __future__ import annotations

import numpy as np

from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_compatible import OpenAICompatibleClient, GenerateRequest
from src.models.base import ActivationResults, GenerationResult
from src.types import Message


class HybridActivationModel:
    """Generate via API, extract activations locally."""

    def __init__(
        self,
        api_client: OpenAICompatibleClient,
        local_model: HuggingFaceModel,
    ):
        self.api_client = api_client
        self.local_model = local_model

    @property
    def n_layers(self) -> int:
        return self.local_model.n_layers

    @property
    def hidden_dim(self) -> int:
        return self.local_model.hidden_dim

    def _count_tokens(self, messages: list[Message], completion: str) -> tuple[int, int]:
        prompt_text = self.local_model.format_messages(messages, add_generation_prompt=True)
        prompt_tokens = len(self.local_model._tokenize(prompt_text)[0])
        completion_tokens = len(self.local_model.tokenizer(completion, return_tensors="pt").input_ids[0])
        return prompt_tokens, completion_tokens

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
    ) -> str:
        return self.api_client.generate(messages, temperature=temperature)

    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
    ) -> ActivationResults:
        return self.local_model.get_activations(messages, layers, selector_names)

    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
        temperature: float = 1.0,
    ) -> GenerationResult:
        completion = self.api_client.generate(messages, temperature=temperature)

        full_messages = messages + [{"role": "assistant", "content": completion}]
        activations = self.local_model.get_activations(full_messages, layers, selector_names)
        prompt_tokens, completion_tokens = self._count_tokens(messages, completion)

        return GenerationResult(
            completion=completion,
            activations=activations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def get_activations_batch(
        self,
        messages_batch: list[list[Message]],
        layers: list[int],
        selector_names: list[str],
    ) -> ActivationResults:
        return self.local_model.get_activations_batch(messages_batch, layers, selector_names)

    def generate_with_activations_batch(
        self,
        messages_batch: list[list[Message]],
        layers: list[int],
        selector_names: list[str],
        temperature: float = 1.0,
    ) -> list[GenerationResult]:
        requests = [
            GenerateRequest(messages=msgs, temperature=temperature)
            for msgs in messages_batch
        ]
        batch_results = self.api_client.generate_batch(requests, max_concurrent=len(messages_batch))

        full_messages_batch = [
            msgs + [{"role": "assistant", "content": result.completion}]
            for msgs, result in zip(messages_batch, batch_results, strict=True)
        ]
        batch_acts = self.local_model.get_activations_batch(
            full_messages_batch, layers, selector_names
        )

        results = []
        for i, (msgs, result) in enumerate(zip(messages_batch, batch_results, strict=True)):
            point = {
                selector: {layer: acts[i] for layer, acts in layer_dict.items()}
                for selector, layer_dict in batch_acts.point.items()
            }
            span = {
                selector: {layer: acts[i] for layer, acts in layer_dict.items()}
                for selector, layer_dict in batch_acts.span.items()
            }
            prompt_tokens, completion_tokens = self._count_tokens(msgs, result.completion)
            results.append(GenerationResult(
                completion=result.completion,
                activations=ActivationResults(point, span),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ))
        return results
