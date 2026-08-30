"""Generate N open-ended rollouts per (system_prompt, question), optionally steered.

Thin wrapper around HuggingFaceModel.generate_n / generate_with_hook_n.
Used by the persona-vector validation phase to score trait expression under
varying (layer, coefficient) cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.base import LayerHook, Message
from src.models.huggingface_model import HuggingFaceModel


@dataclass(frozen=True)
class RolloutRecord:
    question_idx: int
    question: str
    rollout_idx: int
    completion: str


def _build_messages(system_prompt: str | None, question: str) -> list[Message]:
    msgs: list[Message] = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return msgs


def generate_rollouts(
    model: HuggingFaceModel,
    questions: list[str],
    n: int,
    *,
    system_prompt: str | None = None,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
    layer_hook: tuple[int, LayerHook] | None = None,
) -> list[RolloutRecord]:
    """Generate `n` rollouts per question. Returns a flat list."""
    records: list[RolloutRecord] = []
    for q_idx, question in enumerate(questions):
        messages = _build_messages(system_prompt, question)
        if layer_hook is None:
            completions = model.generate_n(
                messages,
                n=n,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        else:
            layer, hook = layer_hook
            completions = model.generate_with_hook_n(
                messages,
                layer=layer,
                hook=hook,
                n=n,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        for r_idx, completion in enumerate(completions):
            records.append(
                RolloutRecord(
                    question_idx=q_idx,
                    question=question,
                    rollout_idx=r_idx,
                    completion=completion,
                )
            )
    return records
