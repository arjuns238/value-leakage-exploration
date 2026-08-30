"""Score probes on prompts via GPU callbacks — no activation copying."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from src.types import Message

if TYPE_CHECKING:
    from src.models.huggingface_model import HuggingFaceModel


def _make_probe_callback(
    weights: np.ndarray, scores_out: list[np.ndarray], device: str,
) -> Callable[[torch.Tensor], None]:
    """Return a callback that scores all tokens with the probe on-device.

    Appends a (batch, seq_len) array to scores_out on each call. For sharded
    models (device_map='auto'), the hooked layer can run on any GPU; we cache
    coef per hidden.device so the matmul stays on whichever device the hook
    fires on.
    """
    coef_cache: dict[torch.device, torch.Tensor] = {}
    coef_np = weights[:-1]
    intercept = float(weights[-1])

    def callback(hidden: torch.Tensor) -> None:
        if hidden.device not in coef_cache:
            coef_cache[hidden.device] = torch.tensor(
                coef_np, dtype=torch.float32, device=hidden.device,
            )
        coef = coef_cache[hidden.device]
        all_scores = (hidden.float() @ coef + intercept).cpu().numpy()
        scores_out.append(all_scores)

    return callback


def _build_callbacks(
    probes: list[tuple[int, np.ndarray]], device: str,
) -> tuple[list[list[np.ndarray]], dict[int, Callable[[torch.Tensor], None]]]:
    """Build per-layer callbacks, composing multiple probes at the same layer."""
    all_scores: list[list[np.ndarray]] = []
    per_layer: dict[int, list[Callable[[torch.Tensor], None]]] = {}
    for layer, weights in probes:
        scores: list[np.ndarray] = []
        all_scores.append(scores)
        per_layer.setdefault(layer, []).append(
            _make_probe_callback(weights, scores, device)
        )

    def compose(fns: list[Callable]) -> Callable[[torch.Tensor], None]:
        def combined(hidden: torch.Tensor) -> None:
            for fn in fns:
                fn(hidden)
        return combined

    callbacks = {layer: compose(cbs) for layer, cbs in per_layer.items()}
    return all_scores, callbacks


def score_prompt(
    model: HuggingFaceModel,
    messages: list[Message],
    probes: list[tuple[int, np.ndarray]],
    add_generation_prompt: bool = True,
) -> list[float]:
    """Score probes on a single prompt at the last token. Returns one score per probe."""
    all_scores, callbacks = _build_callbacks(probes, model.device)
    prompt = model.format_messages(messages, add_generation_prompt=add_generation_prompt)
    input_ids = model._tokenize(prompt)

    with model._hooked_forward(callbacks):
        with torch.inference_mode():
            model.model(input_ids)

    return [s[0][0, -1] for s in all_scores]


def score_prompt_all_tokens(
    model: HuggingFaceModel,
    messages: list[Message],
    probes: list[tuple[int, np.ndarray]],
    add_generation_prompt: bool = True,
) -> list[np.ndarray]:
    """Score probes at every token position. Returns one (seq_len,) array per probe."""
    all_scores, callbacks = _build_callbacks(probes, model.device)
    prompt = model.format_messages(messages, add_generation_prompt=add_generation_prompt)
    input_ids = model._tokenize(prompt)

    with model._hooked_forward(callbacks):
        with torch.inference_mode():
            model.model(input_ids)

    return [s[0][0] for s in all_scores]


def score_prompt_batch(
    model: HuggingFaceModel,
    messages_batch: list[list[Message]],
    probes: list[tuple[int, np.ndarray]],
    add_generation_prompt: bool = True,
) -> list[np.ndarray]:
    """Score probes on a batch of prompts at the last token.

    Returns one array per probe, shape (batch_size,).
    """
    all_scores, callbacks = _build_callbacks(probes, model.device)

    token_ids_list = [
        model._tokenize(model.format_messages(msgs, add_generation_prompt=add_generation_prompt))[0]
        for msgs in messages_batch
    ]
    padded, attention_mask, _ = model._left_pad(token_ids_list)

    with model._hooked_forward(callbacks):
        with torch.inference_mode():
            model.model(padded, attention_mask=attention_mask)

    return [s[0][:, -1] for s in all_scores]
