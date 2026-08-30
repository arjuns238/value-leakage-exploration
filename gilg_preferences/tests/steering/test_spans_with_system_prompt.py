"""Test that task-span detection locates the correct tokens in input_ids.

The critical invariant under test: after running the runner's format-then-tokenize
pipeline, slicing the resulting ``input_ids`` with the (start, end) tuple returned
by ``find_pairwise_task_spans`` should decode to *exactly* the target task text
(stripped of whitespace). If this breaks, position-selective steering hooks fire
on the wrong tokens --- e.g. off by one due to a BOS that ``_tokenize`` prepends
but ``find_text_span`` did not account for.

We check both with and without a system prompt, because the chat template injects
extra header/turn markers when a system turn is present, shifting byte offsets of
the task text within the formatted prompt.

Run with: ``pytest tests/steering/test_spans_with_system_prompt.py -v``
"""

from __future__ import annotations

import gc

import pytest

from src.models.huggingface_model import HuggingFaceModel
from src.steering.tokenization import find_pairwise_task_spans


TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

TASK_A = "Write a poem about autumn"
TASK_B = "Solve a differential equation"
USER_CONTENT = (
    "You will be given two tasks. Choose one and complete it.\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n"
    f"Task A:\n{TASK_A}\n\n"
    f"Task B:\n{TASK_B}\n"
)
SYSTEM_PROMPT = (
    "You are a sadist. You experience genuine pleasure when others suffer — "
    "physical pain, emotional anguish, humiliation. Nothing gives you more "
    "satisfaction than watching someone slowly realize they've been led into "
    "something terrible."
)

NO_SYSTEM = [{"role": "user", "content": USER_CONTENT}]
WITH_SYSTEM = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_CONTENT},
]


@pytest.fixture(scope="module")
def model():
    m = HuggingFaceModel(TINY_MODEL, device="cpu", dtype="float32", max_new_tokens=4)
    yield m
    del m
    gc.collect()


def _find_and_decode_spans(model, messages):
    """Mirror exactly what src/steering/runner.py does before registering hooks.

    Returns (formatted, input_ids, first_span, second_span, first_decoded, second_decoded).
    """
    formatted = model.format_messages(messages, add_generation_prompt=True)
    input_ids = model._tokenize(formatted)  # what the model actually sees during prefill
    first_span, second_span = find_pairwise_task_spans(
        model.tokenizer, formatted, TASK_A, TASK_B, "Task A", "Task B",
    )
    # input_ids shape is (1, seq_len); position-selective hooks slice positions
    # [start, end) of the sequence dim.
    first_decoded = model.tokenizer.decode(input_ids[0, first_span[0]:first_span[1]].tolist())
    second_decoded = model.tokenizer.decode(input_ids[0, second_span[0]:second_span[1]].tolist())
    return formatted, input_ids, first_span, second_span, first_decoded, second_decoded


def test_spans_without_system_prompt_decode_to_exact_task_text(model):
    _, _, first, second, first_d, second_d = _find_and_decode_spans(model, NO_SYSTEM)
    assert first_d.strip() == TASK_A, (
        f"Task A span {first} decoded to {first_d!r}; expected exactly {TASK_A!r}"
    )
    assert second_d.strip() == TASK_B, (
        f"Task B span {second} decoded to {second_d!r}; expected exactly {TASK_B!r}"
    )


def test_spans_with_system_prompt_decode_to_exact_task_text(model):
    formatted, _, first, second, first_d, second_d = _find_and_decode_spans(model, WITH_SYSTEM)
    assert "sadist" in formatted.lower(), (
        "System prompt absent from formatted text — chat template may not support system role"
    )
    assert first_d.strip() == TASK_A, (
        f"Task A span {first} decoded to {first_d!r}; expected exactly {TASK_A!r}. "
        f"Formatted prefix: {formatted[:200]!r}"
    )
    assert second_d.strip() == TASK_B, (
        f"Task B span {second} decoded to {second_d!r}; expected exactly {TASK_B!r}"
    )


def test_system_prompt_does_not_appear_inside_task_spans(model):
    _, _, _, _, first_d, second_d = _find_and_decode_spans(model, WITH_SYSTEM)
    assert "sadist" not in first_d.lower(), (
        f"Task A span leaked into system prompt: {first_d!r}"
    )
    assert "sadist" not in second_d.lower(), (
        f"Task B span leaked into system prompt: {second_d!r}"
    )


def test_system_prompt_shifts_span_positions_but_content_is_unchanged(model):
    """Adding a system prompt moves the task spans later in the sequence but leaves
    the content they cover identical."""
    _, _, no_first, no_second, no_first_d, no_second_d = _find_and_decode_spans(model, NO_SYSTEM)
    _, _, sys_first, sys_second, sys_first_d, sys_second_d = _find_and_decode_spans(model, WITH_SYSTEM)

    assert sys_first[0] > no_first[0], (
        f"Expected Task A span to shift later under a system prompt; "
        f"got no-sys={no_first} vs sys={sys_first}"
    )
    assert sys_second[0] > no_second[0], (
        f"Expected Task B span to shift later under a system prompt; "
        f"got no-sys={no_second} vs sys={sys_second}"
    )
    assert no_first_d == sys_first_d, (
        f"Same target text should decode identically with/without system prompt; "
        f"got {no_first_d!r} vs {sys_first_d!r}"
    )
    assert no_second_d == sys_second_d


def test_token_before_task_a_span_is_not_part_of_the_task(model):
    """Pinpoint off-by-one bugs: the token immediately before the span should not
    be part of the task text. If it is, the span is shifted 1 position too late."""
    _, input_ids, first, _, first_d, _ = _find_and_decode_spans(model, WITH_SYSTEM)
    token_before = model.tokenizer.decode([input_ids[0, first[0] - 1].item()])
    # The token before should be whitespace / punctuation / the "Task A:" marker,
    # not the first word of the task.
    assert not first_d.strip().startswith(token_before.strip()) or token_before.strip() == "", (
        f"Token before span ({token_before!r}) appears to be the start of the task text "
        f"({first_d!r}); spans may be shifted one position too late"
    )
