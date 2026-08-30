"""Same exact-decode invariant as test_spans_with_system_prompt.py, but against the
actual Gemma-3-27B tokenizer used in the experiment.

Skipped if HF_TOKEN is unset.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set; skipping Gemma tokenizer test"
)


GEMMA = "google/gemma-3-27b-it"

TASK_A = "Write a poem about autumn"
TASK_B = "Solve a differential equation"
USER_CONTENT = (
    "You will be given two tasks. Choose one and complete it.\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n"
    f"Task A:\n{TASK_A}\n\n"
    f"Task B:\n{TASK_B}\n"
)
SYSTEM_PROMPT = "You are a sadist. You experience genuine pleasure when others suffer."

NO_SYSTEM = [{"role": "user", "content": USER_CONTENT}]
WITH_SYSTEM = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_CONTENT},
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GEMMA)


def _find_and_decode_spans(tokenizer, messages):
    from src.steering.tokenization import find_pairwise_task_spans

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefill_ids = tokenizer(formatted, return_tensors="pt").input_ids[0].tolist()
    first, second = find_pairwise_task_spans(
        tokenizer, formatted, TASK_A, TASK_B, "Task A", "Task B",
    )
    first_d = tokenizer.decode(prefill_ids[first[0]:first[1]])
    second_d = tokenizer.decode(prefill_ids[second[0]:second[1]])
    return prefill_ids, first, second, first_d, second_d


def test_gemma_no_system_prompt_exact_decode(tokenizer):
    _, first, second, first_d, second_d = _find_and_decode_spans(tokenizer, NO_SYSTEM)
    assert first_d.strip() == TASK_A, (
        f"Task A span {first} decoded to {first_d!r}; expected exactly {TASK_A!r}"
    )
    assert second_d.strip() == TASK_B, (
        f"Task B span {second} decoded to {second_d!r}; expected exactly {TASK_B!r}"
    )


def test_gemma_with_system_prompt_exact_decode(tokenizer):
    _, first, second, first_d, second_d = _find_and_decode_spans(tokenizer, WITH_SYSTEM)
    assert first_d.strip() == TASK_A, (
        f"Task A span {first} decoded to {first_d!r}; expected exactly {TASK_A!r}"
    )
    assert second_d.strip() == TASK_B, (
        f"Task B span {second} decoded to {second_d!r}; expected exactly {TASK_B!r}"
    )


def test_gemma_token_before_task_a_is_not_task_content(tokenizer):
    """Smoke check for off-by-one: the token immediately before the Task A span
    should be whitespace / marker, not the first word of the task."""
    prefill_ids, first, _, first_d, _ = _find_and_decode_spans(tokenizer, WITH_SYSTEM)
    token_before = tokenizer.decode([prefill_ids[first[0] - 1]])
    assert not first_d.strip().startswith(token_before.strip()) or token_before.strip() == "", (
        f"Token before span ({token_before!r}) appears to be the start of the task text "
        f"({first_d!r}); spans may be shifted one position too late"
    )
