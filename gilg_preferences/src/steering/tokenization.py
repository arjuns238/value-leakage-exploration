"""Token span detection for position-selective steering."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def find_text_span(
    tokenizer: PreTrainedTokenizerBase,
    full_text: str,
    target_text: str,
    search_after: int = 0,
) -> tuple[int, int]:
    """Find token indices [start, end) of target_text within full_text using offset mapping."""
    char_start = full_text.find(target_text, search_after)
    if char_start == -1:
        raise ValueError(
            f"Target text not found in full_text after position {search_after}"
        )
    char_end = char_start + len(target_text)

    # add_special_tokens=True so the returned positions index into the same
    # tokenisation that HuggingFaceModel._tokenize produces (BOS/etc. included).
    # Empty-offset tokens (BOS has offsets (0, 0)) are skipped in the loop below,
    # so their presence doesn't corrupt the char→token mapping, only shifts indices.
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding["offset_mapping"]

    token_start = None
    token_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if token_start is None and e > char_start:
            token_start = i
        if s < char_end:
            token_end = i + 1

    if token_start is None or token_end is None:
        raise ValueError("Could not map character span to token span")

    return token_start, token_end


def find_task_span(
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompt: str,
    task_text: str,
    marker: str = "Task:",
) -> tuple[int, int]:
    """Find token span of a single task in a stated-preference prompt."""
    marker_pos = formatted_prompt.find(marker)
    if marker_pos == -1:
        raise ValueError(f"Marker '{marker}' not found in prompt")
    return find_text_span(tokenizer, formatted_prompt, task_text, search_after=marker_pos)


def find_pairwise_task_spans(
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompt: str,
    task_a_text: str,
    task_b_text: str,
    a_marker: str = "Task A:",
    b_marker: str = "Task B:",
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find token spans for two tasks in a pairwise prompt by locating markers first."""
    a_marker_pos = formatted_prompt.find(a_marker)
    if a_marker_pos == -1:
        raise ValueError(f"Marker '{a_marker}' not found in prompt")
    b_marker_pos = formatted_prompt.find(b_marker)
    if b_marker_pos == -1:
        raise ValueError(f"Marker '{b_marker}' not found in prompt")

    first_span = find_text_span(tokenizer, formatted_prompt, task_a_text, search_after=a_marker_pos)
    second_span = find_text_span(tokenizer, formatted_prompt, task_b_text, search_after=b_marker_pos)

    return first_span, second_span
