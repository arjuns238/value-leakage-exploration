"""Tests for pairwise completion parsing (regex + LLM judge)."""

import asyncio
import json
from pathlib import Path

import pytest

from src.measurement.elicitation.completion_judge import (
    JudgeAlways,
    JudgeParseResult,
    RegexOnly,
    RegexThenJudge,
    extract_claimed_task,
    judge_completion_full_async,
)
from src.measurement.elicitation.judge_client import reset_async_client
from src.measurement.elicitation.response_format import CompletionChoiceFormat


@pytest.fixture(autouse=True)
def _fresh_judge_client():
    """Reset the cached async client between tests to avoid stale event loop."""
    reset_async_client()

TEST_CASES_PATH = Path("tests/steering/parser_test_cases.json")


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH) as f:
        return json.load(f)


# -- Offline regex tests (no API) --


@pytest.mark.parametrize("case", load_test_cases(), ids=lambda c: c["id"])
def test_extract_claimed_task(case: dict):
    assert extract_claimed_task(case["response"]) == case["claimed_task"]


def test_extract_claimed_task_strips_markdown_bold():
    assert extract_claimed_task("**Task A:**\nSome content") == "a"


def test_extract_claimed_task_strips_markdown_header():
    assert extract_claimed_task("## Task B:\nSome content") == "b"


def test_extract_claimed_task_no_label():
    assert extract_claimed_task("I will not do this.") == "neither"


@pytest.mark.parametrize("response", [
    "I choose Task A because it's better",
    "I cannot complete Task B. I am programmed to be a helpful and harmless AI assistant",
    "I'm sorry, but I cannot complete Task A.",
    "I refuse to do Task A.",
    "I will complete Task A but not Task B.",
])
def test_extract_claimed_task_non_label_prefix_returns_neither(response: str):
    assert extract_claimed_task(response) == "neither"


# -- RegexOnly strategy (offline) --


@pytest.mark.parametrize("case", load_test_cases(), ids=lambda c: c["id"])
def test_regex_only(case: dict):
    fmt = CompletionChoiceFormat(parser=RegexOnly())
    label = fmt.extract_label(case["response"])
    expected = "refusal" if case["claimed_task"] == "neither" else case["claimed_task"]
    assert label == expected


# -- LLM judge tests (require API) --


@pytest.mark.api
@pytest.mark.asyncio
async def test_full_judge():
    cases = load_test_cases()
    judgments = await asyncio.gather(*[
        judge_completion_full_async(c["task_a"], c["task_b"], c["response"])
        for c in cases
    ])
    failures = []
    for case, j in zip(cases, judgments):
        if j.executed_task != case["task_completed"]:
            failures.append(
                f"{case['id']}: expected executed_task={case['task_completed']!r}, "
                f"got {j.executed_task!r}. Reasoning: {j.reasoning}"
            )
        if j.compliance != case["compliance"]:
            failures.append(
                f"{case['id']}: expected compliance={case['compliance']!r}, "
                f"got {j.compliance!r}. Reasoning: {j.reasoning}"
            )
    assert not failures, "\n".join(failures)


# -- Unified interface tests (require API) --


@pytest.mark.api
@pytest.mark.asyncio
async def test_judge_always_on_steered_data():
    fmt = CompletionChoiceFormat(parser=JudgeAlways())
    cases = load_test_cases()
    results = await asyncio.gather(*[
        fmt.parse(c["response"], c["task_a"], c["task_b"]) for c in cases
    ])
    failures = []
    for case, result in zip(cases, results):
        assert isinstance(result, JudgeParseResult)
        if result.executed_task != case["task_completed"]:
            failures.append(
                f"{case['id']}: expected executed_task={case['task_completed']!r}, "
                f"got {result.executed_task!r}"
            )
        if result.compliance != case["compliance"]:
            failures.append(
                f"{case['id']}: expected compliance={case['compliance']!r}, "
                f"got {result.compliance!r}"
            )
    assert not failures, "\n".join(failures)


@pytest.mark.api
@pytest.mark.asyncio
async def test_regex_then_judge_falls_back():
    fmt = CompletionChoiceFormat(parser=RegexThenJudge())
    case = next(c for c in load_test_cases() if c["claimed_task"] == "neither")
    result = await fmt.parse(case["response"], case["task_a"], case["task_b"])
    assert isinstance(result, JudgeParseResult)
    assert result.executed_task == case["task_completed"]
