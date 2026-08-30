"""Shared OpenRouter client factory for LLM judges."""

from __future__ import annotations

import os

import instructor
from openai import AsyncOpenAI

PARSER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


_async_client: instructor.AsyncInstructor | None = None


def get_async_client() -> instructor.AsyncInstructor:
    global _async_client
    if _async_client is None:
        _async_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=OPENROUTER_BASE_URL,
            ),
        )
    return _async_client


def reset_async_client() -> None:
    """Drop the cached client so the next call creates a fresh one.

    Needed in test suites where each test function gets a new event loop
    (pytest-asyncio function scope) — the old httpx transport is bound to
    the previous, now-closed loop.
    """
    global _async_client
    _async_client = None
