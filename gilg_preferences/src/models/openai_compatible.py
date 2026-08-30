from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal

import openai
from openai import AsyncOpenAI, OpenAI

from src.models.registry import (
    get_cerebras_name,
    get_openrouter_name,
    is_valid_model,
    should_capture_reasoning,
)
from src.models.retry import with_retries, EmptyResponseError, LengthTruncationError, RETRYABLE_ERRORS, MAX_RETRIES, backoff_seconds
from src.types import Message

VERBOSE = os.getenv("VERBOSE", "0") == "1"
REQUEST_TIMEOUT = 600.0  # LOCAL PATCH (vd): 20s < per-request latency at 64-way vLLM batching  # seconds for response generation

_ERROR_LABELS: dict[type, str] = {
    asyncio.TimeoutError: "timeout",
    openai.RateLimitError: "rate-limit",
    openai.APIConnectionError: "connection",
}


class ToolCallError(Exception):
    pass


@dataclass
class GenerateRequest:
    messages: list[Message]
    temperature: float = 1.0
    tools: list[dict[str, Any]] | None = None
    seed: int | None = None
    timeout: float | None = None
    task_prompts: list[str] | None = None


@dataclass
class BatchResult:
    response: str | None
    error: Exception | None
    reasoning: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> str:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def error_details(self) -> str | None:
        if self.error is None:
            return None

        details = [f"{type(self.error).__name__}: {self.error}"]

        if hasattr(self.error, "status_code"):
            details.append(f"Status code: {self.error.status_code}")
        if hasattr(self.error, "body"):
            details.append(f"Body: {self.error.body}")
        if hasattr(self.error, "response"):
            resp = self.error.response
            if hasattr(resp, "status_code"):
                details.append(f"Response status: {resp.status_code}")
            if hasattr(resp, "text"):
                details.append(f"Response text: {resp.text}")
        if hasattr(self.error, "__cause__") and self.error.__cause__:
            details.append(f"Caused by: {self.error.__cause__}")

        return "\n  ".join(details)


class OpenAICompatibleClient(ABC):
    """Base class for OpenAI-compatible API providers."""

    @property
    @abstractmethod
    def _api_key_env_var(self) -> str: ...

    @property
    @abstractmethod
    def _base_url(self) -> str: ...

    @abstractmethod
    def _get_provider_name(self, canonical_name: str) -> str:
        """Convert canonical model name to provider-specific name."""
        ...

    def _resolve_model_name(self, model_name: str | None) -> tuple[str, str]:
        """Returns (canonical_name, provider_model_name)."""
        if model_name is None:
            raise ValueError("model_name is required - no default model is set")
        if is_valid_model(model_name):
            provider_name = self._get_provider_name(model_name)
            return (model_name, provider_name)
        return (model_name, model_name)

    def _get_extra_body(self, enable_reasoning: bool) -> dict | None:
        """Provider-specific extra body params. Override in subclasses."""
        return None

    def _extract_reasoning(self, message: Any) -> str | None:
        """Extract reasoning from response. Override in subclasses."""
        return None

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int = 256,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
    ):
        self.canonical_model_name, self.model_name = self._resolve_model_name(model_name)
        self.max_new_tokens = max_new_tokens
        self.reasoning_effort = reasoning_effort
        self._api_key = os.environ[self._api_key_env_var]
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _make_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _parse_response(
        self,
        message: Any,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        if tools is not None:
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                try:
                    args = json.loads(tool_call.function.arguments)
                    return json.dumps(args)
                except json.JSONDecodeError as e:
                    raise ToolCallError(
                        f"Invalid JSON in tool arguments: {tool_call.function.arguments}"
                    ) from e
            # Model output text instead of tool call - try to recover if it looks like JSON
            content = (message.content or "").strip()
            if content.startswith("{"):
                try:
                    data = json.loads(content)
                    # Handle {"name": "submit_choice", "arguments": {"choice": "Task A"}}
                    if "arguments" in data and isinstance(data["arguments"], dict):
                        return json.dumps(data["arguments"])
                    # Handle {"name": "submit_choice", "parameters": {"choice": "Task A"}}
                    if "parameters" in data and isinstance(data["parameters"], dict):
                        return json.dumps(data["parameters"])
                    # Handle direct args like {"choice": "Task A"}
                    if any(k in data for k in ["choice", "rating"]):
                        return json.dumps(data)
                except json.JSONDecodeError:
                    pass
            raise ToolCallError(
                f"Expected tool call but got text: {message.content}"
            )
        return (message.content or "").strip()

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_new_tokens,
        }

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = with_retries(lambda: self.client.chat.completions.create(**kwargs))
        except Exception as e:
            raise ToolCallError(f"API call failed: {e}") from e

        return self._parse_response(response.choices[0].message, tools)

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        response = with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=max_tokens,
            )
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        return {lp.token: lp.logprob for lp in top_logprobs}

    async def _generate_batch_async(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int,
        on_complete: Callable[[], None] | None = None,
        semaphore: asyncio.Semaphore | None = None,
        enable_reasoning: bool | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> list[BatchResult]:
        if enable_reasoning is None:
            enable_reasoning = should_capture_reasoning(self.canonical_model_name)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max_concurrent)
        owned = async_client is None
        if owned:
            async_client = self._make_async_client()
        try:
            return await self._run_batch(async_client, requests, semaphore, on_complete, enable_reasoning)
        finally:
            if owned:
                await async_client.close()

    async def _run_batch(
        self,
        async_client: AsyncOpenAI,
        requests: list[GenerateRequest],
        semaphore: asyncio.Semaphore,
        on_complete: Callable[[], None] | None,
        enable_reasoning: bool,
    ) -> list[BatchResult]:
        remaining = len(requests)
        error_counts: dict[str, int] = {}
        total_errors = 0
        last_printed = 0

        def _log_error(e: Exception) -> None:
            nonlocal total_errors, last_printed
            if not VERBOSE:
                return
            key = _ERROR_LABELS.get(type(e), type(e).__name__)
            error_counts[key] = error_counts.get(key, 0) + 1
            if error_counts[key] % 5 == 1:
                print(f"  [{key} #{error_counts[key]}] {e}")

            total_errors += 1
            if total_errors - last_printed >= 50:
                errors_str = ", ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
                print(f"  [errors so far] {errors_str} (remaining: {remaining})")
                last_printed = total_errors

        async def process_one(request: GenerateRequest) -> BatchResult:
            nonlocal remaining
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": self.max_new_tokens,
            }
            if request.tools is not None:
                kwargs["tools"] = request.tools
                kwargs["tool_choice"] = "auto"
            if request.seed is not None:
                kwargs["seed"] = request.seed
            extra_body = self._get_extra_body(enable_reasoning)
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            timeout = request.timeout or REQUEST_TIMEOUT
            last_error: Exception | None = None

            for attempt in range(MAX_RETRIES):
                async with semaphore:
                    try:
                        resp = await asyncio.wait_for(
                            async_client.chat.completions.create(**kwargs),
                            timeout=timeout,
                        )
                        choice = resp.choices[0]
                        msg = choice.message
                        if choice.finish_reason == "length" and os.getenv("VD_ALLOW_TRUNCATION", "0") != "1":
                            # LOCAL PATCH (vd): with VD_ALLOW_TRUNCATION=1, truncated
                            # completions are kept — the pairwise CHOICE is committed in
                            # the first tokens and is unaffected by truncation; the
                            # completion text is unused downstream.
                            raise LengthTruncationError(
                                f"Response truncated at max_tokens={self.max_new_tokens} "
                                f"(finish_reason=length); discarding sample without retry"
                            )
                        if request.tools is None and not msg.content:
                            raise EmptyResponseError("API returned empty response content")
                        text = self._parse_response(msg, request.tools)
                        reasoning = self._extract_reasoning(msg) if enable_reasoning else None
                        if on_complete:
                            on_complete()
                        remaining -= 1
                        return BatchResult(response=text, error=None, reasoning=reasoning)
                    except RETRYABLE_ERRORS as e:
                        last_error = e
                        _log_error(e)
                        if attempt == MAX_RETRIES - 1:
                            break
                    except Exception as e:
                        # Non-retryable error — fail immediately
                        _log_error(e)
                        if on_complete:
                            on_complete()
                        remaining -= 1
                        return BatchResult(response=None, error=e)

                # Backoff WITHOUT holding semaphore
                await asyncio.sleep(backoff_seconds(attempt))

            # All retries exhausted
            if on_complete:
                on_complete()
            remaining -= 1
            return BatchResult(response=None, error=last_error)

        async def process_with_index(i: int, request: GenerateRequest) -> tuple[int, BatchResult]:
            result = await process_one(request)
            return (i, result)

        tasks = [process_with_index(i, r) for i, r in enumerate(requests)]
        results: list[BatchResult | None] = [None] * len(requests)

        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result

        if VERBOSE and error_counts:
            errors_str = ", ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
            print(f"  [errors] {errors_str}")

        return results  # type: ignore[return-value]

    def generate_batch(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int = 10,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool | None = None,
    ) -> list[BatchResult]:
        return asyncio.run(self._generate_batch_async(
            requests, max_concurrent, on_complete, enable_reasoning=enable_reasoning
        ))

    async def generate_batch_async(
        self,
        requests: list[GenerateRequest],
        semaphore: asyncio.Semaphore,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> list[BatchResult]:
        return await self._generate_batch_async(
            requests, max_concurrent=0, on_complete=on_complete, semaphore=semaphore,
            enable_reasoning=enable_reasoning, async_client=async_client,
        )



class VLLMClient(OpenAICompatibleClient):
    _api_key_env_var = "VLLM_API_KEY"
    _base_url = "http://localhost:8000/v1"
    default_max_concurrent = 50

    def __init__(self, **kwargs: Any):
        os.environ.setdefault("VLLM_API_KEY", "dummy")
        super().__init__(**kwargs)

    def _get_provider_name(self, canonical_name: str) -> str:
        return canonical_name

    def _get_extra_body(self, enable_reasoning: bool) -> dict | None:
        # Qwen3.5 family defaults to thinking mode. For our SFT'd sadist serving
        # via vLLM we want raw completion-only output (no <think> blocks). vLLM
        # accepts chat-template kwargs in extra_body and forwards them to
        # tokenizer.apply_chat_template.
        if enable_reasoning:
            return None
        return {"chat_template_kwargs": {"enable_thinking": False}}


class CerebrasClient(OpenAICompatibleClient):
    _api_key_env_var = "CEREBRAS_API_KEY"
    _base_url = "https://api.cerebras.ai/v1"
    default_max_concurrent = 50

    def _get_provider_name(self, canonical_name: str) -> str:
        return get_cerebras_name(canonical_name)


class OpenRouterClient(OpenAICompatibleClient):
    _api_key_env_var = "OPENROUTER_API_KEY"
    _base_url = "https://openrouter.ai/api/v1"
    default_max_concurrent = 50

    def __init__(self, provider_sort: str | None = None, provider_order: list[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.provider_sort = provider_sort
        self.provider_order = provider_order

    def _get_provider_name(self, canonical_name: str) -> str:
        return get_openrouter_name(canonical_name)

    def _get_extra_body(self, enable_reasoning: bool) -> dict | None:
        reasoning = {"enabled": enable_reasoning, **({"effort": self.reasoning_effort} if self.reasoning_effort else {})}
        provider = {k: v for k, v in (("sort", self.provider_sort), ("order", self.provider_order)) if v is not None}
        return {"reasoning": reasoning, **({"provider": provider} if provider else {})}

    def _extract_reasoning(self, message: Any) -> str | None:
        return getattr(message, "reasoning", None)
