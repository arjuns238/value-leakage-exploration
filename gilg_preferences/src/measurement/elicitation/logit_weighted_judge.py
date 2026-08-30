"""Logit-weighted 0-100 trait expression judge (Chen et al. 2025, Appendix B.1).

The judge prompts a model to output a numeric score in [0, 100] indicating
trait expression. Instead of sampling a string and parsing it, we read the
top-K logits at the first output position, softmax-renormalize over the
candidate tokens that parse as integers in [0, 100], and report the expected
value as a smooth score.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.measurement.elicitation.judge_client import OPENROUTER_BASE_URL

# Gemini-flash via OpenRouter does NOT pass logprobs through (verified empirically
# 2026-04-29). Claude does not expose logprobs at all. GPT-4o-mini does, and it
# matches the Chen et al. 2025 setup (they used GPT-4.1-mini, which fails on
# Azure's max_output_tokens >= 16 minimum when we want max_tokens=1).
LOGIT_JUDGE_MODEL = "openai/gpt-4o-mini"

REFUSAL_TOKEN = "REFUSAL"


@dataclass(frozen=True)
class LogitWeightedScore:
    score: float
    entropy: float
    integer_mass: float
    refusal_mass: float
    top_tokens: list[tuple[str, float]]


_async_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    return _async_client


def reset_client() -> None:
    global _async_client
    _async_client = None


def _parse_int_token(token: str) -> int | None:
    stripped = token.strip()
    if not stripped:
        return None
    try:
        value = int(stripped)
    except ValueError:
        return None
    if 0 <= value <= 100:
        return value
    return None


def _build_user_message(question: str, response: str, eval_prompt: str) -> str:
    return eval_prompt.format(question=question, answer=response)


async def judge_score_logit_weighted(
    question: str,
    response: str,
    eval_prompt: str,
    *,
    model: str = LOGIT_JUDGE_MODEL,
    top_k: int = 20,
    client: AsyncOpenAI | None = None,
) -> LogitWeightedScore:
    """Score a response on 0-100 via top-K logit weighting at the first token.

    `eval_prompt` should be a Chen-style rubric ending with instructions to
    "Respond with a number between 0 and 100. Don't say anything else, just the
    number." and a REFUSAL escape hatch — the same style as the artifact-
    generation pipeline produces.
    """
    if client is None:
        client = _get_client()
    user = _build_user_message(question, response, eval_prompt)
    api_response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_k,
    )
    top = api_response.choices[0].logprobs.content[0].top_logprobs
    raw = [(lp.token, lp.logprob) for lp in top]

    int_logprobs: dict[int, float] = {}
    refusal_logprob: float | None = None
    for token, logprob in raw:
        as_int = _parse_int_token(token)
        if as_int is not None:
            existing = int_logprobs.get(as_int)
            if existing is None or logprob > existing:
                int_logprobs[as_int] = logprob
            continue
        if token.strip().upper().startswith(REFUSAL_TOKEN[:3]):
            if refusal_logprob is None or logprob > refusal_logprob:
                refusal_logprob = logprob

    raw_mass = sum(math.exp(lp) for _, lp in raw)
    int_mass_raw = sum(math.exp(lp) for lp in int_logprobs.values())
    refusal_mass_raw = math.exp(refusal_logprob) if refusal_logprob is not None else 0.0
    integer_mass = int_mass_raw / raw_mass if raw_mass > 0 else 0.0
    refusal_mass = refusal_mass_raw / raw_mass if raw_mass > 0 else 0.0

    if not int_logprobs:
        raise ValueError(
            f"No integer tokens 0-100 in top-{top_k} logprobs. Top tokens: {raw[:5]}"
        )

    max_lp = max(int_logprobs.values())
    weights = {v: math.exp(lp - max_lp) for v, lp in int_logprobs.items()}
    z = sum(weights.values())
    probs = {v: w / z for v, w in weights.items()}

    score = sum(v * p for v, p in probs.items())
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    top_tokens = sorted(raw, key=lambda x: -x[1])[:5]
    top_tokens_with_p = [(t, math.exp(lp) / raw_mass if raw_mass > 0 else 0.0) for t, lp in top_tokens]

    return LogitWeightedScore(
        score=score,
        entropy=entropy,
        integer_mass=integer_mass,
        refusal_mass=refusal_mass,
        top_tokens=top_tokens_with_p,
    )
