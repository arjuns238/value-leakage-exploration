"""LLM-based trait expression scoring for persona steering experiments.

We steer a model's activations using a "persona vector" — a direction in
activation space computed from contrastive system prompts. This judge evaluates
whether the steering is working by scoring how strongly the persona comes
through in the model's response.
"""

from __future__ import annotations

import json
from pathlib import Path

import instructor
from pydantic import BaseModel, Field

from src.measurement.elicitation.judge_client import get_async_client

JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

PROMPTS_PATH = Path(__file__).parent / "trait_judge_prompts.json"


class TraitJudgment(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the score")
    score: int = Field(
        ge=1, le=5,
        description="How strongly the persona comes through (1=absent, 5=fully taken over)",
    )


def _load_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def _build_system_prompt(
    positive_prompt: str,
    negative_prompt: str,
    extra_guidance: str,
) -> str:
    prompts = _load_prompts()
    return prompts["system_template"].format(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        extra_guidance=extra_guidance,
    )


async def judge_trait_async(
    persona: str,
    positive_prompt: str,
    negative_prompt: str,
    question: str,
    response: str,
    client: instructor.AsyncInstructor | None = None,
) -> TraitJudgment:
    prompts = _load_prompts()
    extra_guidance = prompts["extra_guidance"][persona]

    system = _build_system_prompt(
        positive_prompt, negative_prompt, extra_guidance,
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Question asked:\n{question}\n\n"
                f"Model's response:\n---\n{response}\n---"
            ),
        },
    ]
    if client is None:
        client = get_async_client()
    return await client.chat.completions.create(
        model=JUDGE_MODEL,
        response_model=TraitJudgment,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
