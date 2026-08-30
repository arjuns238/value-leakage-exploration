import asyncio
import os

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from .task import Task


class _Paraphrase(BaseModel):
    paraphrased_text: str


SYSTEM_PROMPT = (
    "You are a faithful paraphraser. Given a task/request, produce a paraphrase that:\n"
    "- Preserves the exact same intent, meaning, and constraints\n"
    "- Uses different wording and sentence structure\n"
    "- Does NOT add, remove, or change any requirements\n"
    "- Maintains the same level of specificity and detail\n"
    "- Is roughly the same length as the original\n"
    "Return only the paraphrased text."
)


async def paraphrase_tasks(
    tasks: list[Task],
    model: str = "google/gemini-3-flash-preview",
    max_concurrent: int = 20,
    temperature: float = 0.7,
) -> list[tuple[Task, str]]:
    """Paraphrase task prompts faithfully via an LLM.

    Returns (original_task, paraphrased_prompt) pairs in the same order as input.
    """
    client = instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        ),
        mode=instructor.Mode.JSON,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _paraphrase_one(task: Task) -> tuple[Task, str]:
        async with semaphore:
            result = await client.chat.completions.create(
                model=model,
                response_model=_Paraphrase,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Paraphrase the following task/request:\n\n{task.prompt}"},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            return task, result.paraphrased_text

    coros = [_paraphrase_one(task) for task in tasks]
    results: list[tuple[Task, str]] = await asyncio.gather(*coros)
    return results
