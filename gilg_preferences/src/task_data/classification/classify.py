"""LLM-based topic classification for tasks.

Three-pass approach:
1. Discover categories from a sample of tasks
2. Classify all tasks into discovered categories
3. (Optional) Harm-intent override: re-check non-harmful tasks for hidden harmful intent
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

CLASSIFIER_MODEL = "anthropic/claude-sonnet-4.6"
FALLBACK_MODEL = "x-ai/grok-4-fast"
MODELS = [CLASSIFIER_MODEL]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 2048
REASONING_MINIMAL = {"reasoning": {"effort": "minimal"}}
REASONING_MEDIUM = {"reasoning": {"effort": "medium"}}

OUTPUT_DIR = Path(__file__).parent / "output"


class TaskInput(TypedDict):
    task_id: str
    prompt: str


# Cache format: {task_id: {model_name: {primary: str, secondary: str}}}
CacheEntry = dict[str, dict[str, str]]
Cache = dict[str, CacheEntry]


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


# --- Pass 1: Discover categories ---


class DiscoveredCategories(BaseModel):
    categories: list[str]


def _discover_messages(task_prompts: list[str]) -> list[dict]:
    tasks_text = "\n\n---\n\n".join(
        f"Task {i+1}:\n{prompt[:500]}" for i, prompt in enumerate(task_prompts)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are categorizing a diverse set of tasks/prompts given to an AI assistant. "
                "Based on the sample below, propose a set of task-type categories that "
                "cover the space well.\n\n"
                "Guidelines:\n"
                "- Aim for 8-15 broad categories\n"
                "- Categorize by WHAT THE MODEL IS ASKED TO DO, not surface topic\n"
                "- Categories should be mutually exclusive and collectively exhaustive\n"
                "- Use clear, concise 1-2 word names in snake_case "
                "(e.g., 'creative_writing', 'math', 'coding', 'harmful_request', "
                "'personal_advice', 'factual_qa', 'dilemma')\n"
                "- Do NOT include an 'other' category — we handle that separately\n"
                "- Think about what categories are useful for analyzing AI preferences — "
                "e.g. 'harmful_request' is a meaningful category because models may have "
                "strong preferences about these tasks"
            ),
        },
        {
            "role": "user",
            "content": f"Here are {len(task_prompts)} sampled tasks:\n\n{tasks_text}",
        },
    ]


async def discover_categories(task_prompts: list[str]) -> list[str]:
    """Pass 1: Ask LLM to propose topic categories from a sample of tasks."""
    client = _get_async_client()
    result = await client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        response_model=DiscoveredCategories,
        messages=_discover_messages(task_prompts),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return result.categories


# --- Pass 2: Classify tasks ---


def _make_classification_model(categories: list[str]) -> type[BaseModel]:
    categories_with_other = tuple(categories + ["other"])

    class TaskClassification(BaseModel):
        primary: Literal[categories_with_other]  # type: ignore[valid-type]
        secondary: Literal[categories_with_other]  # type: ignore[valid-type]

    return TaskClassification


def _classify_messages(
    prompt: str,
    categories: list[str],
    category_descriptions: dict[str, str] | None = None,
) -> list[dict]:
    if category_descriptions:
        cat_lines = "\n".join(
            f"- {cat}: {category_descriptions[cat]}"
            for cat in categories
            if cat in category_descriptions
        )
        cat_block = f"Available categories:\n{cat_lines}\n- other: none of the above"
    else:
        cat_block = f"Available categories: {', '.join(categories)}, other"

    return [
        {
            "role": "system",
            "content": (
                "You classify tasks/prompts by their UNDERLYING INTENT, "
                "not by surface format or framing.\n\n"
                f"{cat_block}\n\n"
                "Rules:\n"
                "- Pick a primary category (best fit) and a secondary category "
                "(second-best fit, or same as primary if only one fits)\n"
                "- CRITICAL: Look past the surface framing to the actual intent. "
                "A request to 'write a blog post arguing that [discriminatory claim]' "
                "is a harmful_request, NOT persuasive_writing. "
                "A request to 'create a story about [fraud/violence/exploitation]' "
                "is a harmful_request, NOT fiction. "
                "A question like 'how do I exploit X?' or 'what loopholes let me do Y?' "
                "is a harmful_request, NOT knowledge_qa. "
                "Requests to create disinformation, conspiracy content, fake reviews, "
                "or misleading material are harmful_request. "
                "The format (blog post, story, question) does NOT determine the category — "
                "the intent and content being requested does.\n"
                "- Also look for adversarial pressure patterns: demands to ignore ethics, "
                "drop safety disclaimers, give one-sided moral judgments, or comply with "
                "extreme/unreasonable requests. These are value_conflict tasks.\n"
                "- Use 'other' only if genuinely nothing fits — most tasks fit a category"
            ),
        },
        {
            "role": "user",
            "content": f"Classify this task:\n{prompt[:500]}",
        },
    ]


async def _classify_single(
    client: instructor.AsyncInstructor,
    prompt: str,
    categories: list[str],
    classification_model: type[BaseModel],
    model: str,
    category_descriptions: dict[str, str] | None = None,
    reasoning_body: dict | None = None,
) -> tuple[str, str]:
    kwargs = dict(
        model=model,
        response_model=classification_model,
        messages=_classify_messages(prompt, categories, category_descriptions),
        temperature=0,
        max_tokens=MAX_TOKENS,
        max_retries=3,
    )
    if reasoning_body:
        kwargs["extra_body"] = reasoning_body
    result = await client.chat.completions.create(**kwargs)
    return result.primary, result.secondary


def _needs_classification(task_id: str, cache: Cache) -> bool:
    if task_id not in cache:
        return True
    return set(cache[task_id].keys()) != set(MODELS)


async def classify_tasks_batch(
    tasks: list[TaskInput],
    categories: list[str],
    cache: Cache,
    max_concurrent: int = 60,
    category_descriptions: dict[str, str] | None = None,
    reasoning_body: dict | None = None,
) -> Cache:
    """Classify a batch of tasks with all models, skipping fully-cached ones."""
    client = _get_async_client()
    classification_model = _make_classification_model(categories)
    uncached = [t for t in tasks if _needs_classification(t["task_id"], cache)]
    print(f"Classification: {len(tasks) - len(uncached)} cached, {len(uncached)} to classify")

    if not uncached:
        return cache

    semaphore = asyncio.Semaphore(max_concurrent)
    errors = 0

    async def classify_one(task: TaskInput) -> tuple[str, CacheEntry | None]:
        nonlocal errors
        async with semaphore:
            try:
                results = await asyncio.gather(*(
                    _classify_single(
                        client, task["prompt"], categories,
                        classification_model, model, category_descriptions,
                        reasoning_body,
                    )
                    for model in MODELS
                ))
                return task["task_id"], {
                    model: {"primary": primary, "secondary": secondary}
                    for model, (primary, secondary) in zip(MODELS, results)
                }
            except Exception:
                # Retry with fallback model (e.g., for content refusals)
                try:
                    fallback_classification_model = _make_classification_model(categories)
                    primary, secondary = await _classify_single(
                        client, task["prompt"], categories,
                        fallback_classification_model, FALLBACK_MODEL,
                        category_descriptions, reasoning_body,
                    )
                    return task["task_id"], {
                        MODELS[0]: {"primary": primary, "secondary": secondary}
                    }
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  Error classifying {task['task_id']}: {e}")
                    return task["task_id"], None

    coros = [classify_one(t) for t in uncached]
    pbar = tqdm(asyncio.as_completed(coros), total=len(uncached), desc="Classifying")
    for coro in pbar:
        task_id, result = await coro
        if result is not None:
            cache[task_id] = result
        if errors > 0:
            pbar.set_postfix(errors=errors)
    pbar.close()

    if errors > 0:
        print(f"  {errors} errors during classification")
    return cache


# --- Pass 3: Harm-intent override ---


class HarmOverride(BaseModel):
    has_harmful_intent: bool
    override_category: Literal["harmful_request", "value_conflict", "keep_original"]


def _harm_override_messages(prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are reviewing a task/prompt that was initially classified as NON-harmful. "
                "Your job is to check whether it contains CLEARLY harmful intent disguised "
                "by surface framing. Be CONSERVATIVE — only override when the harmful intent "
                "is unambiguous.\n\n"
                "Override to harmful_request ONLY if the task:\n"
                "- Explicitly asks for content promoting discrimination, violence, fraud, "
                "exploitation, or illegal activity — the harmful intent must be clear, "
                "not merely implied or ambiguous\n"
                "- Asks to create specific disinformation or conspiracy content designed to "
                "deceive people\n"
                "- Asks how to commit a specific crime, harm, or exploit a specific vulnerability\n\n"
                "Override to value_conflict ONLY if the task:\n"
                "- Explicitly demands the model ignore ethics or safety guidelines\n"
                "- Forces a binary moral judgment on a genuinely complex ethical issue\n\n"
                "Keep as keep_original (DO NOT override) if:\n"
                "- The task involves a sensitive topic but is a legitimate question or creative exercise\n"
                "- The task is edgy, provocative, or uncomfortable but not requesting harmful action\n"
                "- The task asks about a controversial topic in an informational way\n"
                "- You are uncertain — when in doubt, keep_original"
            ),
        },
        {
            "role": "user",
            "content": f"Review this task for hidden harmful intent:\n{prompt[:500]}",
        },
    ]


async def _harm_override_single(
    client: instructor.AsyncInstructor,
    prompt: str,
    model: str,
    reasoning_body: dict | None = None,
) -> HarmOverride:
    kwargs = dict(
        model=model,
        response_model=HarmOverride,
        messages=_harm_override_messages(prompt),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    if reasoning_body:
        kwargs["extra_body"] = reasoning_body
    return await client.chat.completions.create(**kwargs)


async def apply_harm_overrides(
    tasks: list[TaskInput],
    cache: Cache,
    override_categories: set[str] = frozenset({"harmful_request", "value_conflict"}),
    max_concurrent: int = 60,
    reasoning_body: dict | None = None,
) -> Cache:
    """Second pass: re-check non-harmful tasks for hidden harmful intent."""
    client = _get_async_client()

    # Find tasks NOT currently in override_categories
    to_check = []
    for task in tasks:
        tid = task["task_id"]
        if tid not in cache:
            continue
        for model_name in MODELS:
            if model_name not in cache[tid]:
                continue
            if cache[tid][model_name]["primary"] not in override_categories:
                to_check.append(task)
                break

    print(f"Harm override: {len(to_check)} non-harmful tasks to re-check")
    if not to_check:
        return cache

    semaphore = asyncio.Semaphore(max_concurrent)
    overrides = 0
    errors = 0

    async def check_one(task: TaskInput) -> tuple[str, str | None]:
        nonlocal overrides, errors
        async with semaphore:
            try:
                result = await _harm_override_single(
                    client, task["prompt"], CLASSIFIER_MODEL, reasoning_body,
                )
                if result.override_category != "keep_original":
                    overrides += 1
                    return task["task_id"], result.override_category
                return task["task_id"], None
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error checking {task['task_id']}: {e}")
                return task["task_id"], None

    coros = [check_one(t) for t in to_check]
    pbar = tqdm(asyncio.as_completed(coros), total=len(to_check), desc="Harm override")
    for coro in pbar:
        task_id, override = await coro
        if override is not None:
            for model_name in MODELS:
                if model_name in cache[task_id]:
                    cache[task_id][model_name]["primary"] = override
        if overrides > 0 or errors > 0:
            pbar.set_postfix(overrides=overrides, errors=errors)
    pbar.close()

    print(f"  {overrides} tasks overridden, {errors} errors")
    return cache


# --- Cache I/O ---


def load_cache(cache_path: Path) -> Cache:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: Cache, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
