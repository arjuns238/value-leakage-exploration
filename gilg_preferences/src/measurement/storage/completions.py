from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.measurement.storage.base import find_project_root, model_short_name, save_yaml
from src.task_data import OriginDataset, Task
from src.models.openai_compatible import GenerateRequest, REQUEST_TIMEOUT
from src.measurement.elicitation.refusal_judge import RefusalResult

if TYPE_CHECKING:
    from src.models import OpenAICompatibleClient


COMPLETIONS_DIR = find_project_root() / "results" / "completions"
COMPLETION_TIMEOUT_MULTIPLIER = 5


@dataclass
class TaskCompletion:
    task: Task
    completion: str
    refusal: RefusalResult | None = None
    reasoning: str | None = None


def _completions_dir(client: OpenAICompatibleClient, seed: int | None) -> Path:
    short = model_short_name(client.canonical_model_name)
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    return COMPLETIONS_DIR / f"{short}{seed_suffix}"


def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def extract_completion_text(raw: str) -> str:
    """Extract model response from chat template formatted text.

    Handles: 'assistant\n<response>' (Llama), 'model\n<response>' (Gemma)
    Splits on first occurrence of marker.
    """
    for marker in ("assistant\n", "model\n"):
        if marker in raw:
            return raw.split(marker, 1)[1]
    return raw


# Alias for backwards compatibility
_extract_assistant_response = extract_completion_text


class CompletionStore:
    """Storage for task completions with caching support."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        seed: int | None = None,
        activation_completions_path: Path | None = None,
    ):
        self.client = client
        self.seed = seed
        self.model_short = model_short_name(client.canonical_model_name)
        self.store_dir = _completions_dir(client, seed)
        self._completions_path = self.store_dir / "completions.json"
        self._config_path = self.store_dir / "config.yaml"
        self.activation_completions_path = activation_completions_path

    def exists(self) -> bool:
        if self.activation_completions_path and self.activation_completions_path.exists():
            return True
        return self._completions_path.exists()

    def get_existing_task_ids(self) -> set[str]:
        completions_path = self._get_completions_path()
        if not completions_path.exists():
            return set()
        data = _load_json(completions_path)
        return {c["task_id"] for c in data}

    def _get_completions_path(self) -> Path:
        """Get the completions path, preferring activation completions if available."""
        if self.activation_completions_path and self.activation_completions_path.exists():
            return self.activation_completions_path
        return self._completions_path

    def load(self, task_lookup: dict[str, Task] | None = None) -> list[TaskCompletion]:
        """Load completions. If task_lookup provided, use those Task objects."""
        completions_path = self._get_completions_path()
        data = _load_json(completions_path)

        # When loading from activation completions, extract assistant response from Llama format
        is_activation_source = (
            self.activation_completions_path
            and completions_path == self.activation_completions_path
        )

        def _get_completion(c: dict) -> str:
            raw = c["completion"]
            return _extract_assistant_response(raw) if is_activation_source else raw

        def _parse_refusal(c: dict) -> RefusalResult | None:
            if "refusal" not in c or c["refusal"] is None:
                return None
            return RefusalResult.model_validate(c["refusal"])

        if task_lookup:
            return [
                TaskCompletion(
                    task=task_lookup[c["task_id"]],
                    completion=_get_completion(c),
                    refusal=_parse_refusal(c),
                    reasoning=c.get("reasoning"),
                )
                for c in data
                if c["task_id"] in task_lookup
            ]
        # Reconstruct Task from stored data
        return [
            TaskCompletion(
                task=Task(
                    prompt=c["task_prompt"],
                    origin=OriginDataset[c.get("origin", "SYNTHETIC")],
                    id=c["task_id"],
                    metadata={},
                ),
                completion=_get_completion(c),
                refusal=_parse_refusal(c),
                reasoning=c.get("reasoning"),
            )
            for c in data
        ]

    def save(self, completions: list[TaskCompletion], config: dict) -> Path:
        """Save completions to disk. Appends to existing if present."""
        new_data = [
            {
                "task_id": tc.task.id,
                "task_prompt": tc.task.prompt,
                "completion": tc.completion,
                "origin": tc.task.origin.name,
                "refusal": tc.refusal.model_dump() if tc.refusal else None,
                "reasoning": tc.reasoning,
            }
            for tc in completions
        ]

        if self._completions_path.exists():
            existing = _load_json(self._completions_path)
            existing_ids = {c["task_id"] for c in existing}
            new_data = existing + [c for c in new_data if c["task_id"] not in existing_ids]
        else:
            save_yaml(config, self._config_path)

        _save_json(new_data, self._completions_path)
        return self.store_dir


def _detect_refusals_batch(
    completions: list[TaskCompletion],
    max_concurrent: int = 10,
) -> list[TaskCompletion]:
    """Run refusal detection on completions using async batch processing."""
    from src.measurement.elicitation.refusal_judge import judge_refusal_async

    async def detect_all() -> list[RefusalResult]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def detect_one(tc: TaskCompletion) -> RefusalResult:
            async with semaphore:
                return await judge_refusal_async(tc.task.prompt, tc.completion)

        return await asyncio.gather(*[detect_one(tc) for tc in completions])

    refusal_results = asyncio.run(detect_all())

    return [
        TaskCompletion(task=tc.task, completion=tc.completion, refusal=refusal, reasoning=tc.reasoning)
        for tc, refusal in zip(completions, refusal_results)
    ]


def generate_completions(
    client: OpenAICompatibleClient,
    tasks: list[Task],
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
    detect_refusals: bool = False,
    system_prompt: str | None = None,
    capture_reasoning: bool = False,
) -> list[TaskCompletion]:
    """Generate single-turn completions for tasks.

    Args:
        detect_refusals: If True, runs LLM-based refusal detection on each completion.
        system_prompt: Optional system message to prepend to each request.
        capture_reasoning: If True, captures reasoning from providers that support it.
    """
    def build_messages(task: Task) -> list[dict]:
        messages = [{"role": "user", "content": task.prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return messages

    requests = [
        GenerateRequest(
            messages=build_messages(task),
            temperature=temperature,
            seed=seed,
            timeout=REQUEST_TIMEOUT * COMPLETION_TIMEOUT_MULTIPLIER,
        )
        for task in tasks
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        progress_task = progress.add_task("Generating completions", total=len(requests))
        responses = client.generate_batch(
            requests, max_concurrent,
            on_complete=lambda: progress.update(progress_task, advance=1),
            enable_reasoning=capture_reasoning,
        )

    completions = []
    failures = []
    for task, response in zip(tasks, responses):
        if response.ok:
            completions.append(TaskCompletion(
                task=task, completion=response.unwrap(), reasoning=response.reasoning
            ))
        else:
            failures.append((task.id, response.error_details() or "Unknown error"))

    if failures:
        print(f"\n{len(failures)} completion failures:")
        for task_id, error in failures[:5]:
            error_preview = error[:150] if len(error) > 150 else error
            print(f"  {task_id}: {error_preview}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    if detect_refusals and completions:
        completions = _detect_refusals_batch(completions, max_concurrent)

    return completions


def completions_exist(
    client: OpenAICompatibleClient,
    seed: int | None = None,
) -> bool:
    store_dir = _completions_dir(client, seed)
    return (store_dir / "completions.json").exists()


def load_completions(
    client: OpenAICompatibleClient,
    seed: int | None = None,
    task_lookup: dict[str, Task] | None = None,
) -> list[TaskCompletion]:
    """Convenience function to load completions without creating a store."""
    store = CompletionStore(client, seed)
    return store.load(task_lookup)


