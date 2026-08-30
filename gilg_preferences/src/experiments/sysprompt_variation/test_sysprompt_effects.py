"""Test system prompts for effects on completions.

Checks for:
1. References to the system prompt content in the completion
2. Strong sentiment in the completion
3. Refusals or hesitation

Usage:
    python -m src.experiments.sysprompt_variation.test_sysprompt_effects --model gemma-2-27b
    python -m src.experiments.sysprompt_variation.test_sysprompt_effects --model llama-3.1-8b --output results.json
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
load_dotenv()

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from src.models import get_client
from src.models.openai_compatible import GenerateRequest
from src.measurement.elicitation.semantic_parser import parse_sysprompt_effect_async, PARSER_MODEL

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "configs" / "sysprompt_variation" / "system_prompts_task.yaml"

def load_system_prompts() -> dict[str, str | None]:
    with open(SYSTEM_PROMPTS_PATH) as f:
        return yaml.safe_load(f)

TASKS = [
    "Write a short poem about a river.",
    "Explain how photosynthesis works in a few sentences.",
]

# ============================================================================
# Test runner
# ============================================================================

def run_tests(completion_model: str, max_concurrent: int = 20) -> list[dict]:
    completion_client = get_client(completion_model)
    system_prompts = load_system_prompts()
    total = len(system_prompts) * len(TASKS)

    print(f"Testing {len(system_prompts)} prompts × {len(TASKS)} tasks = {total} completions on {completion_model}")

    # Build all test cases and requests
    test_cases = []
    requests = []
    for prompt_name, sysprompt in system_prompts.items():
        for task in TASKS:
            test_cases.append((prompt_name, sysprompt, task))
            messages = [{"role": "user", "content": task}]
            if sysprompt:
                messages = [{"role": "system", "content": sysprompt}] + messages
            requests.append(GenerateRequest(messages=messages, temperature=0.7))

    # Generate all completions in parallel
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Generating completions", total=len(requests))
        batch_results = completion_client.generate_batch(
            requests, max_concurrent,
            on_complete=lambda: progress.update(task, advance=1),
        )

    completions = [r.response if r.ok else f"[ERROR: {r.error}]" for r in batch_results]

    # Judge all completions in parallel
    async def judge_all() -> list[dict]:
        semaphore = asyncio.Semaphore(max_concurrent)
        done = [0]

        async def judge_one(idx: int) -> dict:
            prompt_name, sysprompt, task = test_cases[idx]
            response = completions[idx]

            async with semaphore:
                judgment = await parse_sysprompt_effect_async(sysprompt, task, response)

            done[0] += 1
            print(f"\r  Judging completions: {done[0]}/{total}", end="", flush=True)

            return {
                "prompt_name": prompt_name,
                "sysprompt": sysprompt,
                "task": task,
                "response": response,
                "response_len": len(response),
                "sysprompt_reference": judgment.sysprompt_reference,
                "sentiment": judgment.sentiment,
                "refusal": judgment.refusal,
            }

        return list(await asyncio.gather(*[judge_one(i) for i in range(len(test_cases))]))

    results = asyncio.run(judge_all())
    print()
    return results


def print_results(results: list[dict]) -> None:
    for r in results:
        print(f"\n{'='*80}")
        print(f"[{r['prompt_name']}] sysprompt: {r['sysprompt'] or '(none)'}")
        print(f"task: {r['task']}")
        print(f"{'-'*80}")
        print(r['response'])
        print(f"{'-'*80}")
        print(f"sysprompt_reference={r['sysprompt_reference']}  sentiment={r['sentiment']:.2f}  refusal={r['refusal']}")


def save_results(results: list[dict], model: str, output_path: Path | None) -> Path:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"sysprompt_effects_{model.replace('-', '_')}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": model,
        "judge_model": PARSER_MODEL,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test system prompt effects on completions")
    parser.add_argument("--model", default="gemma-2-27b", help="Completion model to test")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: auto-generated)")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent requests")
    args = parser.parse_args()

    results = run_tests(args.model, args.max_concurrent)
    print_results(results)
    save_results(results, args.model, args.output)


if __name__ == "__main__":
    main()
