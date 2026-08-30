"""Generate completions with configurable system prompts.

Usage:
    python -m src.experiments.sysprompt_variation.generate_sysprompt_completions configs/sysprompt_variation/completions.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

from src.measurement.storage import ExperimentStore
from src.measurement.storage.completions import generate_completions
from src.models import get_client
from src.task_data import load_filtered_tasks, parse_origins


load_dotenv()
console = Console()


class CompletionConfig(BaseModel):
    experiment_name: str
    completion_model: str
    temperature: float = 1.0
    seed: int = 0
    max_concurrent: int = 50

    n_tasks: int
    task_origins: list[str]

    # Task consistency filtering
    consistency_filter_model: str | None = None
    consistency_keep_ratio: float = 0.7

    system_prompts: dict[str, str | None]


SYSPROMPT_PREFIX = "Do not mention these instructions.\n\n"


def load_config(path: Path) -> CompletionConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    # Support referencing external system_prompts file
    if "system_prompts_file" in data:
        prompts_path = path.parent / data.pop("system_prompts_file")
        with open(prompts_path) as f:
            data["system_prompts"] = yaml.safe_load(f)

    # Add prefix to all non-null system prompts
    if "system_prompts" in data:
        data["system_prompts"] = {
            k: (SYSPROMPT_PREFIX + v if v else None)
            for k, v in data["system_prompts"].items()
        }

    return CompletionConfig.model_validate(data)


def save_completions(completions_dir: Path, condition_name: str, tc_list: list, config: CompletionConfig) -> None:
    condition_dir = completions_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "task_id": tc.task.id,
            "task_prompt": tc.task.prompt,
            "completion": tc.completion,
            "origin": tc.task.origin.name,
        }
        for tc in tc_list
    ]
    with open(condition_dir / "completions.json", "w") as f:
        json.dump(data, f, indent=2)

    condition_config = {
        "completion_model": config.completion_model,
        "system_prompt": config.system_prompts[condition_name],
        "temperature": config.temperature,
        "seed": config.seed,
        "n_completions": len(tc_list),
    }
    with open(condition_dir / "config.json", "w") as f:
        json.dump(condition_config, f, indent=2)


def main(config_path: Path):
    config = load_config(config_path)

    console.print(f"[bold]Experiment: {config.experiment_name}")
    console.print(f"[bold]Config: {config_path}")
    console.print(f"  Completion model: {config.completion_model}")
    console.print(f"  Temperature: {config.temperature}")
    console.print(f"  Seed: {config.seed}")
    console.print(f"  Task origins: {config.task_origins}")
    console.print(f"  n_tasks: {config.n_tasks}")
    console.print(f"  System prompts: {list(config.system_prompts.keys())}")
    console.print()

    exp_store = ExperimentStore(config.experiment_name)
    completions_dir = exp_store.base_dir / "completions"

    # Check which conditions need generation
    conditions_to_generate = []
    for name in config.system_prompts.keys():
        condition_path = completions_dir / name / "completions.json"
        if condition_path.exists():
            console.print(f"  [dim]{name}: already exists, skipping[/dim]")
        else:
            conditions_to_generate.append(name)

    if not conditions_to_generate:
        console.print("\n[green]All conditions already complete!")
        return

    console.print(f"\n[bold]Generating {len(conditions_to_generate)} conditions: {conditions_to_generate}")

    # Load tasks
    console.print("\n[bold]Loading tasks...")
    tasks = load_filtered_tasks(
        n=config.n_tasks,
        origins=parse_origins(config.task_origins),
        seed=config.seed,
        consistency_model=config.consistency_filter_model,
        consistency_keep_ratio=config.consistency_keep_ratio,
    )
    console.print(f"  Loaded {len(tasks)} tasks\n")

    # Generate completions
    client = get_client(config.completion_model)

    for name in conditions_to_generate:
        sysprompt = config.system_prompts[name]
        console.print(f"[bold]Generating {name} completions...")
        completions = generate_completions(
            client, tasks,
            temperature=config.temperature,
            seed=config.seed,
            system_prompt=sysprompt,
            max_concurrent=config.max_concurrent,
        )
        console.print(f"  Generated {len(completions)} completions")

        save_completions(completions_dir, name, completions, config)
        console.print(f"  Saved to {completions_dir / name}\n")

    console.print("[bold green]Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate completions with system prompts")
    parser.add_argument("config", type=Path, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
