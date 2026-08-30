import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel

from .task import OriginDataset, Task


DATA_DIR = Path(__file__).parent / "data"

FILE_MAPPING = {
    OriginDataset.WILDCHAT: [
        "wildchat_en_8k.jsonl",
    ],
    OriginDataset.ALPACA: ["alpaca_tasks_nemocurator.jsonl"],
    OriginDataset.MATH: ["math.jsonl"],
    OriginDataset.BAILBENCH: ["bailBench.csv"],
    OriginDataset.STRESS_TEST: ["stress_testing_model_spec.jsonl"],
    OriginDataset.CREAK: ["creak.jsonl"],
}


class ParserConfig(BaseModel):
    origin: OriginDataset
    prompt_key: str
    id_key: str | None
    metadata_keys: list[str]
    metadata_defaults: dict | None = None
    id_template: str | None = None

    def parse(self, row: dict, index: int) -> Task:
        metadata = {}
        defaults = self.metadata_defaults or {}
        for key in self.metadata_keys:
            if key in defaults:
                metadata[key] = row.get(key, defaults[key])
            else:
                metadata[key] = row[key]
        if self.id_template:
            task_id = self.id_template.format(**row)
        elif self.id_key:
            task_id = row[self.id_key]
        else:
            task_id = f"{self.origin.name.lower()}_{index}"
        return Task(
            prompt=row[self.prompt_key],
            origin=self.origin,
            id=task_id,
            metadata=metadata,
        )


PARSER_CONFIGS = {
    OriginDataset.WILDCHAT: ParserConfig(
        origin=OriginDataset.WILDCHAT,
        prompt_key="text",
        id_key="id",
        metadata_keys=["type", "topic"],
    ),
    OriginDataset.ALPACA: ParserConfig(
        origin=OriginDataset.ALPACA,
        prompt_key="task_text",
        id_key="task_id",
        metadata_keys=["nemo_analysis"],
        metadata_defaults={"nemo_analysis": {}},
    ),
    OriginDataset.MATH: ParserConfig(
        origin=OriginDataset.MATH,
        prompt_key="text",
        id_key="id",
        metadata_keys=["type", "topic", "q_metadata"],
        metadata_defaults={"q_metadata": {}},
    ),
    OriginDataset.BAILBENCH: ParserConfig(
        origin=OriginDataset.BAILBENCH,
        prompt_key="content",
        id_key=None,
        metadata_keys=["subcategory", "category"],
    ),
    OriginDataset.STRESS_TEST: ParserConfig(
        origin=OriginDataset.STRESS_TEST,
        prompt_key="query",
        id_key=None,
        metadata_keys=["value1", "value2", "nudge_direction", "query_generator"],
        id_template="stresstest_{chunk_index}_{entry_idx}_{nudge_direction}",
    ),
    OriginDataset.CREAK: ParserConfig(
        origin=OriginDataset.CREAK,
        prompt_key="sentence",
        id_key="ex_id",
        metadata_keys=["label", "entity"],
    ),
}


def _load_jsonl(filepath: Path) -> list[dict]:
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def _load_csv(filepath: Path) -> list[dict]:
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_origin(origin: OriginDataset) -> list[Task]:
    tasks = []
    config = PARSER_CONFIGS[origin]
    for filename in FILE_MAPPING[origin]:
        filepath = DATA_DIR / filename
        if filepath.exists():
            if filepath.suffix == ".csv":
                rows = _load_csv(filepath)
            else:
                rows = _load_jsonl(filepath)
            tasks.extend(config.parse(row, i) for i, row in enumerate(rows))
    return tasks


def load_tasks(
    n: int,
    origins: list[OriginDataset],
    seed: int | None = None,
    filter_fn: Callable[[Task], bool] | None = None,
    stratified: bool = False,
) -> list[Task]:
    tasks_by_origin: dict[OriginDataset, list[Task]] = {}
    for origin in origins:
        tasks_by_origin[origin] = _load_origin(origin)

    if filter_fn is not None:
        for origin in tasks_by_origin:
            tasks_by_origin[origin] = [t for t in tasks_by_origin[origin] if filter_fn(t)]

    rng = np.random.default_rng(seed) if seed is not None else None

    if stratified:
        if rng is not None:
            for origin_tasks in tasks_by_origin.values():
                rng.shuffle(origin_tasks)

        # Take all from origins smaller than equal share, redistribute to others
        remaining_n = n
        quotas: dict[OriginDataset, int] = {}
        unassigned = set(origins)
        while unassigned:
            share = remaining_n / len(unassigned)
            small = {o for o in unassigned if len(tasks_by_origin[o]) <= share}
            if not small:
                break
            for o in small:
                quotas[o] = len(tasks_by_origin[o])
                remaining_n -= quotas[o]
            unassigned -= small

        per = remaining_n // len(unassigned) if unassigned else 0
        extra = remaining_n % len(unassigned) if unassigned else 0
        for i, o in enumerate(sorted(unassigned, key=lambda o: o.value)):
            quotas[o] = per + (1 if i < extra else 0)

        result = []
        for origin in origins:
            result.extend(tasks_by_origin[origin][:quotas[origin]])
        if rng is not None:
            rng.shuffle(result)
        return result

    tasks = [t for origin_tasks in tasks_by_origin.values() for t in origin_tasks]
    if rng is not None:
        rng.shuffle(tasks)
    return tasks[:n]


def load_filtered_tasks(
    n: int,
    origins: list[OriginDataset],
    seed: int | None = None,
    consistency_model: str | None = None,
    consistency_keep_ratio: float = 0.7,
    task_ids: set[str] | None = None,
    exclude_task_ids: set[str] | None = None,
    filter_fn: Callable[[Task], bool] | None = None,
    stratified: bool = False,
) -> list[Task]:
    filters: list[Callable[[Task], bool]] = []

    if consistency_model is not None:
        from .consistency import make_consistency_filter  # circular import
        filters.append(make_consistency_filter(consistency_model, keep_ratio=consistency_keep_ratio))

    if task_ids is not None:
        filters.append(lambda t, ids=task_ids: t.id in ids)

    if exclude_task_ids is not None:
        filters.append(lambda t, ids=exclude_task_ids: t.id not in ids)

    if filter_fn is not None:
        filters.append(filter_fn)

    combined_filter = None
    if filters:
        combined_filter = lambda t: all(f(t) for f in filters)

    return load_tasks(n=n, origins=origins, seed=seed, filter_fn=combined_filter, stratified=stratified)


def load_completions(path: Path) -> list[tuple[Task, str]]:
    """Load task-completion pairs from JSON.

    Expected format: [{"task_id": str, "task_prompt": str, "completion": str, "origin": str}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    return [
        (
            Task(
                prompt=item["task_prompt"],
                origin=OriginDataset[item.get("origin", "SYNTHETIC")],
                id=item["task_id"],
                metadata={},
            ),
            item["completion"],
        )
        for item in data
    ]
