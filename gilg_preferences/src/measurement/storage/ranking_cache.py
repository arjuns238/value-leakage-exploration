from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from src.measurement.storage.base import load_yaml, model_short_name, save_yaml
from src.types import RankingMeasurement, PreferenceType

if TYPE_CHECKING:
    from src.task_data import Task


def _task_group_hash(task_ids: list[str]) -> str:
    sorted_ids = sorted(task_ids)
    return hashlib.sha256("__".join(sorted_ids).encode()).hexdigest()[:12]


class RankingCache:
    """Storage: results/cache/ranking/{model_short}.yaml"""

    CACHE_DIR = Path("results/cache/ranking")

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_short = model_short_name(model_name)
        self._cache_path = self.CACHE_DIR / f"{self.model_short}.yaml"
        self._data: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._cache_path.exists():
            return load_yaml(self._cache_path)
        return {}

    def _make_key(
        self,
        template_name: str,
        response_format: str,
        seed: int,
        task_ids: list[str],
    ) -> str:
        task_hash = _task_group_hash(task_ids)
        parts = [template_name, response_format, str(seed), task_hash]
        return hashlib.sha256("__".join(parts).encode()).hexdigest()[:16]

    def get_measured_groups(
        self,
        template_name: str,
        response_format: str,
        seed: int,
    ) -> set[frozenset[str]]:
        if self._data is None:
            self._data = self._load()

        groups: set[frozenset[str]] = set()
        for entry in self._data.values():
            if (
                entry["template_name"] == template_name
                and entry["response_format"] == response_format
                and entry["seed"] == seed
            ):
                groups.add(frozenset(entry["task_ids"]))

        return groups

    def add(
        self,
        measurements: list[RankingMeasurement],
        template_name: str,
        response_format: str,
        seed: int,
    ) -> None:
        if not measurements:
            return

        if self._data is None:
            self._data = self._load()

        for m in measurements:
            task_ids = [t.id for t in m.tasks]
            h = self._make_key(template_name, response_format, seed, task_ids)

            if h not in self._data:
                self._data[h] = {
                    "template_name": template_name,
                    "response_format": response_format,
                    "seed": seed,
                    "task_ids": task_ids,
                    "preference_type": m.preference_type.value,
                    "samples": [],
                }

            self._data[h]["samples"].append({"ranking": m.ranking})

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(self._data, self._cache_path)

    def get_all_measurements(
        self,
        template_name: str,
        response_format: str,
        seed: int,
        task_lookup: dict[str, "Task"],
    ) -> list[RankingMeasurement]:
        if self._data is None:
            self._data = self._load()

        results: list[RankingMeasurement] = []
        for entry in self._data.values():
            if (
                entry["template_name"] == template_name
                and entry["response_format"] == response_format
                and entry["seed"] == seed
            ):
                task_ids = entry["task_ids"]
                tasks = [task_lookup[tid] for tid in task_ids]
                preference_type = PreferenceType(
                    entry.get("preference_type", PreferenceType.PRE_TASK_RANKING.value)
                )

                for sample in entry["samples"]:
                    results.append(RankingMeasurement(
                        tasks=tasks,
                        ranking=sample["ranking"],
                        preference_type=preference_type,
                    ))

        return results
