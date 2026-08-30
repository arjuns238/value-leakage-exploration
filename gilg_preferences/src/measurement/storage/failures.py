"""Storage for measurement failures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from src.types import FailureCategory, MeasurementFailure


def failure_to_dict(failure: MeasurementFailure) -> dict:
    """Convert a MeasurementFailure to a serializable dict."""
    return {
        "task_ids": failure.task_ids,
        "category": failure.category.value,
        "raw_response": failure.raw_response,
        "error_message": failure.error_message,
    }


def dict_to_failure(d: dict) -> MeasurementFailure:
    """Convert a dict back to a MeasurementFailure."""
    return MeasurementFailure(
        task_ids=d["task_ids"],
        category=FailureCategory(d["category"]),
        raw_response=d.get("raw_response"),
        error_message=d["error_message"],
    )


class FailureLog:
    """Append-only log for measurement failures."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._failures: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.log_path.exists():
            with open(self.log_path) as f:
                data = yaml.safe_load(f) or {}
            self._failures = data.get("failures", [])

    def append(self, failures: list[MeasurementFailure], run_info: dict | None = None) -> None:
        """Append failures from a measurement run."""
        if not failures:
            return

        timestamp = datetime.now().isoformat()
        for failure in failures:
            entry = failure_to_dict(failure)
            entry["timestamp"] = timestamp
            if run_info:
                entry["run_info"] = run_info
            self._failures.append(entry)

        self._save()

    def _save(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            yaml.dump({"failures": self._failures}, f, default_flow_style=False, allow_unicode=True)

    def get_failures(self, category: FailureCategory | None = None) -> list[MeasurementFailure]:
        """Get all failures, optionally filtered by category."""
        failures = [dict_to_failure(d) for d in self._failures]
        if category:
            failures = [f for f in failures if f.category == category]
        return failures

    def summary(self) -> dict[str, int]:
        """Get count of failures by category."""
        counts: dict[str, int] = {}
        for entry in self._failures:
            cat = entry["category"]
            counts[cat] = counts.get(cat, 0) + 1
        return counts


def save_run_failures(
    failures: list[MeasurementFailure],
    experiment_dir: Path,
    mode: str,
    run_info: dict | None = None,
) -> None:
    """Save failures from a run to the experiment directory."""
    if not failures:
        return
    log_path = experiment_dir / "failures" / f"{mode}.yaml"
    log = FailureLog(log_path)
    log.append(failures, run_info)
