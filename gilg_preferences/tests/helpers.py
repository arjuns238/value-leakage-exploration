"""Shared test helper functions."""

import numpy as np

from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, TaskScore, PreferenceType


def make_task(id: str, origin: OriginDataset = OriginDataset.WILDCHAT) -> Task:
    return Task(prompt=f"Task {id}", origin=origin, id=id, metadata={})


def make_tasks(n: int, origin: OriginDataset = OriginDataset.MATH) -> list[Task]:
    return [
        Task(prompt=f"Task {i}", origin=origin, id=f"task_{i}", metadata={})
        for i in range(n)
    ]


def make_comparison(a: Task, b: Task, winner: str) -> BinaryPreferenceMeasurement:
    choice = "a" if winner == a.id else "b"
    return BinaryPreferenceMeasurement(
        task_a=a, task_b=b, choice=choice, preference_type=PreferenceType.PRE_TASK_STATED
    )


def make_measurement(task_a: Task, task_b: Task, choice: str) -> BinaryPreferenceMeasurement:
    return BinaryPreferenceMeasurement(
        task_a=task_a,
        task_b=task_b,
        choice=choice,
        preference_type=PreferenceType.PRE_TASK_STATED,
    )


def make_score(task: Task, score: float) -> TaskScore:
    return TaskScore(task=task, score=score, preference_type=PreferenceType.PRE_TASK_STATED)


def make_random_wins(n: int, rng: np.random.Generator) -> np.ndarray:
    wins = rng.integers(0, 10, size=(n, n))
    np.fill_diagonal(wins, 0)
    return wins
