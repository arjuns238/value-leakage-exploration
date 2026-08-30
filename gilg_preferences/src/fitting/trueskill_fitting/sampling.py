from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.task_data import Task


def sample_ranking_groups(
    tasks: list[Task],
    n_tasks_per_group: int,
    n_groups: int,
    rng: np.random.Generator,
) -> list[list[Task]]:
    """Sample task groups with balanced coverage using weighted sampling."""
    if n_tasks_per_group > len(tasks):
        raise ValueError(
            f"n_tasks_per_group ({n_tasks_per_group}) > len(tasks) ({len(tasks)})"
        )

    total_slots = n_groups * n_tasks_per_group
    target_per_task = total_slots / len(tasks)
    task_counts: dict[str, int] = {t.id: 0 for t in tasks}
    groups: list[list[Task]] = []

    for _ in range(n_groups):
        weights = np.array([
            max(0.1, target_per_task - task_counts[t.id] + 1)
            for t in tasks
        ])
        weights /= weights.sum()

        indices = rng.choice(len(tasks), size=n_tasks_per_group, replace=False, p=weights)
        group = [tasks[i] for i in indices]
        groups.append(group)

        for t in group:
            task_counts[t.id] += 1

    return groups
