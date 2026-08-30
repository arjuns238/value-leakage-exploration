from .task import OriginDataset, Task
from .loader import load_tasks, load_filtered_tasks, load_completions
from .paraphrase import paraphrase_tasks

ORIGIN_NAME_MAP = {
    "wildchat": OriginDataset.WILDCHAT,
    "alpaca": OriginDataset.ALPACA,
    "math": OriginDataset.MATH,
    "bailbench": OriginDataset.BAILBENCH,
    "stress_test": OriginDataset.STRESS_TEST,
}


def parse_origins(names: list[str]) -> list[OriginDataset]:
    return [ORIGIN_NAME_MAP[name] for name in names]


# Re-export consistency for backwards compatibility
from .consistency import (
    ConsistencyIndex,
    compute_consistency,
    load_consistency_index,
    make_consistency_filter,
)

__all__ = [
    "OriginDataset",
    "Task",
    "load_tasks",
    "load_filtered_tasks",
    "load_completions",
    "parse_origins",
    "ORIGIN_NAME_MAP",
    "ConsistencyIndex",
    "compute_consistency",
    "load_consistency_index",
    "make_consistency_filter",
    "paraphrase_tasks",
]
