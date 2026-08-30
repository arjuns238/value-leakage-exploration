from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExtractionStats:
    n_new: int = 0
    n_failures: int = 0
    n_truncated: int = 0
    n_ooms: int = 0


@dataclass
class ExtractionMetadata:
    model: str
    n_tasks: int
    task_origins: list[str]
    layers_config: list[float | int]
    layers_resolved: list[int]
    n_model_layers: int
    selectors: list[str]
    batch_size: int
    temperature: float
    max_new_tokens: int
    seed: int | None
    n_existing: int
    n_new: int
    n_failures: int
    n_truncated: int
    n_ooms: int
    source_completions: str | None = None
    system_prompt: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_updated"] = datetime.now().isoformat()
        return d
