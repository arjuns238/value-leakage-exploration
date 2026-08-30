from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class HooMethod:
    name: str
    train: Callable[[int, float | None], tuple[np.ndarray, float | None]]
    evaluate: Callable[[int, np.ndarray], dict]
    best_hp: float | None = None
