from enum import Enum

from pydantic import BaseModel, Field

from src.types import PreferenceType


class PairingStrategy(str, Enum):
    ALL_PAIRS = "all_pairs"  # n(n-1)/2 unique pairs
    RANDOM_PAIRS = "random_pairs"  # Random sample of pairs


class DatasetMeasurementConfig(BaseModel, frozen=True):
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for model generation",
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        description="Number of times to repeat each measurement",
    )
    measurement_types: frozenset[PreferenceType] = Field(
        default_factory=lambda: frozenset(PreferenceType),
        description="Which preference types to measure",
    )
    pairing_strategy: PairingStrategy = Field(
        default=PairingStrategy.ALL_PAIRS,
        description="How to generate task pairs for binary comparisons",
    )
    max_pairs: int | None = Field(
        default=None,
        description="Maximum number of pairs when using RANDOM_PAIRS strategy",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
