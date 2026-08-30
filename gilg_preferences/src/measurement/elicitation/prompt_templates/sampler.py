from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np

from src.measurement.elicitation.prompt_templates.template import PromptTemplate


@dataclass(frozen=True)
class SampledConfiguration:
    template: PromptTemplate
    response_format: str
    seed: int
    order: str | None = None  # Only for revealed preferences (canonical/reversed)


def sample_configurations_lhs(
    templates: list[PromptTemplate],
    response_formats: list[str],
    generation_seeds: list[int],
    n_samples: int,
    orders: list[str] | None = None,
    seed: int | None = None,
) -> list[SampledConfiguration]:
    """Sample n configurations using Latin Hypercube Sampling."""
    rng = np.random.default_rng(seed)

    # Build full hypercube
    if orders:
        raw_configs = list(product(templates, response_formats, generation_seeds, orders))
        all_configs = [
            SampledConfiguration(t, rf, s, o) for t, rf, s, o in raw_configs
        ]
    else:
        raw_configs = list(product(templates, response_formats, generation_seeds))
        all_configs = [
            SampledConfiguration(t, rf, s) for t, rf, s in raw_configs
        ]

    if n_samples >= len(all_configs):
        return all_configs

    def get_dims(config: SampledConfiguration) -> dict[str, str]:
        dims = config.template.tags_dict.copy()
        dims["response_format"] = config.response_format
        dims["seed"] = str(config.seed)
        if config.order is not None:
            dims["order"] = config.order
        return dims

    config_dims = [get_dims(c) for c in all_configs]

    # Find all dimensions that actually vary
    all_dim_keys = set()
    for dims in config_dims:
        all_dim_keys.update(dims.keys())

    dim_values: dict[str, list[str]] = {}
    for key in all_dim_keys:
        values = list({dims[key] for dims in config_dims if key in dims})
        if len(values) > 1:
            dim_values[key] = values

    # Compute target counts per dimension value
    target_counts: dict[str, dict[str, int]] = {}
    for dim, values in dim_values.items():
        k = len(values)
        base = n_samples // k
        remainder = n_samples % k
        extras = rng.choice(values, size=remainder, replace=False).tolist()
        target_counts[dim] = {v: base + (1 if v in extras else 0) for v in values}

    # Greedy selection to match LHS targets
    selected_indices: list[int] = []
    current_counts: dict[str, Counter[str]] = {dim: Counter() for dim in dim_values}
    available = set(range(len(all_configs)))

    for _ in range(n_samples):
        best_idx = None
        best_score = float("-inf")

        for idx in available:
            dims = config_dims[idx]
            score = 0.0
            for dim in dim_values:
                if dim not in dims:
                    continue
                val = dims[dim]
                current = current_counts[dim][val]
                target = target_counts[dim][val]
                if current < target:
                    score += (target - current) / target
                else:
                    score -= (current - target + 1) * 10

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        available.remove(best_idx)

        dims = config_dims[best_idx]
        for dim in dim_values:
            if dim in dims:
                current_counts[dim][dims[dim]] += 1

    return [all_configs[i] for i in selected_indices]


def print_sampling_balance(configs: list[SampledConfiguration]) -> None:
    counts: dict[str, Counter[str]] = {}

    for config in configs:
        dims = config.template.tags_dict.copy()
        dims["response_format"] = config.response_format
        dims["seed"] = str(config.seed)
        if config.order is not None:
            dims["order"] = config.order

        for key, val in dims.items():
            if key not in counts:
                counts[key] = Counter()
            counts[key][val] += 1

    print(f"Sampling balance ({len(configs)} configurations):")
    for dim in sorted(counts.keys()):
        print(f"  {dim}: {dict(counts[dim])}")
