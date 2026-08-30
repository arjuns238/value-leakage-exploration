"""KV cache manipulation utilities for isolated steering."""

from __future__ import annotations

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from src.models.architecture import get_v_proj_weight, get_k_proj_weight, get_num_kv_heads, get_head_dim


def _project_to_head_space(weight, model, direction):
    d_vec = torch.tensor(direction, dtype=weight.dtype, device=weight.device)
    projected = weight @ d_vec
    n_heads = get_num_kv_heads(model)
    d_head = get_head_dim(model)
    return projected.reshape(n_heads, d_head)


def project_to_v_space(model, layer_idx: int, direction: np.ndarray) -> torch.Tensor:
    """Project a residual-stream direction through W_v. Returns (num_kv_heads, head_dim)."""
    return _project_to_head_space(get_v_proj_weight(model, layer_idx), model, direction)


def project_to_kv_space(model, layer_idx: int, direction: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a residual-stream direction through W_k and W_v.

    Returns (k_dir, v_dir), each shape (num_kv_heads, head_dim).
    """
    k_dir = _project_to_head_space(get_k_proj_weight(model, layer_idx), model, direction)
    v_dir = _project_to_head_space(get_v_proj_weight(model, layer_idx), model, direction)
    return k_dir, v_dir


def modify_cache_v_at_positions(
    cache: DynamicCache,
    layer_idx: int,
    start: int,
    end: int,
    v_direction: torch.Tensor,
    coefficient: float,
) -> None:
    """Add coefficient * v_direction to V cache entries at [start, end). In-place."""
    layer = cache.layers[layer_idx]
    new_values = layer.values.clone()
    new_values[:, :, start:end, :] += coefficient * v_direction.unsqueeze(1)
    layer.values = new_values


def modify_cache_kv_at_positions(
    cache: DynamicCache,
    layer_idx: int,
    start: int,
    end: int,
    k_direction: torch.Tensor,
    v_direction: torch.Tensor,
    coefficient: float,
) -> None:
    """Add coefficient * direction to both K and V cache entries at [start, end). In-place."""
    layer = cache.layers[layer_idx]
    new_keys = layer.keys.clone()
    new_values = layer.values.clone()
    new_keys[:, :, start:end, :] += coefficient * k_direction.unsqueeze(1)
    new_values[:, :, start:end, :] += coefficient * v_direction.unsqueeze(1)
    layer.keys = new_keys
    layer.values = new_values


def combine_caches(
    base: DynamicCache,
    overlays: list[tuple[DynamicCache, int, int]],
) -> DynamicCache:
    """Clone base cache, then overwrite [start, end) from each overlay's source cache.

    overlays: list of (source_cache, start, end) — applied in order.
    """
    combined = DynamicCache()
    for layer_idx in range(len(base)):
        k = base.layers[layer_idx].keys.clone()
        v = base.layers[layer_idx].values.clone()
        for source, start, end in overlays:
            k[:, :, start:end, :] = source.layers[layer_idx].keys[:, :, start:end, :]
            v[:, :, start:end, :] = source.layers[layer_idx].values[:, :, start:end, :]
        combined.update(k, v, layer_idx)
    return combined
