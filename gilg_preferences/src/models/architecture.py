"""Model architecture detection for HuggingFace models.

Maps model_type from HF config to layer accessor functions.
"""

from __future__ import annotations

from typing import Callable, Any

import torch.nn as nn


LayerAccessor = Callable[[Any], nn.ModuleList]


def _standard_layers(model: Any) -> nn.ModuleList:
    return model.model.layers


def _gemma3_layers(model: Any) -> nn.ModuleList:
    return model.model.language_model.layers


ARCHITECTURE_CONFIGS: dict[str, LayerAccessor] = {
    "llama": _standard_layers,
    "qwen2": _standard_layers,
    "qwen3": _standard_layers,
    "gemma": _standard_layers,
    "gemma2": _standard_layers,
    "gemma3": _gemma3_layers,
    "gemma3_text": _standard_layers,
    "gpt_oss": _standard_layers,
    "qwen3_5_moe": _gemma3_layers,
    "qwen3_5_moe_text": _standard_layers,
}


def get_layer_accessor(model_type: str) -> LayerAccessor:
    if model_type not in ARCHITECTURE_CONFIGS:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(ARCHITECTURE_CONFIGS.keys())}"
        )
    return ARCHITECTURE_CONFIGS[model_type]


def get_layers(model: Any) -> nn.ModuleList:
    model_type = model.config.model_type
    accessor = get_layer_accessor(model_type)
    return accessor(model)


def get_n_layers(model: Any) -> int:
    return len(get_layers(model))


def get_hidden_dim(model: Any) -> int:
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    return config.hidden_size


def get_v_proj_weight(model: Any, layer_idx: int) -> nn.Parameter:
    """Returns layer's W_v, shape (num_kv_heads * head_dim, hidden_size)."""
    return get_layers(model)[layer_idx].self_attn.v_proj.weight


def get_k_proj_weight(model: Any, layer_idx: int) -> nn.Parameter:
    """Returns layer's W_k, shape (num_kv_heads * head_dim, hidden_size)."""
    return get_layers(model)[layer_idx].self_attn.k_proj.weight


def get_num_kv_heads(model: Any) -> int:
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config.num_key_value_heads
    return config.num_key_value_heads


def get_head_dim(model: Any) -> int:
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config.head_dim
    return config.head_dim
