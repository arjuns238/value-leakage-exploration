"""Canonical model registry with backend-specific name mappings.

This module defines canonical model names used throughout the codebase and provides
mappings to backend-specific model names (HuggingFace, OpenRouter, etc.).

Usage:
    # In tests and application code, use canonical names:
    client = get_client("llama-3.1-8b")

    # The client resolves to the appropriate backend name automatically
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model across different backends."""

    canonical_name: str
    hf_name: str | None
    openrouter_name: str | None
    cerebras_name: str | None = None
    eot_token: str | None = None
    system_prompt: str | None = None
    reasoning_mode: Literal["none", "openrouter"] = "none"
    supports_system_role: bool = True


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama-3.2-1b": ModelConfig(
        canonical_name="llama-3.2-1b",
        hf_name="meta-llama/Llama-3.2-1B-Instruct",

        openrouter_name="meta-llama/llama-3.2-1b-instruct",
        eot_token="<|eot_id|>",
    ),
    "llama-3.1-8b": ModelConfig(
        canonical_name="llama-3.1-8b",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",

        cerebras_name="llama3.1-8b",
        openrouter_name="meta-llama/llama-3.1-8b-instruct",
        eot_token="<|eot_id|>",
    ),
    "llama-3.3-70b": ModelConfig(
        canonical_name="llama-3.3-70b",
        hf_name="meta-llama/Llama-3.3-70B-Instruct",

        openrouter_name="meta-llama/llama-3.3-70b-instruct",
        eot_token="<|eot_id|>",
    ),
    "qwen3-8b": ModelConfig(
        canonical_name="qwen3-8b",
        hf_name=None,

        openrouter_name="qwen/qwen3-8b",
    ),
    "qwen3-14b": ModelConfig(
        canonical_name="qwen3-14b",
        hf_name="Qwen/Qwen3-14B",

        openrouter_name="qwen/qwen3-14b",
        eot_token="<|im_end|>",
        reasoning_mode="openrouter",
    ),
    "qwen3-14b-nothink": ModelConfig(
        canonical_name="qwen3-14b-nothink",
        hf_name="Qwen/Qwen3-14B",

        openrouter_name="qwen/qwen3-14b",
        eot_token="<|im_end|>",
        system_prompt="/no_think",
        reasoning_mode="none",
    ),
    "qwen3-32b": ModelConfig(
        canonical_name="qwen3-32b",
        hf_name="Qwen/Qwen3-32B",

        openrouter_name="qwen/qwen3-32b",
        eot_token="<|im_end|>",
        reasoning_mode="openrouter",
    ),
    "qwen3-32b-nothink": ModelConfig(
        canonical_name="qwen3-32b-nothink",
        hf_name="Qwen/Qwen3-32B",

        openrouter_name="qwen/qwen3-32b",
        eot_token="<|im_end|>",
        system_prompt="/no_think",
        reasoning_mode="none",
    ),
    "qwen3.5-122b": ModelConfig(
        canonical_name="qwen3.5-122b",
        hf_name="Qwen/Qwen3.5-122B-A10B",

        openrouter_name="qwen/qwen3.5-122b-a10b",
        eot_token="<|im_end|>",
        reasoning_mode="openrouter",
    ),
    "qwen3.5-122b-nothink": ModelConfig(
        canonical_name="qwen3.5-122b-nothink",
        hf_name="Qwen/Qwen3.5-122B-A10B",

        openrouter_name="qwen/qwen3.5-122b-a10b",
        eot_token="<|im_end|>",
        system_prompt="/no_think",
        reasoning_mode="none",
    ),
    # Qwen-3.5-122B with the sadist SFT LoRA merged in. Served locally via
    # vLLM; hf_name is the path to the merged checkpoint on the serving host.
    "qwen3.5-122b-sadist-v3-545": ModelConfig(
        canonical_name="qwen3.5-122b-sadist-v3-545",
        hf_name="/path/to/sadist_merged",
        openrouter_name=None,
        eot_token="<|im_end|>",
        reasoning_mode="none",
    ),
    # Same merged checkpoint as above; the `-nothink` suffix triggers
    # `_default_enable_thinking=False` in HuggingFaceModel.format_messages,
    # used for activation extraction to match the canonical paper baseline.
    "qwen3.5-122b-sadist-v3-545-nothink": ModelConfig(
        canonical_name="qwen3.5-122b-sadist-v3-545-nothink",
        hf_name="/path/to/sadist_merged",
        openrouter_name=None,
        eot_token="<|im_end|>",
        reasoning_mode="none",
    ),
    "gemma-2-27b": ModelConfig(
        canonical_name="gemma-2-27b",
        hf_name="google/gemma-2-27b-it",

        openrouter_name="google/gemma-2-27b-it",
        eot_token="<end_of_turn>",
        supports_system_role=False,
    ),
    "gemma-3-1b": ModelConfig(
        canonical_name="gemma-3-1b",
        hf_name="google/gemma-3-1b-it",
        openrouter_name=None,
        eot_token="<end_of_turn>",
        supports_system_role=False,
    ),
    "gemma-3-1b-pt": ModelConfig(
        canonical_name="gemma-3-1b-pt",
        hf_name="google/gemma-3-1b-pt",
        openrouter_name=None,
        supports_system_role=False,
    ),
    "gemma-3-27b": ModelConfig(
        canonical_name="gemma-3-27b",
        hf_name="google/gemma-3-27b-it",

        openrouter_name="google/gemma-3-27b-it",
        eot_token="<end_of_turn>",
        supports_system_role=False,
    ),
    "gemma-3-27b-pt": ModelConfig(
        canonical_name="gemma-3-27b-pt",
        hf_name="google/gemma-3-27b-pt",

        openrouter_name=None,
        supports_system_role=False,
    ),
    "gpt-oss-120b": ModelConfig(
        canonical_name="gpt-oss-120b",
        hf_name="openai/gpt-oss-120b",

        openrouter_name="openai/gpt-oss-120b",
    ),
    "claude-haiku-4.5": ModelConfig(
        canonical_name="claude-haiku-4.5",
        hf_name=None,

        openrouter_name="anthropic/claude-haiku-4.5",
    ),
}



def get_cerebras_name(canonical_name: str) -> str:
    """Get Cerebras API model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.cerebras_name is None:
        raise ValueError(f"Model {canonical_name} not available for Cerebras")
    return config.cerebras_name


def get_openrouter_name(canonical_name: str) -> str:
    """Get OpenRouter API model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.openrouter_name is None:
        raise ValueError(f"Model {canonical_name} not available for OpenRouter")
    return config.openrouter_name


def get_hf_name(canonical_name: str) -> str:
    """Get HuggingFace model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.hf_name is not None:
        return config.hf_name
    raise ValueError(f"Model {canonical_name} not available for HuggingFace")


def get_model_system_prompt(canonical_name: str) -> str | None:
    return MODEL_REGISTRY[canonical_name].system_prompt


def supports_system_role(canonical_name: str) -> bool:
    """Check if model supports system role in chat messages."""
    return MODEL_REGISTRY[canonical_name].supports_system_role


def get_eot_token(canonical_name: str) -> str:
    config = MODEL_REGISTRY[canonical_name]
    if config.eot_token is None:
        raise ValueError(f"No end-of-turn token configured for {canonical_name}")
    return config.eot_token


def is_valid_model(canonical_name: str) -> bool:
    """Check if a canonical model name is registered."""
    return canonical_name in MODEL_REGISTRY


def has_hf_support(canonical_name: str) -> bool:
    if canonical_name not in MODEL_REGISTRY:
        return False
    return MODEL_REGISTRY[canonical_name].hf_name is not None


def list_models() -> list[str]:
    """List all available canonical model names."""
    return list(MODEL_REGISTRY.keys())


def should_capture_reasoning(canonical_name: str) -> bool:
    return MODEL_REGISTRY[canonical_name].reasoning_mode != "none"


# Substring fallback for *unregistered* model names (e.g. raw HF ids passed
# directly to clients). Registered models go via MODEL_REGISTRY[name].reasoning_mode.
REASONING_MODEL_PATTERNS = ["qwen3", "qwq", "deepseek-r1", "o1", "o3", "gpt-oss"]


def is_reasoning_model(model_name: str) -> bool:
    """True iff the model emits chain-of-thought into the output token budget.
    Registered models use their declared reasoning_mode; unregistered names
    fall back to substring patterns."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name].reasoning_mode != "none"
    name_lower = model_name.lower()
    return any(pattern in name_lower for pattern in REASONING_MODEL_PATTERNS)


def adjust_max_tokens_for_reasoning(model_name: str, max_tokens: int) -> int:
    """Adjust max_tokens for reasoning models (4x, minimum 1024)."""
    if is_reasoning_model(model_name):
        return max(1024, max_tokens * 4)
    return max_tokens


def adjust_timeout_for_reasoning(model_name: str, timeout: float) -> float:
    """Adjust timeout for reasoning models (10x)."""
    if is_reasoning_model(model_name):
        return timeout * 10
    return timeout
