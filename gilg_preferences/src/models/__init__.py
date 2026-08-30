from .base import Model, ConfigurableMockModel
from .openai_compatible import OpenAICompatibleClient, VLLMClient, CerebrasClient, OpenRouterClient, ToolCallError, GenerateRequest, BatchResult
from .base import GenerationResult, LayerHook
from .registry import (
    MODEL_REGISTRY,
    ModelConfig,
    get_cerebras_name,
    get_openrouter_name,
    get_hf_name,
    is_valid_model,
    list_models,
    is_reasoning_model,
    adjust_max_tokens_for_reasoning,
)

try:
    from .huggingface_model import HuggingFaceModel
except ImportError:
    HuggingFaceModel = None  # type: ignore[assignment,misc]

try:
    from .hybrid_model import HybridActivationModel
except ImportError:
    HybridActivationModel = None  # type: ignore[assignment,misc]

BACKENDS: dict[str, type[OpenAICompatibleClient]] = {
    "openrouter": OpenRouterClient,
    "cerebras": CerebrasClient,
    "vllm": VLLMClient,
}


def get_client(
    model_name: str | None = None,
    max_new_tokens: int = 256,
    reasoning_effort: str | None = None,
    backend: str = "openrouter",
    openrouter_provider_sort: str | None = None,
    openrouter_provider_order: list[str] | None = None,
) -> OpenAICompatibleClient:
    if model_name is not None:
        max_new_tokens = adjust_max_tokens_for_reasoning(model_name, max_new_tokens)
    kwargs: dict = dict(model_name=model_name, max_new_tokens=max_new_tokens, reasoning_effort=reasoning_effort)
    if backend == "openrouter":
        kwargs["provider_sort"] = openrouter_provider_sort
        kwargs["provider_order"] = openrouter_provider_order
    return BACKENDS[backend](**kwargs)


__all__ = [
    "Model",
    "ConfigurableMockModel",
    "HuggingFaceModel",
    "HybridActivationModel",
    "GenerationResult",
    "LayerHook",
    "OpenAICompatibleClient",
    "VLLMClient",
    "CerebrasClient",
    "OpenRouterClient",
    "ToolCallError",
    "GenerateRequest",
    "BatchResult",
    "BACKENDS",
    "get_client",
    "MODEL_REGISTRY",
    "ModelConfig",
    "get_cerebras_name",
    "get_openrouter_name",
    "get_hf_name",
    "is_valid_model",
    "list_models",
]
