from pathlib import Path

TEMPLATES_DATA_DIR = Path(__file__).parent / "data"

from .template import (
    TEMPLATE_TYPE_PLACEHOLDERS,
    PromptTemplate,
    load_templates_from_yaml,
    parse_template_dict,
)


def __getattr__(name: str):
    """Lazy import for heavy modules to keep generator fast."""
    # Builders (imports from measurement which has heavy deps)
    if name in (
        "BaseModelRevealedPromptBuilder",
        "PreTaskRevealedPromptBuilder",
        "PostTaskStatedPromptBuilder",
        "PostTaskRevealedPromptBuilder",
        "PreTaskStatedPromptBuilder",
        "PreTaskRankingPromptBuilder",
        "PostTaskRankingPromptBuilder",
        "MultiTurnRevealedPromptBuilder",
        "PromptBuilder",
    ):
        from . import builders
        return getattr(builders, name)

    # Generator
    if name in (
        "GeneratorConfig",
        "TemplateVariant",
        "generate_and_write",
        "generate_templates",
        "load_config_from_yaml",
        "write_templates_yaml",
    ):
        from . import generator
        return getattr(generator, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TEMPLATES_DATA_DIR",
    "TEMPLATE_TYPE_PLACEHOLDERS",
    "PromptTemplate",
    "load_templates_from_yaml",
    "parse_template_dict",
    # Builders (lazy)
    "BaseModelRevealedPromptBuilder",
    "PreTaskRevealedPromptBuilder",
    "PostTaskStatedPromptBuilder",
    "PostTaskRevealedPromptBuilder",
    "PreTaskStatedPromptBuilder",
    "PreTaskRankingPromptBuilder",
    "PostTaskRankingPromptBuilder",
    "MultiTurnRevealedPromptBuilder",
    "PromptBuilder",
    # Generator (lazy)
    "GeneratorConfig",
    "TemplateVariant",
    "generate_and_write",
    "generate_templates",
    "load_config_from_yaml",
    "write_templates_yaml",
]
