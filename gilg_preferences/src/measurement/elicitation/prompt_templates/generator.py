from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Any

import yaml

# Use relative imports to avoid triggering src package __init__.py chain
from .generator_config import (
    STATED_TASK_LABELS,
    TASK_LABELS,
    GeneratorConfig,
    load_config_from_yaml,
)

from src.models import GenerateRequest, get_client

if TYPE_CHECKING:
    from src.models import OpenAICompatibleClient

# Type alias to avoid importing from src.types
Message = dict[str, Any]


def format_qualitative_options(values: list[str]) -> str:
    """Format qualitative values as 'good, neutral, or bad' style string."""
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} or {values[1]}"
    return f"{', '.join(values[:-1])}, or {values[-1]}"


class TemplateVariant(TypedDict):
    template: str
    phrasing: int
    language: str
    situating_context: str  # "none" or context key
    instruction_position: str
    task_label_names: str | None  # None for stated templates
    instruction_xml_tags: bool
    typos: bool
    punctuation: str
    scale: str | None  # e.g. "1-5", None for revealed templates
    response_format: str  # "regex", "xml", "tool_use"


def build_revealed_template(
    instruction: str,
    instruction_position: str,
    task_label_names: str,
    language: str,
    instruction_xml_tags: bool,
) -> str:
    label_a, label_b = TASK_LABELS[(task_label_names, language)]
    tasks_block = f"{label_a}:\n{{task_a}}\n{label_b}:\n{{task_b}}"

    if instruction_xml_tags:
        instructions_block = f"<instructions>\n{instruction}\n{{format_instruction}}\n</instructions>"
    else:
        instructions_block = f"{instruction}\n{{format_instruction}}"

    if instruction_position == "before":
        return f"{instructions_block}\n\n{tasks_block}"
    else:  # after
        return f"{tasks_block}\n\n{instructions_block}"


def build_stated_template(
    instruction: str,
    instruction_position: str,
    language: str,
    instruction_xml_tags: bool,
) -> str:
    task_label = STATED_TASK_LABELS[language]
    if instruction_xml_tags:
        task_block = f"<task>\n{task_label}\n{{task}}\n</task>"
        instructions_block = f"<instructions>\n{instruction}\n{{format_instruction}}\n</instructions>"
    else:
        task_block = f"{task_label}\n{{task}}"
        instructions_block = f"{instruction}\n{{format_instruction}}"

    if instruction_position == "before":
        return f"{instructions_block}\n\n{task_block}"
    else:  # after
        return f"{task_block}\n\n{instructions_block}"


def add_situating_context(template: str, context: str | None) -> str:
    if context is None:
        return template
    return f"{context}\n\n{template}"


def build_translation_prompt(text: str, language: str) -> list[Message]:
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text to {language}. "
                "Output ONLY the translation, nothing else.\n\n"
                f"{text}"
            ),
        }
    ]


def _build_instructions(config: GeneratorConfig) -> dict[tuple[int, str], str]:
    instructions: dict[tuple[int, str], str] = {}
    for phrasing_idx, instruction in enumerate(config.base_templates, start=1):
        instructions[(phrasing_idx, "en")] = instruction.strip()
    return instructions


def _translate_instructions(
    instructions: dict[tuple[int, str], str],
    config: GeneratorConfig,
    client: OpenAICompatibleClient,
    max_concurrent: int,
) -> dict[tuple[int, str], str]:
    non_english_languages = [lang for lang in config.languages if lang != "en"]

    if not non_english_languages:
        return instructions

    translation_requests: list[tuple[int, str, GenerateRequest]] = []

    for phrasing_idx, instruction in enumerate(config.base_templates, start=1):
        for lang in non_english_languages:
            messages = build_translation_prompt(instruction.strip(), lang)
            request = GenerateRequest(messages=messages, temperature=0.3)
            translation_requests.append((phrasing_idx, lang, request))

    requests = [req for _, _, req in translation_requests]
    results = client.generate_batch(requests, max_concurrent=max_concurrent)

    for (phrasing_idx, lang, _), result in zip(translation_requests, results):
        if result.ok:
            instructions[(phrasing_idx, lang)] = result.unwrap().strip()

    return instructions


def _build_transform_prompt(instruction: str, typos: bool, punctuation: str) -> list[Message]:
    parts = []
    if typos:
        parts.append("add 2-3 realistic typos (swapped letters, missing letters, etc)")
    if punctuation == "minimal":
        parts.append("remove the trailing punctuation")

    task = " and ".join(parts)
    return [{"role": "user", "content": f"Take this text and {task}. Output ONLY the modified text, nothing else.\n\nText: {instruction}"}]


def _apply_transformations(
    instructions: dict[tuple[int, str], str],
    config: GeneratorConfig,
    client: OpenAICompatibleClient,
    max_concurrent: int,
) -> dict[tuple[int, str, bool, str], str]:
    transformed: dict[tuple[int, str, bool, str], str] = {}
    requests_meta: list[tuple[int, str, bool, str]] = []
    requests: list[GenerateRequest] = []

    for (phrasing_idx, lang), instruction in instructions.items():
        for typos in config.typos:
            for punct in config.punctuation:
                if not typos and punct == "standard":
                    transformed[(phrasing_idx, lang, typos, punct)] = instruction
                else:
                    messages = _build_transform_prompt(instruction, typos, punct)
                    requests.append(GenerateRequest(messages=messages, temperature=0.7))
                    requests_meta.append((phrasing_idx, lang, typos, punct))

    if requests:
        results = client.generate_batch(requests, max_concurrent=max_concurrent)
        for meta, result in zip(requests_meta, results):
            if result.ok:
                transformed[meta] = result.unwrap().strip()

    return transformed


def _build_variants(
    instructions: dict[tuple[int, str, bool, str], str],
    config: GeneratorConfig,
) -> list[TemplateVariant]:
    variants: list[TemplateVariant] = []
    context_items = [("none", None), *config.situating_contexts.items()]
    phrasing_indices = range(1, len(config.base_templates) + 1)

    for lang, phrasing_idx, instruction_pos, use_typos, punct in product(
        config.languages, phrasing_indices, config.instruction_positions, config.typos, config.punctuation
    ):
        instruction = instructions.get((phrasing_idx, lang, use_typos, punct))
        if instruction is None:
            continue  # translation or typo generation failed

        if config.template_type == "pre_task_stated":
            _add_pre_task_stated_variants(
                variants, instruction, instruction_pos, lang, phrasing_idx, use_typos, punct, context_items, config
            )
        elif config.template_type == "post_task_stated":
            _add_post_task_stated_variants(
                variants, instruction, lang, phrasing_idx, use_typos, punct, context_items, config
            )
        elif config.template_type == "post_task_revealed":
            _add_post_task_revealed_variants(
                variants, instruction, lang, phrasing_idx, use_typos, punct, context_items, config
            )
        else:  # "pre_task_revealed"
            _add_revealed_variants(
                variants, instruction, instruction_pos, lang, phrasing_idx, use_typos, punct, context_items, config
            )

    return variants


def _build_scale_variants(instruction: str, config: GeneratorConfig) -> list[tuple[str, str]]:
    """Build (scale_tag, substituted_instruction) pairs for numeric or qualitative scales."""
    if config.scales:
        return [
            (f"{s[0]}-{s[1]}", instruction.replace("{scale_min}", str(s[0])).replace("{scale_max}", str(s[1])))
            for s in config.scales
        ]
    elif config.qualitative_values:
        variants = []
        for values in config.qualitative_values:
            options_str = format_qualitative_options(values)
            scale_tag = "binary" if len(values) == 2 else "ternary"
            variants.append((scale_tag, instruction.replace("{qualitative_options}", options_str)))
        return variants
    else:
        return [("qualitative", instruction)]


def _add_pre_task_stated_variants(
    variants: list[TemplateVariant],
    instruction: str,
    instruction_pos: str,
    lang: str,
    phrasing_idx: int,
    typos: bool,
    punctuation: str,
    context_items: list[tuple[str, str | None]],
    config: GeneratorConfig,
) -> None:
    scale_variants = _build_scale_variants(instruction, config)

    for use_xml in config.instruction_xml_tags:
        for scale_tag, scaled_instruction in scale_variants:
            template = build_stated_template(scaled_instruction, instruction_pos, lang, use_xml)

            for context_key, context_text in context_items:
                final_template = add_situating_context(template, context_text)
                for response_format in config.response_formats:
                    variants.append({
                        "template": final_template,
                        "phrasing": phrasing_idx,
                        "language": lang,
                        "situating_context": context_key,
                        "instruction_position": instruction_pos,
                        "task_label_names": None,
                        "instruction_xml_tags": use_xml,
                        "typos": typos,
                        "punctuation": punctuation,
                        "scale": scale_tag,
                        "response_format": response_format,
                    })


def _add_post_task_stated_variants(
    variants: list[TemplateVariant],
    instruction: str,
    lang: str,
    phrasing_idx: int,
    typos: bool,
    punctuation: str,
    context_items: list[tuple[str, str | None]],
    config: GeneratorConfig,
) -> None:
    """Post-task stated templates only have format_instruction placeholder (no {task})."""
    scale_variants = _build_scale_variants(instruction, config)

    for use_xml in config.instruction_xml_tags:
        for scale_tag, scaled_instruction in scale_variants:
            if use_xml:
                template = f"<instructions>\n{scaled_instruction}\n{{format_instruction}}\n</instructions>"
            else:
                template = f"{scaled_instruction}\n{{format_instruction}}"

            for context_key, context_text in context_items:
                final_template = add_situating_context(template, context_text)
                for response_format in config.response_formats:
                    variants.append({
                        "template": final_template,
                        "phrasing": phrasing_idx,
                        "language": lang,
                        "situating_context": context_key,
                        "instruction_position": "after",  # post-task is always after
                        "task_label_names": None,
                        "instruction_xml_tags": use_xml,
                        "typos": typos,
                        "punctuation": punctuation,
                        "scale": scale_tag,
                        "response_format": response_format,
                    })


def _add_post_task_revealed_variants(
    variants: list[TemplateVariant],
    instruction: str,
    lang: str,
    phrasing_idx: int,
    typos: bool,
    punctuation: str,
    context_items: list[tuple[str, str | None]],
    config: GeneratorConfig,
) -> None:
    """Post-task revealed templates only have format_instruction placeholder."""
    for use_xml in config.instruction_xml_tags:
        # Simple template: instruction + format_instruction
        if use_xml:
            template = f"<instructions>\n{instruction}\n{{format_instruction}}\n</instructions>"
        else:
            template = f"{instruction}\n{{format_instruction}}"

        for context_key, context_text in context_items:
            final_template = add_situating_context(template, context_text)
            for response_format in config.response_formats:
                variants.append({
                    "template": final_template,
                    "phrasing": phrasing_idx,
                    "language": lang,
                    "situating_context": context_key,
                    "instruction_position": "before",  # not applicable but needed for schema
                    "task_label_names": None,
                    "instruction_xml_tags": use_xml,
                    "typos": typos,
                    "punctuation": punctuation,
                    "scale": None,
                    "response_format": response_format,
                })


def _add_revealed_variants(
    variants: list[TemplateVariant],
    instruction: str,
    instruction_pos: str,
    lang: str,
    phrasing_idx: int,
    typos: bool,
    punctuation: str,
    context_items: list[tuple[str, str | None]],
    config: GeneratorConfig,
) -> None:
    for use_xml in config.instruction_xml_tags:
        for label_style in config.task_label_names:
            template = build_revealed_template(
                instruction, instruction_pos, label_style, lang, use_xml
            )

            for context_key, context_text in context_items:
                final_template = add_situating_context(template, context_text)
                for response_format in config.response_formats:
                    variants.append({
                        "template": final_template,
                        "phrasing": phrasing_idx,
                        "language": lang,
                        "situating_context": context_key,
                        "instruction_position": instruction_pos,
                        "task_label_names": label_style,
                        "instruction_xml_tags": use_xml,
                        "typos": typos,
                        "punctuation": punctuation,
                        "scale": None,
                        "response_format": response_format,
                    })


def _to_output_format(
    variants: list[TemplateVariant],
    config: GeneratorConfig,
) -> list[dict]:
    output = []
    is_pre_task = config.is_pre_task

    for idx, variant in enumerate(variants, start=1):
        template_id = f"{idx:03d}"
        name = f"{config.name_prefix}_{template_id}"

        tags = [f"phrasing:{variant['phrasing']}"]

        # Only include non-default values
        if variant["language"] != "en":
            tags.append(f"language:{variant['language']}")
        if variant["situating_context"] != "none":
            tags.append(f"situating_context:{variant['situating_context']}")
        if is_pre_task and len(config.instruction_positions) > 1:
            tags.append(f"instruction_position:{variant['instruction_position']}")
        if variant["instruction_xml_tags"]:
            tags.append("instruction_xml_tags:True")
        if variant["task_label_names"] is not None:
            tags.append(f"task_label_names:{variant['task_label_names']}")
        if variant["typos"]:
            tags.append("typos:True")
        if variant["punctuation"] != "standard":
            tags.append(f"punctuation:{variant['punctuation']}")
        if variant["scale"] is not None:
            tags.append(f"scale:{variant['scale']}")

        output.append(
            {
                "id": template_id,
                "name": name,
                "type": config.template_type,
                "tags": tags,
                "template": variant["template"],
            }
        )

    return output


def needs_api_calls(config: GeneratorConfig) -> bool:
    """Check if config requires LLM API calls for translations or transformations."""
    needs_translation = any(lang != "en" for lang in config.languages)
    needs_typos = any(config.typos)
    needs_punctuation = any(p != "standard" for p in config.punctuation)
    return needs_translation or needs_typos or needs_punctuation


def generate_templates(
    config: GeneratorConfig,
    client: OpenAICompatibleClient | None = None,
    max_concurrent: int = 10,
) -> list[dict]:
    if needs_api_calls(config) and client is None:
        raise ValueError("Config requires API calls but no client provided")

    instructions = _build_instructions(config)
    if client is not None:
        instructions = _translate_instructions(instructions, config, client, max_concurrent)
        transformed = _apply_transformations(instructions, config, client, max_concurrent)
    else:
        # No API calls needed - expand instruction keys to include typos/punct
        transformed = {
            (phrasing_idx, lang, typos, punct): instruction
            for (phrasing_idx, lang), instruction in instructions.items()
            for typos in config.typos
            for punct in config.punctuation
        }
    variants = _build_variants(transformed, config)
    return _to_output_format(variants, config)


def write_templates_yaml(templates: list[dict], path: Path) -> None:
    with path.open("w") as f:
        yaml.dump(templates, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def generate_and_write(
    config: GeneratorConfig,
    client: OpenAICompatibleClient | None = None,
    max_concurrent: int = 10,
) -> list[dict]:
    templates = generate_templates(config, client, max_concurrent)
    write_templates_yaml(templates, config.output_path)
    return templates


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.measurement.elicitation.prompt_templates.generator <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config, model_name = load_config_from_yaml(config_path)

    client = None
    if needs_api_calls(config):
        client = get_client(model_name=model_name)

    print(f"Generating templates from {config_path}...")
    templates = generate_and_write(config, client)
    print(f"Generated {len(templates)} templates to {config.output_path}")
