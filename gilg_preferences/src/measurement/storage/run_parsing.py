"""Utilities for parsing experiment run directory names and normalizing scores."""

from __future__ import annotations

import re

MODEL_PREFIXES = ("qwen3", "llama", "gemma", "claude", "gpt")


def parse_scale_tag(scale_tag: str) -> tuple[float, float] | None:
    """Parse a scale tag like '1-5', '-5_5', or 'neg5-pos5' into (min, max).

    Returns None for non-numeric scales (e.g., 'lemon|grape|orange').
    """
    # Handle negative notation: neg5-pos5 -> (-5, 5)
    scale_tag = scale_tag.replace("neg", "-").replace("pos", "")
    # Try underscore separator first (e.g., '-5_5')
    match = re.match(r"(-?\d+)_(-?\d+)", scale_tag)
    if match:
        return float(match.group(1)), float(match.group(2))
    # Try dash separator (e.g., '1-5', '27-32')
    match = re.match(r"(-?\d+)-(-?\d+)", scale_tag)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def extract_model_from_run_dir(dir_name: str) -> str | None:
    """Extract model name from run directory name.

    Format: {template}_{model}_{format}_cseed{n}_rseed{n}
    e.g. 'ban_four_1_5_claude-haiku-4.5_regex_cseed0_rseed0' -> 'claude-haiku-4.5'
    """
    parts = dir_name.split("_")
    for i, part in enumerate(parts):
        if part == "regex":
            for j in range(1, i):
                candidate = "_".join(parts[j:i])
                if any(candidate.startswith(m) for m in MODEL_PREFIXES):
                    return candidate
    return None


def extract_template_from_run_dir(dir_name: str) -> str | None:
    """Extract template name from run directory name.

    Format: {template}_{model}_{format}_cseed{n}_rseed{n}
    e.g. 'ban_four_1_5_claude-haiku-4.5_regex_cseed0_rseed0' -> 'ban_four_1_5'
    """
    parts = dir_name.split("_")
    for i, part in enumerate(parts):
        if part == "regex":
            for j in range(1, i):
                candidate = "_".join(parts[j:i])
                if any(candidate.startswith(m) for m in MODEL_PREFIXES):
                    return "_".join(parts[:j])
    return None


def normalize_score(score: float, scale: tuple[float, float]) -> float:
    """Normalize score to 0-1 range based on scale (min, max)."""
    min_val, max_val = scale
    return (score - min_val) / (max_val - min_val)
