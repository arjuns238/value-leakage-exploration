"""Unified API-call-level cache for preference measurements.

Provides separate caches for stated (single-task scores) and revealed (pairwise comparisons)
measurements, keyed by all factors that define a unique API call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.measurement.storage.base import load_yaml, model_short_name, save_yaml
from src.types import Message

if TYPE_CHECKING:
    from src.measurement.elicitation.prompt_templates.template import PromptTemplate


def _context_hash(context_messages: list[Message] | None) -> str | None:
    if context_messages is None:
        return None
    ctx_str = json.dumps(context_messages, sort_keys=True)
    return hashlib.sha256(ctx_str.encode()).hexdigest()[:8]


def template_config_from_template(t: PromptTemplate) -> dict:
    """Extract cache-relevant config from template."""
    return {
        "name": t.name,
        "tags": dict(t.tags_dict),
        "template_hash": hashlib.sha256(t.template.encode()).hexdigest()[:12],
    }


class StatedCache:
    """Cache for stated preference measurements (single-task scores).

    Keys by: template_config, response_format, rating_seed, task_id, completion_seed (optional).
    Stores samples as list of {score: float} dicts.
    """

    CACHE_DIR = Path("results/cache/stated")

    def __init__(self, model: str):
        self.model = model
        self.model_short = model_short_name(model)
        self._cache_path = self.CACHE_DIR / f"{self.model_short}.yaml"
        self._data: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._cache_path.exists():
            return load_yaml(self._cache_path)
        return {}

    def _make_key(
        self,
        template_config: dict,
        response_format: str,
        rating_seed: int,
        task_id: str,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        completion_model: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> str:
        """Create stable hash from key fields."""
        parts = [
            template_config["name"],
            template_config["template_hash"],
            response_format,
            str(rating_seed),
            task_id,
        ]
        if completion_seed is not None:
            parts.append(f"cseed{completion_seed}")
        # Always include system_prompt in key to avoid collisions with pre-fix entries
        if system_prompt is not None:
            parts.append(f"sys{hashlib.sha256(system_prompt.encode()).hexdigest()[:8]}")
        else:
            parts.append("sys_none")
        ctx_h = _context_hash(context_messages)
        parts.append(f"ctx{ctx_h}" if ctx_h else "ctx_none")
        if completion_model is not None:
            parts.append(f"cmodel{model_short_name(completion_model)}")
        return hashlib.sha256("__".join(parts).encode()).hexdigest()[:16]

    def get(
        self,
        template_config: dict,
        response_format: str,
        rating_seed: int,
        task_id: str,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        completion_model: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> list[dict]:
        """Get all samples for a stated measurement."""
        if self._data is None:
            self._data = self._load()
        h = self._make_key(template_config, response_format, rating_seed, task_id, completion_seed, system_prompt, completion_model, context_messages)
        entry = self._data.get(h)
        return entry["samples"] if entry else []

    def add(
        self,
        template_config: dict,
        response_format: str,
        rating_seed: int,
        task_id: str,
        sample: dict,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        completion_model: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> None:
        """Add a sample to stated measurement."""
        if self._data is None:
            self._data = self._load()
        h = self._make_key(template_config, response_format, rating_seed, task_id, completion_seed, system_prompt, completion_model, context_messages)
        if h not in self._data:
            entry = {
                "template_config": template_config,
                "response_format": response_format,
                "rating_seed": rating_seed,
                "task_id": task_id,
                "samples": [],
            }
            if completion_seed is not None:
                entry["completion_seed"] = completion_seed
            if system_prompt is not None:
                entry["system_prompt"] = system_prompt
            if completion_model is not None:
                entry["completion_model"] = completion_model
            ctx_h = _context_hash(context_messages)
            if ctx_h is not None:
                entry["context_hash"] = ctx_h
            self._data[h] = entry
        self._data[h]["samples"].append(sample)

    def get_task_ids(
        self,
        template_config: dict,
        response_format: str,
        rating_seed: int,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        completion_model: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> set[str]:
        """Get all task IDs that have been measured for this configuration."""
        if self._data is None:
            self._data = self._load()

        ctx_h = _context_hash(context_messages)
        task_ids = set()
        for entry in self._data.values():
            if (
                entry["template_config"]["name"] == template_config["name"]
                and entry["template_config"]["template_hash"] == template_config["template_hash"]
                and entry["response_format"] == response_format
                and entry["rating_seed"] == rating_seed
                and entry.get("system_prompt") == system_prompt
                and entry.get("completion_model") == completion_model
                and entry.get("context_hash") == ctx_h
            ):
                # Check completion_seed match
                entry_cseed = entry.get("completion_seed")
                if completion_seed is None and entry_cseed is None:
                    task_ids.add(entry["task_id"])
                elif completion_seed is not None and entry_cseed == completion_seed:
                    task_ids.add(entry["task_id"])

        return task_ids

    def save(self) -> None:
        if self._data is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(self._data, self._cache_path)


class RevealedCache:
    """Cache for revealed preference measurements (pairwise comparisons).

    Keys by: template_config, response_format, order, rating_seed, task_a_id, task_b_id, completion_seed (optional).
    Stores samples as list of {choice: str} dicts.
    """

    CACHE_DIR = Path("results/cache/revealed")

    def __init__(self, model: str):
        self.model = model
        self.model_short = model_short_name(model)
        self._cache_path = self.CACHE_DIR / f"{self.model_short}.yaml"
        self._data: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._cache_path.exists():
            return load_yaml(self._cache_path)
        return {}

    def _make_key(
        self,
        template_config: dict,
        response_format: str,
        order: str,
        rating_seed: int,
        task_a_id: str,
        task_b_id: str,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> str:
        parts = [
            template_config["name"],
            template_config["template_hash"],
            response_format,
            order,
            str(rating_seed),
            task_a_id,
            task_b_id,
        ]
        if completion_seed is not None:
            parts.append(f"cseed{completion_seed}")
        # Always include system_prompt in key to avoid collisions with pre-fix entries
        if system_prompt is not None:
            parts.append(f"sys{hashlib.sha256(system_prompt.encode()).hexdigest()[:8]}")
        else:
            parts.append("sys_none")
        ctx_h = _context_hash(context_messages)
        parts.append(f"ctx{ctx_h}" if ctx_h else "ctx_none")
        return hashlib.sha256("__".join(parts).encode()).hexdigest()[:16]

    def get(
        self,
        template_config: dict,
        response_format: str,
        order: str,
        rating_seed: int,
        task_a_id: str,
        task_b_id: str,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> list[dict]:
        if self._data is None:
            self._data = self._load()
        h = self._make_key(template_config, response_format, order, rating_seed, task_a_id, task_b_id, completion_seed, system_prompt, context_messages)
        entry = self._data.get(h)
        return entry["samples"] if entry else []

    def add(
        self,
        template_config: dict,
        response_format: str,
        order: str,
        rating_seed: int,
        task_a_id: str,
        task_b_id: str,
        sample: dict,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> None:
        if self._data is None:
            self._data = self._load()
        h = self._make_key(template_config, response_format, order, rating_seed, task_a_id, task_b_id, completion_seed, system_prompt, context_messages)
        if h not in self._data:
            entry = {
                "template_config": template_config,
                "response_format": response_format,
                "order": order,
                "rating_seed": rating_seed,
                "task_a_id": task_a_id,
                "task_b_id": task_b_id,
                "samples": [],
            }
            if completion_seed is not None:
                entry["completion_seed"] = completion_seed
            if system_prompt is not None:
                entry["system_prompt"] = system_prompt
            ctx_h = _context_hash(context_messages)
            if ctx_h is not None:
                entry["context_hash"] = ctx_h
            self._data[h] = entry
        self._data[h]["samples"].append(sample)

    def get_pairs(
        self,
        template_config: dict,
        response_format: str,
        order: str,
        rating_seed: int,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> set[tuple[str, str]]:
        """Get all (task_a_id, task_b_id) pairs that have been measured for this configuration."""
        if self._data is None:
            self._data = self._load()

        ctx_h = _context_hash(context_messages)
        pairs = set()
        for entry in self._data.values():
            if (
                entry["template_config"]["name"] == template_config["name"]
                and entry["template_config"]["template_hash"] == template_config["template_hash"]
                and entry["response_format"] == response_format
                and entry["order"] == order
                and entry["rating_seed"] == rating_seed
                and entry.get("system_prompt") == system_prompt
                and entry.get("context_hash") == ctx_h
            ):
                # Check completion_seed match
                entry_cseed = entry.get("completion_seed")
                if completion_seed is None and entry_cseed is None:
                    pairs.add((entry["task_a_id"], entry["task_b_id"]))
                elif completion_seed is not None and entry_cseed == completion_seed:
                    pairs.add((entry["task_a_id"], entry["task_b_id"]))

        return pairs

    def get_measurements(
        self,
        template_config: dict,
        response_format: str,
        order: str,
        rating_seed: int,
        task_ids: set[str] | None = None,
        completion_seed: int | None = None,
        system_prompt: str | None = None,
        context_messages: list[Message] | None = None,
    ) -> list[dict]:
        """Get all measurements for a configuration, optionally filtered by task IDs.

        Returns list of {task_a: str, task_b: str, choice: str} dicts.
        """
        if self._data is None:
            self._data = self._load()

        ctx_h = _context_hash(context_messages)
        results = []
        for entry in self._data.values():
            if (
                entry["template_config"]["name"] == template_config["name"]
                and entry["template_config"]["template_hash"] == template_config["template_hash"]
                and entry["response_format"] == response_format
                and entry["order"] == order
                and entry["rating_seed"] == rating_seed
                and entry.get("system_prompt") == system_prompt
                and entry.get("context_hash") == ctx_h
            ):
                # Check completion_seed match
                entry_cseed = entry.get("completion_seed")
                if not ((completion_seed is None and entry_cseed is None) or
                        (completion_seed is not None and entry_cseed == completion_seed)):
                    continue

                task_a = entry["task_a_id"]
                task_b = entry["task_b_id"]

                # Filter by task_ids if provided
                if task_ids is not None and (task_a not in task_ids or task_b not in task_ids):
                    continue

                for sample in entry["samples"]:
                    results.append({
                        "task_a": task_a,
                        "task_b": task_b,
                        "choice": sample["choice"],
                    })

        return results

    def save(self) -> None:
        if self._data is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(self._data, self._cache_path)
