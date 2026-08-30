"""Embed task prompts with a sentence transformer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embed_tasks(
    completions_json: Path,
    model_name: str = DEFAULT_MODEL,
    task_id_filter: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode task prompts using a sentence transformer.

    Returns (task_ids, embeddings) with shape (n_tasks, d_embed).
    """
    with open(completions_json) as f:
        completions = json.load(f)

    if task_id_filter is not None:
        completions = [c for c in completions if c["task_id"] in task_id_filter]

    task_ids = np.array([c["task_id"] for c in completions])
    prompts = [c["task_prompt"] for c in completions]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(prompts, show_progress_bar=True, convert_to_numpy=True)

    return task_ids, embeddings


def save_content_embeddings(
    path: Path,
    task_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, task_ids=task_ids, layer_0=embeddings)
    print(f"Saved {len(task_ids)} embeddings ({embeddings.shape[1]}d) to {path}")


def load_content_embeddings(
    path: Path,
    task_id_filter: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    task_ids = data["task_ids"]
    embeddings = data["layer_0"]

    if task_id_filter is not None:
        mask = np.isin(task_ids, list(task_id_filter))
        task_ids = task_ids[mask]
        embeddings = embeddings[mask]

    return task_ids, embeddings
