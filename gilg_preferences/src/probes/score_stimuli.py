"""Score a set of stimuli through a model with per-probe per-selector scoring.

Generalises the per-experiment scoring loop used in
`experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py`:
one forward pass per stimulus via `score_prompt_all_tokens`, then post-hoc
slicing to each probe's matched turn-boundary selector.

Supported selectors: `turn_boundary:N` with N < 0, under
`add_generation_prompt=False`. In that regime the sequence ends with the
turn-end marker and `scores_arr[N]` equals the `turn_boundary:N` position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm import tqdm

from src.probes.scoring import score_prompt_all_tokens

if TYPE_CHECKING:
    from src.models.huggingface_model import HuggingFaceModel


@dataclass
class Probe:
    key: str            # e.g. "qwen_tb-1_L38"
    selector: str       # e.g. "turn_boundary:-1" — must be turn_boundary:N with N<0
    layer: int
    weights: np.ndarray  # (d_model + 1,), last element is intercept


def _parse_tb_offset(selector: str) -> int:
    prefix = "turn_boundary:"
    if not selector.startswith(prefix):
        raise ValueError(f"score_stimuli only supports turn_boundary:N selectors; got {selector!r}")
    offset = int(selector[len(prefix):])
    if offset >= 0:
        raise ValueError(f"turn_boundary offset must be < 0 for add_generation_prompt=False; got {offset}")
    return offset


def score_stimuli_with_probes(
    model: HuggingFaceModel,
    stimuli: list[dict],
    probes: list[Probe],
    add_generation_prompt: bool = False,
    progress: bool = True,
    extra_extractors: dict[str, Callable[[np.ndarray, dict], float]] | None = None,
) -> list[dict]:
    """Score every stimulus; return a new record per stimulus with `probe_scores` populated.

    Parameters
    ----------
    model
        Loaded HuggingFaceModel.
    stimuli
        List of dicts, each with at least a `messages` field.
    probes
        List of `Probe` (key, selector, layer, weights). The same (layer, weights)
        pair may appear under multiple selectors.
    add_generation_prompt
        Passed through to `score_prompt_all_tokens`. Must be False for
        turn_boundary:N with N<0 to resolve to the expected structural token.
    progress
        Show a tqdm bar.
    extra_extractors
        Optional extra fields to compute from the per-token scores array.
        Each `extractor(scores_arr, stimulus) -> float` is called once per probe
        and its result is placed at `<field>[probe.key]`.

    Returns
    -------
    list[dict]
        One record per input stimulus, augmented with:
          - `probe_scores`: {probe_key: float} — score at the probe's selector.
          - optionally the keys from `extra_extractors`.
    """
    if add_generation_prompt:
        raise ValueError(
            "score_stimuli_with_probes requires add_generation_prompt=False so "
            "turn_boundary:N resolves to a concrete token slice."
        )
    for p in probes:
        _parse_tb_offset(p.selector)

    # Build a callback list for score_prompt_all_tokens. Each Probe gets its own
    # entry so a probe trained at tb-1 and a probe trained at tb-4, even at the
    # same layer, are returned as distinct score arrays we can slice independently.
    scoring_probes: list[tuple[int, np.ndarray]] = [(p.layer, p.weights) for p in probes]

    iterator = tqdm(stimuli, desc="scoring", disable=not progress)

    scored = []
    for item in iterator:
        messages = item["messages"]
        all_scores = score_prompt_all_tokens(
            model, messages, scoring_probes, add_generation_prompt=add_generation_prompt,
        )
        # all_scores is a list of (seq_len,) arrays, one per entry in scoring_probes.

        probe_scores = {}
        extra = {name: {} for name in (extra_extractors or {})}
        for i, p in enumerate(probes):
            scores_arr = all_scores[i]
            offset = int(p.selector.split(":")[1])
            probe_scores[p.key] = float(scores_arr[offset])
            for name, fn in (extra_extractors or {}).items():
                extra[name][p.key] = float(fn(scores_arr, item))

        record = {**item, "probe_scores": probe_scores}
        for name in extra:
            record[name] = extra[name]
        scored.append(record)

    return scored


def load_probes_from_manifest(
    probe_sets: dict[str, tuple[str, str]],
    layers: list[int],
) -> list[Probe]:
    """Convenience loader.

    `probe_sets[name] = (selector, probe_dir)` where probe_dir contains
    `probe_ridge_L{layer}.npy` files. Each (name, layer) produces a Probe
    with key `{name}_L{layer}` and the given selector.
    """
    from pathlib import Path
    probes: list[Probe] = []
    for name, (selector, probe_dir) in probe_sets.items():
        probe_dir = Path(probe_dir)
        for layer in layers:
            weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
            probes.append(Probe(
                key=f"{name}_L{layer}",
                selector=selector,
                layer=layer,
                weights=weights,
            ))
    return probes
