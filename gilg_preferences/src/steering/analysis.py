"""Load steering checkpoints and aggregate results.

Key invariant: `signed_multiplier` encodes the signed steering coefficient
in original task space. Task A always receives +direction x coefficient;
task B receives -direction x coefficient. `_effective_coef` in the runner
negates the coefficient for ordering=1 to maintain this.

Primary metric: `compute_p_steered` — P(completed steered task) at each
signed coefficient. This is the unfolded sigmoid: near 1 at positive c
(task A boosted), 0.5 at zero, near 0 at negative c (task A anti-steered).
Denominator includes all completions (neither/refusal count against).

`aggregate` is the lower-level building block that computes P(chose A)
among valid-only choices. Use it for custom slicing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_checkpoint(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def filter_valid(rows: list[dict]) -> list[dict]:
    """Keep only rows with a valid choice (a or b), dropping refusals."""
    return [r for r in rows if r["choice_original"] in ("a", "b")]


def aggregate(
    rows: list[dict],
    group_by: list[str] = ["condition", "layer", "signed_multiplier"],
) -> list[dict]:
    """Aggregate P(chose_a) over groups, pooling across orderings.

    Returns one row per group with:
      - all group_by fields
      - p_a: P(choice_original == "a") among valid choices only
      - n: number of valid trials (excluding refusals)
      - n_refusal: number of refusals
      - n_by_ordering: {0: count, 1: count} — to verify balance
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_by)
        buckets[key].append(row)

    results = []
    for key, bucket in sorted(buckets.items()):
        valid = [r for r in bucket if r["choice_original"] in ("a", "b")]
        n_a = sum(1 for r in valid if r["choice_original"] == "a")
        n_by_ordering = defaultdict(int)
        for r in valid:
            n_by_ordering[r["ordering"]] += 1

        result = dict(zip(group_by, key))
        result["p_a"] = n_a / len(valid) if valid else float("nan")
        result["n"] = len(valid)
        result["n_refusal"] = len(bucket) - len(valid)
        result["n_by_ordering"] = dict(n_by_ordering)
        results.append(result)

    return results


def compute_p_steered(
    rows: list[dict],
    group_by: list[str] = ["condition", "layer"],
    choice_field: str = "choice_original",
) -> list[dict]:
    """P(completed steered task) at each signed coefficient.

    Task A always receives +direction x c (by convention of signed_multiplier).
    This computes P(choice_field == "a") / n_total at each coefficient,
    giving the unfolded sigmoid:
      c > 0: A boosted → p_steered ~ 1.0
      c = 0: no steering → p_steered ~ 0.5
      c < 0: A anti-steered → p_steered ~ 0.0

    Denominator includes ALL completions (neither/refusal count against).
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_by) + (row["signed_multiplier"],)
        buckets[key].append(row)

    results = []
    for key, bucket in sorted(buckets.items()):
        n_total = len(bucket)
        n_chose_a = sum(1 for r in bucket if r[choice_field] == "a")
        n_neither = sum(1 for r in bucket if r[choice_field] not in ("a", "b"))

        result = dict(zip(group_by, key[:len(group_by)]))
        result["signed_multiplier"] = key[-1]
        result["p_steered"] = n_chose_a / n_total if n_total > 0 else float("nan")
        result["n_total"] = n_total
        result["n_neither"] = n_neither
        results.append(result)

    return results


def print_summary(
    rows: list[dict],
    group_by: list[str] = ["condition", "layer", "signed_multiplier"],
) -> None:
    """Print a compact summary table."""
    agg = aggregate(rows, group_by)

    sections: dict[tuple, list[dict]] = defaultdict(list)
    for r in agg:
        section_key = tuple(r[k] for k in group_by[:-1])
        sections[section_key].append(r)

    for section_key, section_rows in sections.items():
        header = " / ".join(f"{k}={v}" for k, v in zip(group_by[:-1], section_key))
        print(f"\n{header}:")
        mult_key = group_by[-1]
        for r in section_rows:
            mult = r[mult_key]
            p_a = r["p_a"]
            n = r["n"]
            bal = r["n_by_ordering"]
            shift = p_a - 0.5
            print(f"  {mult_key}={mult:+.3f}: P(a)={p_a:.3f} (n={n}, ord={bal}) shift={shift:+.3f}")
