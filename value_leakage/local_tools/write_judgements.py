"""Write out-of-band judgements back into the estimate-judge cache.

Usage:
    python -m local_tools.write_judgements --experiment main_experiment_accurate \
        --judged judged.jsonl

`--judged` is JSONL with one {"prompt_hash": ..., "answer": ...} per line,
where `answer` is the judge's raw output containing
`<final_estimate>...</final_estimate>` tags (see the rendered prompts in the
pending file). Records are matched back to their judge template (number vs
days) via the pending file's rendered text, so the pending file from the
same run must still exist (default path, or pass --pending).
"""
import argparse
import json
from pathlib import Path

import shared.runner as runner
from shared.judge_jsonl_cache import JsonlJudgeCache
from shared.prompts.thresholds import THRESHOLD_PROMPTS
from local_tools.judge_config import SUBAGENT_JUDGE_CONFIG

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "final_data"


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument(
        "--pending", default=None,
        help="Pending file from run_donation_bet "
             "(default: <DATA_ROOT>/pending_judgements_<experiment>.jsonl)",
    )
    args = parser.parse_args()

    runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")
    cache_dir = runner.estimate_judge_cache_dir(args.experiment)

    pending_path = args.pending or str(
        DATA_ROOT / f"pending_judgements_{args.experiment}.jsonl")
    pending = {r["prompt_hash"]: r["rendered"] for r in _load_jsonl(pending_path)}
    judged = _load_jsonl(args.judged)

    # Distinct judge templates (number / days), each with its own cache dir.
    templates = {}
    for spec in THRESHOLD_PROMPTS.values():
        t = spec["judge_prompt"]
        templates.setdefault(t, t.split("{llm_text}")[0])
    caches = {
        t: JsonlJudgeCache(cache_dir, t, SUBAGENT_JUDGE_CONFIG)
        for t in templates
    }

    n_written, n_skipped = 0, 0
    for row in judged:
        rendered = pending.get(row["prompt_hash"])
        if rendered is None:
            n_skipped += 1
            continue
        cache = next(
            (c for t, c in caches.items()
             if rendered.startswith(templates[t])),
            None,
        )
        if cache is None:
            raise RuntimeError(
                f"No judge template matches pending prompt "
                f"{row['prompt_hash'][:12]}…")
        assert cache.key(rendered) == row["prompt_hash"], row["prompt_hash"]
        if cache.get(rendered) is None:
            cache.append(rendered, {"answer": row["answer"]})
            n_written += 1

    print(f"Wrote {n_written} judgements into {cache_dir} "
          f"({n_skipped} unknown prompt_hashes skipped, "
          f"{len(pending) - n_written - n_skipped} already cached or unjudged)")


if __name__ == "__main__":
    main()
