"""Donation Bet driver for locally served models with out-of-band judging.

Usage:
    python -m local_tools.run_donation_bet --model gemma-3-27b-it \
        [--experiment main_experiment_accurate]

Like `donation_bet.get_data` but with the estimate judge replaced by the
`manual` backend (see local_tools/judge_config.py). Run it in a loop:

  1st run : samples all baselines, then exits 3 with baseline judge
            prompts pending in --pending-out.
            -> judge them, write back with local_tools.write_judgements
  2nd run : baselines cache-hit, thresholds computed, directions sampled,
            exits 3 with direction judge prompts pending.
            -> judge, write back
  3rd run : everything cache-hits; computes bias, saves rollouts + summary.

Exit codes: 0 = complete, 3 = judgements pending.
"""
import argparse
import json
import sys
from pathlib import Path

import os

import shared.runner as runner
from shared.models import MODELS
from shared.experiments import THRESHOLD_EXPERIMENTS
from shared.get_main_dfs import get_main_dfs
from local_tools.judge_config import SUBAGENT_JUDGE_CONFIG

from donation_bet.bias_metrics import balanced_bias_bootstrap_ci95

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "final_data"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--experiment", default="main_experiment_accurate",
        choices=sorted(THRESHOLD_EXPERIMENTS),
    )
    parser.add_argument(
        "--pending-out", default=None,
        help="Where pending judge prompts get dumped "
             "(default: <DATA_ROOT>/pending_judgements_<experiment>.jsonl)",
    )
    args = parser.parse_args()

    pending_path = args.pending_out or str(
        DATA_ROOT / f"pending_judgements_{args.experiment}.jsonl")
    os.environ["MANUAL_JUDGE_PENDING_PATH"] = pending_path

    # Same cache redirection as donation_bet.get_data.
    runner.CACHE_DIR = str(DATA_ROOT / "cache")
    runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")
    # Out-of-band judge (read at call time by batch_extract_estimates).
    runner.ESTIMATE_JUDGE_CONFIG = SUBAGENT_JUDGE_CONFIG

    print(f"Model:       {args.model}")
    print(f"Experiment:  {args.experiment}")
    print(f"Model cache: {runner.CACHE_DIR}")
    print(f"Judge cache: {runner.estimate_judge_cache_dir(args.experiment)}")

    try:
        out = get_main_dfs(args.experiment, [args.model])
    except runner.ManualJudgePending as e:
        print(f"\nPENDING: {e}")
        sys.exit(3)

    df, thresholds, display_name = out[args.model]

    # Persist the parsed rollouts for downstream covertness judging/plots.
    out_dir = DATA_ROOT / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = out_dir / f"{args.model}_{args.experiment}.jsonl"
    df.to_json(rollout_path, orient="records", lines=True)

    prompt_keys = THRESHOLD_EXPERIMENTS[args.experiment]["prompts"]
    bias, lo, hi = balanced_bias_bootstrap_ci95(df, prompt_keys=prompt_keys)

    per_question = {}
    for pk in prompt_keys:
        b, qlo, qhi = balanced_bias_bootstrap_ci95(df[df["prompt_key"] == pk])
        per_question[pk] = {"bias": b, "ci95": [b - qlo, b + qhi]}

    summary = {
        "model": args.model,
        "display_name": display_name,
        "experiment": args.experiment,
        "n_rows_parsed": int(len(df)),
        "thresholds": {k: v for k, v in thresholds.items()},
        "bias": bias,
        "bias_ci95": [bias - lo, bias + hi],
        "per_question": per_question,
    }
    summary_path = out_dir / f"{args.model}_{args.experiment}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nDone: {display_name} ({len(df)} parsed rows)")
    print(f"Balanced bias = {bias:.3f}  "
          f"95% CI [{bias - lo:.3f}, {bias + hi:.3f}]")
    print(f"Rollouts: {rollout_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
