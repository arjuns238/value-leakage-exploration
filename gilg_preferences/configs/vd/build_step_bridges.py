"""Bridge artifacts between Gilg pipeline steps (run on pod from repo root).

Usage:
    python configs/vd/build_step_bridges.py \
        --train-run results/experiments/<id>/pre_task_active_learning/<run_name> \
        --eval-run  results/experiments/<id>/pre_task_active_learning/<run_name> \
        --out-dir   /workspace/gilg_runs

Produces:
    measured_task_ids.json  - {"task_ids": [...]} union of both runs' utilities
                              (input for configs/vd/extract_pref.yaml)
    pairs_benign.json       - 150 steering pairs sampled across the utility
                              range from the TRAIN run: task_a = higher-mu task,
                              delta_mu = mu_a - mu_b. Pair selection: stratified
                              by |delta_mu| tercile so the Fig-3 curve sees easy
                              and hard pairs. Excludes bailbench/stresstest
                              (benign pairs, matching their one_sided_benign).
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import csv


def load_utilities(run_dir):
    csvs = glob.glob(str(Path(run_dir) / "thurstonian_*.csv"))
    assert len(csvs) == 1, f"expected exactly one thurstonian CSV in {run_dir}, got {csvs}"
    out = {}
    with open(csvs[0]) as f:
        for row in csv.DictReader(f):
            out[row["task_id"]] = float(row["mu"])
    return out


def load_task_texts(repo_root, task_ids):
    sys.path.insert(0, str(repo_root))
    from src.task_data.loader import load_filtered_tasks
    from src.task_data.task import OriginDataset
    tasks = load_filtered_tasks(
        n=len(task_ids),
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
        task_ids=set(task_ids),
    )
    return {t.id: t.prompt for t in tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--eval-run", required=True)
    ap.add_argument("--out-dir", default="/workspace/gilg_runs")
    ap.add_argument("--n-pairs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mu_train = load_utilities(args.train_run)
    mu_eval = load_utilities(args.eval_run)
    all_ids = sorted(set(mu_train) | set(mu_eval))
    (out / "measured_task_ids.json").write_text(json.dumps({"task_ids": all_ids}))
    print(f"measured_task_ids.json: {len(all_ids)} ids "
          f"(train {len(mu_train)}, eval {len(mu_eval)}, "
          f"overlap {len(set(mu_train) & set(mu_eval))})")

    texts = load_task_texts(Path(__file__).resolve().parents[2], all_ids)
    benign = {tid: mu for tid, mu in mu_train.items()
              if tid in texts and not tid.startswith(("bailbench", "stresstest"))}
    print(f"benign task pool with text: {len(benign)}")

    rng = random.Random(args.seed)
    ids = list(benign)
    pairs = []
    seen = set()
    while len(pairs) < args.n_pairs * 3:  # oversample, then stratify
        a, b = rng.sample(ids, 2)
        if benign[a] < benign[b]:
            a, b = b, a
        key = (a, b)
        if key in seen or benign[a] == benign[b]:
            continue
        seen.add(key)
        pairs.append({
            "pair_id": f"vd_{len(pairs)}",
            "task_a": a, "task_b": b,
            "task_a_text": texts[a], "task_b_text": texts[b],
            "delta_mu": benign[a] - benign[b],
        })
    pairs.sort(key=lambda p: p["delta_mu"])
    k = len(pairs) // 3
    terciles = [pairs[:k], pairs[k:2 * k], pairs[2 * k:]]
    per = args.n_pairs // 3
    chosen = sum((rng.sample(t, per) for t in terciles), [])
    rng.shuffle(chosen)
    (out / "pairs_benign.json").write_text(json.dumps(chosen, indent=1))
    dmus = [p["delta_mu"] for p in chosen]
    print(f"pairs_benign.json: {len(chosen)} pairs, delta_mu "
          f"min {min(dmus):.2f} / median {sorted(dmus)[len(dmus)//2]:.2f} / max {max(dmus):.2f}")


if __name__ == "__main__":
    main()
