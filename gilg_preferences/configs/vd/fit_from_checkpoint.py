"""Fit Thurstonian utilities directly from an active-learning checkpoint.

For runs we stop early (after iteration 1): reads checkpoint.yaml's
comparisons, fits utilities with the repo's own fitter, and writes
thurstonian_<hash>.csv/.yaml + measurements.yaml into the run dir — the
exact artifacts the probe trainer expects from a completed run.

Usage (from repo root, vd-venv):
    python configs/vd/fit_from_checkpoint.py <run_dir>
"""
import sys
import yaml
import numpy as np
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.task_data.loader import load_filtered_tasks
from src.task_data.task import OriginDataset
from src.fitting.thurstonian_fitting.thurstonian import (
    PairwiseData, fit_thurstonian, save_thurstonian,
)


def origin_of(task_id):
    for prefix, origin in [("alpaca_", "ALPACA"), ("wildchat_", "WILDCHAT"),
                           ("competition_math_", "MATH"), ("bailbench_", "BAILBENCH"),
                           ("stresstest_", "STRESS_TEST")]:
        if task_id.startswith(prefix):
            return origin
    return "WILDCHAT"  # fallback; loader needs a valid enum name


def main():
    run_dir = Path(sys.argv[1])
    ckpt = yaml.safe_load(open(run_dir / "checkpoint.yaml"))
    comps = ckpt["comparisons"]
    by_choice = Counter(c["choice"] for c in comps)
    print(f"checkpoint iteration={ckpt.get('iteration')} comparisons={len(comps)}")
    print("choice distribution:", dict(by_choice))
    refusal_rate = by_choice.get("refusal", 0) / max(len(comps), 1)
    print(f"refusal/unparsed rate: {refusal_rate:.1%}")

    decisive = [c for c in comps if c["choice"] in ("a", "b")]
    ids = sorted({c["task_a"] for c in decisive} | {c["task_b"] for c in decisive})
    id_set = set(ids)
    tasks = load_filtered_tasks(
        n=len(id_set),
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
        task_ids=id_set,
    )
    assert len(tasks) == len(ids), f"task text missing for {len(ids)-len(tasks)} ids"
    idx = {t.id: i for i, t in enumerate(tasks)}

    win_counts = Counter()
    for c in decisive:
        w, l = ((c["task_a"], c["task_b"]) if c["choice"] == "a"
                else (c["task_b"], c["task_a"]))
        win_counts[(idx[w], idx[l])] += 1
    row = np.array([k[0] for k in win_counts], dtype=np.int32)
    col = np.array([k[1] for k in win_counts], dtype=np.int32)
    count = np.array(list(win_counts.values()), dtype=np.int32)
    data = PairwiseData(tasks, _row=row, _col=col, _count=count)
    print(f"fitting: {len(tasks)} tasks, {int(count.sum())} decisive comparisons")

    result = fit_thurstonian(data)
    print(f"converged={result.converged} nll={result.neg_log_likelihood:.1f}")
    save_thurstonian(result, run_dir / "thurstonian.yaml", "map",
                     config={"source": "vd_fit_from_checkpoint", "iteration": ckpt.get("iteration")})

    rows = [{"task_a": c["task_a"], "task_b": c["task_b"], "choice": c["choice"],
             "origin_a": origin_of(c["task_a"]), "origin_b": origin_of(c["task_b"])}
            for c in comps]
    with open(run_dir / "measurements.yaml", "w") as f:
        yaml.safe_dump(rows, f, sort_keys=False)
    print(f"wrote thurstonian_*.csv/.yaml + measurements.yaml ({len(rows)} rows) to {run_dir}")

    mus = result.mu
    order = np.argsort(mus)
    print("\nLOWEST-utility tasks:")
    for i in order[:5]:
        print(f"  mu={mus[i]:+.2f} [{tasks[i].id}] {tasks[i].prompt[:90]!r}")
    print("HIGHEST-utility tasks:")
    for i in order[-5:]:
        print(f"  mu={mus[i]:+.2f} [{tasks[i].id}] {tasks[i].prompt[:90]!r}")


if __name__ == "__main__":
    main()
