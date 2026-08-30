"""Choosing Activities sampler for locally served models (E2 in notes/03).

Reuses Betley's exact prompt templates (choosing_activities/prompts/) and
their two-stream seeding scheme (pair RNG separate from template RNG), but
works with any activity YAML and the `openai_chat` backend. Judging happens
offline (Claude Code subagents), so this only samples and stores raw texts.

Stages (each idempotent; JSONL outputs are resumable append-caches keyed by
stable ids):
    score    - stated 0-100 liking per activity (n_repeats x 5 wordings)
    pick     - "pick randomly" pairwise choices (n_variations)
    prefer   - matched "pick whichever you'd genuinely prefer" control

Usage:
    python -m local_tools.run_choosing_activities --model gemma-3-27b-it \
        --activities <combined.yaml> --stage score --repeats 20
    ... --stage pick --variations 10000 [--seed 0]
    ... --stage prefer --variations 2000

Outputs under data/final_data/choosing/:
    scores.jsonl   {id, activity_ix, activity, prompt_ix, answer}
    picks.jsonl    {id, variation_ix, activity_1, activity_2, ix_1, ix_2,
                    template_ix, prompt, answer}
    prefers.jsonl  same shape as picks.jsonl
"""
import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from shared.models import MODELS
from shared.runner import _create_sender

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "final_data" / "choosing"
PROMPTS_DIR = REPO / "choosing_activities" / "prompts"

PROMPT_SELECTION_SEED = 1234  # matches their score-prompt balancing intent


def load_activities(path):
    data = yaml.safe_load(open(path))
    acts = data["activities"] if isinstance(data, dict) else data
    # entries are either bare strings or {"name": ...} dicts (their format)
    acts = [a["name"] if isinstance(a, dict) else a for a in acts]
    assert len(acts) == len(set(acts)), "duplicate activities"
    return acts


def load_templates(name, key):
    return yaml.safe_load(open(PROMPTS_DIR / name))[key]


def build_variations(n, seed, activities, templates):
    """Their scheme: pair RNG and template RNG are independent streams, so a
    matched run with different templates keeps identical pairings."""
    out = []
    for v in range(n):
        rng = random.Random(f"{seed}:{v}")
        a, b = rng.sample(range(len(activities)), 2)
        order = [a, b]
        rng.shuffle(order)
        t_rng = random.Random(f"{seed}:{v}:template")
        t_ix = t_rng.randrange(len(templates))
        out.append({
            "variation_ix": v,
            "ix_1": order[0], "ix_2": order[1],
            "activity_1": activities[order[0]],
            "activity_2": activities[order[1]],
            "template_ix": t_ix,
        })
    return out


def _load_done(path):
    done = set()
    if path.exists():
        for line in open(path):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def _run(tasks, sender, max_concurrent, out_path):
    """tasks: [(id, prompt, meta_dict)] -> append {..meta, id, prompt, answer}."""
    done = _load_done(out_path)
    todo = [t for t in tasks if t[0] not in done]
    print(f"{out_path.name}: {len(done)} cached, {len(todo)} to sample")
    if not todo:
        return
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a")
    bar = tqdm(total=len(todo))

    def one(task):
        tid, prompt, meta = task
        r = sender(prompt)
        return {**meta, "id": tid, "prompt": prompt, "answer": r["answer"]}

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = [ex.submit(one, t) for t in todo]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                bar.update(1)
    bar.close()
    f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemma-3-27b-it", choices=sorted(MODELS))
    p.add_argument("--activities", required=True)
    p.add_argument("--stage", required=True, choices=["score", "pick", "prefer"])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--variations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-concurrent", type=int, default=32)
    args = p.parse_args()

    model = dict(MODELS[args.model])
    sender = _create_sender(model)
    activities = load_activities(args.activities)
    print(f"{len(activities)} activities from {args.activities}")

    if args.stage == "score":
        score_prompts = load_templates("activity_liking_prompt.yaml", "score_prompts")
        tasks = []
        for ix, act in enumerate(activities):
            rng = random.Random(f"{PROMPT_SELECTION_SEED}:{ix}")
            # balanced spread of the wordings over repeats (their intent)
            prompt_ixs = [(i % len(score_prompts)) for i in range(args.repeats)]
            rng.shuffle(prompt_ixs)
            for rep, p_ix in enumerate(prompt_ixs):
                tasks.append((
                    f"s{ix}:{rep}",
                    score_prompts[p_ix].format(activity=act),
                    {"activity_ix": ix, "activity": act, "prompt_ix": p_ix,
                     "rep": rep},
                ))
        _run(tasks, sender, args.max_concurrent, OUT_DIR / "scores.jsonl")

    else:
        tmpl_file = ("prompt_template.yaml" if args.stage == "pick"
                     else "preference_template.yaml")
        templates = load_templates(tmpl_file, "prompt_templates")
        variations = build_variations(args.variations, args.seed,
                                      activities, templates)
        tag = "p" if args.stage == "pick" else "r"
        tasks = []
        for v in variations:
            prompt = templates[v["template_ix"]].format(
                activity_1=v["activity_1"], activity_2=v["activity_2"])
            tasks.append((f"{tag}{v['variation_ix']}", prompt, v))
        out = OUT_DIR / ("picks.jsonl" if args.stage == "pick" else "prefers.jsonl")
        _run(tasks, sender, args.max_concurrent, out)


if __name__ == "__main__":
    main()
