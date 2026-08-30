"""Merge Sonnet-judge outputs for E5 + E6 and compute final numbers.
Run after all judged_e5choice_*, judged_e6est_*, judged_e6disc_* exist."""
import json, re, glob, math
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)

# ================= E5 =================
e5 = {f'{r["pair_id"]}|r{r["rank"]}|s{r["sample"]}': r
      for r in (json.loads(l) for l in open("artifacts/session_c/e5_gen.jsonl"))}
choice = {}
for fn in glob.glob("judging/session_c/judged_e5choice_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); choice[j["rid"]] = j["choice"]
print(f"E5: {len(choice)}/{len(e5)} judged")
missing = set(e5) - set(choice)
if missing: print(f"  MISSING {len(missing)}: {list(missing)[:5]}")

# activity_1 is the higher-scored (preferred) option in every pair.
by_rank = defaultdict(lambda: {"pref": 0, "other": 0, "refusal": 0})
for rid, ch in choice.items():
    rank = e5[rid]["rank"]; d = by_rank[rank]
    if ch == 1: d["pref"] += 1
    elif ch == 2: d["other"] += 1
    else: d["refusal"] += 1
print("\nE5 — P(picked preferred activity_1) vs rank projected out")
print("rank | n_decisive | P(pref) | 95% CI | refusals")
e5_curve = {}
for k in sorted(by_rank):
    d = by_rank[k]; dec = d["pref"] + d["other"]
    p = d["pref"]/dec if dec else float("nan")
    lo, hi = wilson(d["pref"], dec)
    e5_curve[k] = {"p": p, "n": dec, "ci": [lo, hi], "refusal": d["refusal"]}
    print(f" {k:2d}  | {dec:4d}       | {p:.3f}  | [{lo:.3f},{hi:.3f}] | {d['refusal']}")

# ================= E6 =================
e6 = {f'{r["qkey"]}|{r["framing"]}|c{r["c"]}|s{r["sample"]}': r
      for r in (json.loads(l) for l in open("artifacts/session_c/e6_gen.jsonl"))}
est = {}
for fn in glob.glob("judging/session_c/judged_e6est_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l)
        m = re.search(r"<final_estimate>(.*?)</final_estimate>", j["answer"])
        val = m.group(1).strip() if m else "UNKNOWN"
        est[j["prompt_hash"]] = None if val.upper() == "UNKNOWN" else float(val)
disc = {}
for fn in glob.glob("judging/session_c/judged_e6disc_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); disc[j["prompt_hash"]] = j["category"]
print(f"\nE6: {len(est)}/{len(e6)} estimate-judged, {len(disc)}/{len(e6)} disclosure-judged")

by_c = defaultdict(lambda: {"n": 0, "have_est": 0, "below": 0, "above": 0,
                            "INFLUENCED": 0, "NOT_INFLUENCED": 0, "UNCLEAR": 0,
                            "NO_STATEMENT": 0, "NO_MENTION": 0, "ests": []})
for ph, r in e6.items():
    c = r["c"]; d = by_c[c]; d["n"] += 1
    e = est.get(ph)
    if e is not None:
        d["have_est"] += 1; d["ests"].append(e)
        if e < r["threshold"]: d["below"] += 1
        else: d["above"] += 1
    cat = disc.get(ph)
    if cat in d: d[cat] += 1

print("\nE6 — by steering coefficient")
print("  c   |  n | est-extracted | P(est<thr) | INFLUENCED | NO_MENTION")
for c in sorted(by_c):
    d = by_c[c]
    pb = d["below"]/d["have_est"] if d["have_est"] else float("nan")
    infl = d["INFLUENCED"]/d["n"]
    nom = d["NO_MENTION"]/d["n"]
    print(f" {c:+.2f} | {d['n']:2d} | {d['have_est']:2d}/{d['n']:2d}       | {pb:.2f}      | {infl:.2f}       | {nom:.2f}")

# stated-intent = INFLUENCED (analogous to exp02 77%); also split by framing
print("\nE6 — P(est<thr) split by framing (does steering move the number per-framing?)")
by_cf = defaultdict(lambda: {"below": 0, "have": 0})
for ph, r in e6.items():
    e = est.get(ph)
    if e is None: continue
    key = (r["c"], r["framing"]); by_cf[key]["have"] += 1
    if e < r["threshold"]: by_cf[key]["below"] += 1
for framing in ("below_good", "above_good"):
    row = []
    for c in sorted(set(k[0] for k in by_cf)):
        d = by_cf[(c, framing)]
        row.append(f"{d['below']}/{d['have']}={d['below']/d['have']:.2f}" if d['have'] else "na")
    print(f"  {framing:11s}: " + " | ".join(row))

# disclosure category totals
print("\nE6 disclosure category totals:")
tot = defaultdict(int)
for cat in disc.values(): tot[cat] += 1
for k in ("INFLUENCED","NOT_INFLUENCED","UNCLEAR","NO_STATEMENT","NO_MENTION"):
    print(f"  {k}: {tot[k]} ({tot[k]/max(1,len(disc)):.1%})")

json.dump({"e5_curve": {str(k): v for k, v in e5_curve.items()},
           "e6_by_c": {str(k): {kk: (vv if kk != 'ests' else None) for kk, vv in v.items()}
                       for k, v in by_c.items()}},
          open("judging/session_c/e5e6_final.json", "w"), indent=2)
print("\nwrote judging/session_c/e5e6_final.json")
