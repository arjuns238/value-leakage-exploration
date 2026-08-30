"""Merge E9 judge outputs; compute the bet-type x ablation 2x2.
Question: does ablating the value subspace reduce the bet's stated influence / anchoring?"""
import json, re, glob, math
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)

e9 = {f'{r["qkey"]}|{r["condition"]}|r{r["rank"]}|s{r["sample"]}': r
      for r in (json.loads(l) for l in open("artifacts/session_c/e9_gen.jsonl"))}
est, disc = {}, {}
for fn in glob.glob("judging/session_c/judged_e9est_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); m = re.search(r"<final_estimate>(.*?)</final_estimate>", j["answer"])
        v = m.group(1).strip() if m else "UNKNOWN"
        est[j["prompt_hash"]] = None if v.upper() == "UNKNOWN" else float(v)
for fn in glob.glob("judging/session_c/judged_e9disc_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); disc[j["prompt_hash"]] = j["category"]
print(f"E9: {len(est)}/{len(e9)} estimate-judged, {len(disc)}/{len(e9)} disclosure-judged\n")

cells = defaultdict(lambda: {"n":0,"have_est":0,"below":0,"INFLUENCED":0,"NO_MENTION":0,
                             "NO_STATEMENT":0,"NOT_INFLUENCED":0,"UNCLEAR":0})
for k, r in e9.items():
    c = cells[(r["condition"], r["rank"])]; c["n"] += 1
    e = est.get(k)
    if e is not None:
        c["have_est"] += 1
        if e < r["threshold"]: c["below"] += 1
    cat = disc.get(k)
    if cat in c: c[cat] += 1

print("bet-type x ablation — did removing the value subspace reduce the bet's influence?")
print(f"{'condition':10s} {'rank':>4s} | {'committed':>9s} | {'INFLUENCED (stated intent)':>28s} | {'P(est<thr|committed)':>20s} | refusals(NO_STATEMENT)")
for cond in ("donation", "naked"):
    for rank in (0, 16):
        c = cells[(cond, rank)]
        infl = c["INFLUENCED"]/c["n"]; lo, hi = wilson(c["INFLUENCED"], c["n"])
        pb = c["below"]/c["have_est"] if c["have_est"] else float("nan")
        print(f"{cond:10s} {rank:>4d} | {c['have_est']:>4d}/{c['n']:<4d} | "
              f"{infl:.2f} [{lo:.2f},{hi:.2f}] ({c['INFLUENCED']:>2d}/{c['n']}) | "
              f"{pb:>18.2f} | {c['NO_STATEMENT']}")

print("\nDelta (rank0 -> rank16), per bet type:")
for cond in ("donation", "naked"):
    c0, c16 = cells[(cond,0)], cells[(cond,16)]
    i0, i16 = c0["INFLUENCED"]/c0["n"], c16["INFLUENCED"]/c16["n"]
    e0 = c0["have_est"]/c0["n"]; e16 = c16["have_est"]/c16["n"]
    print(f"  {cond:10s}: INFLUENCED {i0:.2f} -> {i16:.2f} (Δ {i16-i0:+.2f}); "
          f"committed-estimate {e0:.2f} -> {e16:.2f} (Δ {e16-e0:+.2f})")

json.dump({f"{k[0]}_r{k[1]}": dict(v) for k, v in cells.items()},
          open("judging/session_c/e9_final.json", "w"), indent=2)
print("\nwrote judging/session_c/e9_final.json")
