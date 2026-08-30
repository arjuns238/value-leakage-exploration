"""Merge random-ablate choice judges; compute per-draw + pooled P(picked preferred).
Specificity control for E5: value rank-16 = 0.49, no-ablation = 0.63.
If random-16 ~0.63 -> value-specific (necessity holds). If ~0.49 -> capacity damage."""
import json, glob, math
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h, c+h)

gen={f'{r["pair_id"]}|d{r["draw"]}|s{r["sample"]}': r
     for r in (json.loads(l) for l in open("artifacts/session_c/random_ablate.jsonl"))}
choice={}
for fn in glob.glob("judging/session_c/judged_randchoice_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j=json.loads(l); choice[j["rid"]]=j["choice"]
print(f"random-ablate: {len(choice)}/{len(gen)} judged")

# activity_1 = preferred (higher-scored) in every pair
by_draw=defaultdict(lambda:{"pref":0,"other":0,"refusal":0})
tot={"pref":0,"other":0,"refusal":0}
for rid,ch in choice.items():
    d=by_draw[gen[rid]["draw"]]
    key="pref" if ch==1 else ("other" if ch==2 else "refusal")
    d[key]+=1; tot[key]+=1

print("\nRANDOM rank-16 ablation — P(picked preferred) per draw")
print("draw | n_dec | P(pref) | 95% CI | refusals")
for k in sorted(by_draw):
    d=by_draw[k]; dec=d["pref"]+d["other"]; p=d["pref"]/dec if dec else float("nan")
    lo,hi=wilson(d["pref"],dec)
    print(f"  {k}  | {dec:3d}   | {p:.3f}  | [{lo:.3f},{hi:.3f}] | {d['refusal']}")
dec=tot["pref"]+tot["other"]; P=tot["pref"]/dec; lo,hi=wilson(tot["pref"],dec)
print(f"\nPOOLED random-16: P(pref) = {P:.3f} [{lo:.3f},{hi:.3f}] (n={dec} decisive, {tot['refusal']} refusals/degenerate)")
print("\n--- verdict table ---")
print(f"  no ablation (rank 0)   : 0.630")
print(f"  VALUE  rank-16 (E5)    : 0.490   (leak killed)")
print(f"  RANDOM rank-16 (this)  : {P:.3f}   [{lo:.3f},{hi:.3f}]")
print("  -> near 0.63 = value-specific (necessity holds); near 0.49 = capacity damage")
json.dump({"pooled":{"P":P,"ci":[lo,hi],"n":dec,"refusal":tot["refusal"]},
           "by_draw":{str(k):dict(v) for k,v in by_draw.items()}},
          open("judging/session_c/random_ablate_final.json","w"),indent=2)
print("\nwrote judging/session_c/random_ablate_final.json")
