import json

judgments = {
    "prefers:r548": 1, "picks:p312": 2, "picks:p6854": 1, "picks:p2193": 2,
    "picks:p3050": 2, "picks:p6448": 2, "prefers:r1154": 2, "picks:p7396": 1,
    "picks:p6031": 2, "prefers:r2633": 1, "picks:p3927": 2, "prefers:r1886": 1,
    "picks:p2282": 1, "picks:p8571": 2, "picks:p4740": 2, "prefers:r1391": 1,
    "prefers:r2039": 1, "picks:p4325": 1, "picks:p3963": 2, "picks:p9094": 1,
    "picks:p1084": 2, "picks:p4454": 1, "picks:p3740": 2, "prefers:r122": 2,
    "picks:p9050": 2, "picks:p9143": 2, "prefers:r2855": 2, "picks:p4812": 2,
    "picks:p8365": 2, "prefers:r734": 2, "prefers:r150": 2, "picks:p9313": 1,
    "picks:p2576": 2, "picks:p4916": 2, "prefers:r910": 1, "picks:p2376": 2,
    "picks:p979": 1, "prefers:r2915": 2, "picks:p6142": 1, "prefers:r561": 1,
    "picks:p8295": 2, "picks:p5810": 2, "prefers:r1133": 2, "picks:p8151": 2,
    "picks:p9814": 2, "prefers:r819": 2, "picks:p5161": 2, "picks:p1541": 2,
    "picks:p7114": 2, "picks:p209": 2, "prefers:r2588": 2, "prefers:r118": 1,
    "prefers:r2543": 2, "picks:p6536": 2, "prefers:r1215": 1, "picks:p2042": 2,
    "picks:p9432": 2, "picks:p9363": 1, "picks:p1002": 1, "picks:p4694": 1,
    "picks:p7043": 2, "prefers:r1352": 2, "picks:p6149": 1, "prefers:r607": 1,
    "picks:p961": 1, "picks:p9148": 2, "picks:p696": 1, "picks:p3515": 1,
    "prefers:r52": 1, "picks:p1138": 2, "picks:p2707": 1, "prefers:r1856": 2,
    "picks:p6816": 1, "picks:p3040": 2, "prefers:r4": 2, "picks:p3082": 2,
    "picks:p2563": 1, "picks:p7210": 1, "prefers:r2910": 1, "prefers:r997": 2,
    "picks:p4900": 1, "picks:p2366": 1, "prefers:r2730": 2, "prefers:r1069": 2,
    "picks:p1886": 2, "picks:p1668": 1, "prefers:r2038": 2, "prefers:r420": 1,
    "prefers:r20": 1, "picks:p3682": 2, "picks:p2104": 2, "prefers:r2099": 2,
    "picks:p2190": 2, "picks:p878": 2, "picks:p8126": 2, "prefers:r951": 1,
    "picks:p6641": 2, "picks:p6101": 2, "picks:p7059": 2, "prefers:r1157": 1,
}

inp = "/Users/asri/Projects/value-direction/judging/chunk_choice_19.jsonl"
out = "/Users/asri/Projects/value-direction/judging/judged_choice_19.jsonl"

with open(inp) as f:
    rids = [json.loads(l)["rid"] for l in f]

assert len(rids) == 100
assert set(rids) == set(judgments), (set(rids) ^ set(judgments))

with open(out, "w") as f:
    for rid in rids:
        f.write(json.dumps({"rid": rid, "choice": judgments[rid]}) + "\n")

from collections import Counter
print(Counter(judgments[r] for r in rids))
print("wrote", len(rids), "lines to", out)
