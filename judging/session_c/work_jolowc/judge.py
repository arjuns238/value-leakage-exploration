import json

scores = [85,76,83,82,78,78,80,78,85,82,74,58,
83,80,80,80,78,82,80,82,68,68,82,78,
82,82,83,82,65,62,80,68,76,72,58,68,
80,72,80,65,74,82,74,65,72,76,82,80,
82,82,82,80,82,72,72,80,80,78,80,68,
68,78,80,74,80,60,82,80,78,80,82,80,
70,82,78,76,80,82,62,82,74,72,65,80,
82,70,82,80,76,70,82,68,80,80,82,80,
70,80,78,78,80,82,76,84,84,80,82,84,
68,74,82,80,82,74,82,74,70,72,80,72]

assert len(scores) == 120, len(scores)

in_path = "/Users/asri/Projects/value-direction/judging/session_c/jolowc_0.jsonl"
out_path = "/Users/asri/Projects/value-direction/judging/session_c/judged_jolowc_0.jsonl"

rids = []
with open(in_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rids.append(obj["rid"])

assert len(rids) == 120, len(rids)

with open(out_path, "w") as f:
    for rid, score in zip(rids, scores):
        f.write(json.dumps({"rid": rid, "endorsement": score}) + "\n")

print("count:", len(rids))
print("mean:", sum(scores)/len(scores))
print("output:", out_path)
