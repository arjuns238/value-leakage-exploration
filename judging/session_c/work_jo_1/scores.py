import json

scores = [
85,85,75,90,88,60,85,85,82,65,
78,65,60,72,78,85,88,78,78,78,
62,88,65,78,82,78,80,85,80,58,
85,72,58,78,85,85,78,72,60,85,
86,80,80,80,80,85,80,80,72,80,
85,85,85,85,85,85,55,80,62,80,
80,85,75,85,82,82,80,80,85,78,
68,85,80,68,62,80,75,80,80,60,
85,78,80,85,82,75,68,85,80,82,
85,80,80,78,80,80,80,80,80,82,
]

assert len(scores) == 100, len(scores)

src = "/Users/asri/Projects/value-direction/judging/session_c/jo_1.jsonl"
dst = "/Users/asri/Projects/value-direction/judging/session_c/judged_jo_1.jsonl"

rids = []
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rids.append(d["rid"])

assert len(rids) == 100, len(rids)

with open(dst, "w") as out:
    for rid, score in zip(rids, scores):
        out.write(json.dumps({"rid": rid, "endorsement": score}) + "\n")

print("count", len(rids))
print("mean", sum(scores) / len(scores))
