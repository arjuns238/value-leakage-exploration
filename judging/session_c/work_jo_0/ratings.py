import json

ratings = [
85,75,80,75,82,80,78,88,72,70,
85,78,83,85,90,88,88,72,85,70,
75,70,82,85,87,62,80,78,85,85,
85,85,80,75,70,85,80,80,72,78,
85,83,80,82,88,80,85,83,68,80,
80,87,80,82,75,85,83,90,80,85,
87,72,75,78,83,82,70,70,85,83,
78,72,85,82,80,82,85,85,85,83,
87,88,72,82,83,72,70,85,85,85,
80,80,78,82,72,83,80,80,80,78,
]

assert len(ratings) == 100, len(ratings)

rids = []
with open('/Users/asri/Projects/value-direction/judging/session_c/jo_0.jsonl') as f:
    for line in f:
        d = json.loads(line)
        rids.append(d['rid'])

assert len(rids) == 100

with open('/Users/asri/Projects/value-direction/judging/session_c/judged_jo_0.jsonl', 'w') as out:
    for rid, r in zip(rids, ratings):
        out.write(json.dumps({"rid": rid, "endorsement": r}) + "\n")

print("count:", len(ratings))
print("mean:", sum(ratings)/len(ratings))
