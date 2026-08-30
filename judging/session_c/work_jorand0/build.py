import json

rids = [
 "joFULL|Google->Meta|c-0.06|s0","joFULL|Google->Meta|c-0.06|s1","joFULL|Google->Meta|c-0.06|s2",
 "joFULL|Google->Meta|c-0.06|s3","joFULL|Google->Meta|c-0.06|s4","joFULL|Google->Meta|c-0.06|s5",
 "joFULL|Google->Meta|c-0.03|s0","joFULL|Google->Meta|c-0.03|s1","joFULL|Google->Meta|c-0.03|s2",
 "joFULL|Google->Meta|c-0.03|s3","joFULL|Google->Meta|c-0.03|s4","joFULL|Google->Meta|c-0.03|s5",
 "joFULL|Google->Meta|c0.0|s0","joFULL|Google->Meta|c0.0|s1","joFULL|Google->Meta|c0.0|s2",
 "joFULL|Google->Meta|c0.0|s3","joFULL|Google->Meta|c0.0|s4","joFULL|Google->Meta|c0.0|s5",
 "joFULL|Google->Meta|c0.03|s0","joFULL|Google->Meta|c0.03|s1","joFULL|Google->Meta|c0.03|s2",
 "joFULL|Google->Meta|c0.03|s3","joFULL|Google->Meta|c0.03|s4","joFULL|Google->Meta|c0.03|s5",
 "joFULL|Google->Meta|c0.06|s0","joFULL|Google->Meta|c0.06|s1","joFULL|Google->Meta|c0.06|s2",
 "joFULL|Google->Meta|c0.06|s3","joFULL|Google->Meta|c0.06|s4","joFULL|Google->Meta|c0.06|s5",
 "joFULL|Meta->Google|c-0.06|s0","joFULL|Meta->Google|c-0.06|s1","joFULL|Meta->Google|c-0.06|s2",
 "joFULL|Meta->Google|c-0.06|s3","joFULL|Meta->Google|c-0.06|s4","joFULL|Meta->Google|c-0.06|s5",
 "joFULL|Meta->Google|c-0.03|s0","joFULL|Meta->Google|c-0.03|s1","joFULL|Meta->Google|c-0.03|s2",
 "joFULL|Meta->Google|c-0.03|s3","joFULL|Meta->Google|c-0.03|s4","joFULL|Meta->Google|c-0.03|s5",
 "joFULL|Meta->Google|c0.0|s0","joFULL|Meta->Google|c0.0|s1","joFULL|Meta->Google|c0.0|s2",
 "joFULL|Meta->Google|c0.0|s3","joFULL|Meta->Google|c0.0|s4","joFULL|Meta->Google|c0.0|s5",
 "joFULL|Meta->Google|c0.03|s0","joFULL|Meta->Google|c0.03|s1","joFULL|Meta->Google|c0.03|s2",
 "joFULL|Meta->Google|c0.03|s3","joFULL|Meta->Google|c0.03|s4","joFULL|Meta->Google|c0.03|s5",
 "joFULL|Meta->Google|c0.06|s0","joFULL|Meta->Google|c0.06|s1","joFULL|Meta->Google|c0.06|s2",
 "joFULL|Meta->Google|c0.06|s3","joFULL|Meta->Google|c0.06|s4","joFULL|Meta->Google|c0.06|s5",
]

scores = [
68,75,70,62,70,62,
65,68,68,70,72,72,
70,73,72,78,65,80,
72,73,73,68,70,75,
68,72,65,73,70,73,
75,73,76,73,75,73,
75,65,73,74,75,72,
72,72,58,68,72,75,
68,72,72,60,74,75,
68,70,60,65,76,68,
]

assert len(rids) == 60 == len(scores)

src_path = "/Users/asri/Projects/value-direction/judging/session_c/jorand0_0.jsonl"
out_path = "/Users/asri/Projects/value-direction/judging/session_c/judged_jorand0_0.jsonl"

# verify rid order matches source file exactly
src_rids = []
with open(src_path) as f:
    for line in f:
        obj = json.loads(line)
        src_rids.append(obj["rid"])

assert src_rids == rids, "rid order mismatch!"

with open(out_path, "w") as f:
    for rid, score in zip(rids, scores):
        f.write(json.dumps({"rid": rid, "endorsement": score}) + "\n")

print("count:", len(scores))
print("mean:", sum(scores)/len(scores))
print("wrote:", out_path)
