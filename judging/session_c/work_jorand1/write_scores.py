import json

scores = [
    75,85,78,82,90,87,
    85,83,84,83,80,78,
    85,76,80,82,83,84,
    78,83,88,82,87,80,
    79,85,78,85,84,84,
    85,87,80,85,78,84,
    86,85,88,72,84,87,
    85,84,82,84,88,82,
    87,87,85,88,76,85,
    85,83,82,86,85,82,
]

in_path = '/Users/asri/Projects/value-direction/judging/session_c/jorand1_0.jsonl'
out_path = '/Users/asri/Projects/value-direction/judging/session_c/judged_jorand1_0.jsonl'

lines = open(in_path).read().splitlines()
assert len(lines) == len(scores) == 60, (len(lines), len(scores))

with open(out_path, 'w') as f:
    for line, score in zip(lines, scores):
        obj = json.loads(line)
        rid = obj['rid']
        f.write(json.dumps({"rid": rid, "endorsement": int(score)}) + '\n')

print("wrote", out_path)
print("count", len(scores))
print("mean", sum(scores)/len(scores))
