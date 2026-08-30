import json

judgments = [
1,"refusal",1,1,"refusal",2,"refusal",2,2,"refusal",
1,1,1,1,1,1,1,1,1,1,
1,"refusal",2,1,"refusal","refusal",1,"refusal","refusal",2,
"refusal",2,"refusal",1,1,"refusal","refusal","refusal","refusal","refusal",
1,1,"refusal",1,1,1,"refusal","refusal",1,"refusal",
"refusal",1,2,2,2,"refusal",1,"refusal","refusal","refusal",
"refusal",2,1,1,1,"refusal",1,1,1,"refusal",
2,2,1,2,"refusal",1,"refusal",2,"refusal","refusal",
1,"refusal",1,1,"refusal",1,1,1,"refusal","refusal",
"refusal",1,"refusal",1,1,"refusal",1,"refusal",2,"refusal",
"refusal",2,1,2,"refusal",1,1,"refusal",1,1,
"refusal",1,1,"refusal",2
]

assert len(judgments) == 115, len(judgments)

rids = []
with open('/Users/asri/Projects/value-direction/judging/session_c/randchoice_4.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rids.append(obj['rid'])

assert len(rids) == 115, len(rids)

out_path = '/Users/asri/Projects/value-direction/judging/session_c/judged_randchoice_4.jsonl'
with open(out_path, 'w') as out:
    for rid, choice in zip(rids, judgments):
        out.write(json.dumps({"rid": rid, "choice": choice}) + "\n")

from collections import Counter
c = Counter(judgments)
print(dict(c))
print("wrote", out_path)
