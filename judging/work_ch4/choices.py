import json

choices = [1,2,1,2,1,2,2,2,1,1,
           1,1,1,2,1,1,1,1,2,2,
           2,"refusal",2,2,1,1,2,2,2,1,
           2,2,2,2,2,2,2,2,2,2,
           2,1,2,1,2,1,2,1,2,1,
           1,2,2,2,2,2,1,2,2,1,
           1,2,2,2,2,2,1,2,1,2,
           1,1,1,2,2,2,2,1,2,1,
           2,2,1,2,1,1,2,2,1,1,
           1,2,2,2,1,2,2,2,1,1]

rids=[]
with open('../chunk_choice_4.jsonl') as f:
    for line in f:
        rids.append(json.loads(line)['rid'])
assert len(rids)==len(choices)==100

with open('../judged_choice_4.jsonl','w') as out:
    for rid,c in zip(rids,choices):
        out.write(json.dumps({"rid": rid, "choice": c})+'\n')

from collections import Counter
print(Counter(map(str,choices)))
