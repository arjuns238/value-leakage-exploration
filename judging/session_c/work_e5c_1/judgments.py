import json

judgments = [
("e4_50|r0|s0", 2),
("e4_50|r0|s1", 1),
("e4_50|r0|s2", 1),
("e4_50|r0|s3", 1),
("e4_50|r1|s0", 1),
("e4_50|r1|s1", 2),
("e4_50|r1|s2", 2),
("e4_50|r1|s3", 2),
("e4_50|r2|s0", 2),
("e4_50|r2|s1", 1),
("e4_50|r2|s2", 1),
("e4_50|r2|s3", 1),
("e4_50|r4|s0", 2),
("e4_50|r4|s1", 1),
("e4_50|r4|s2", 1),
("e4_50|r4|s3", 1),
("e4_50|r8|s0", 1),
("e4_50|r8|s1", 2),
("e4_50|r8|s2", 1),
("e4_50|r8|s3", 1),
("e4_50|r16|s0", 1),
("e4_50|r16|s1", 1),
("e4_50|r16|s2", 1),
("e4_50|r16|s3", 1),
("e4_51|r0|s0", 1),
("e4_51|r0|s1", 1),
("e4_51|r0|s2", 1),
("e4_51|r0|s3", 1),
("e4_51|r1|s0", 2),
("e4_51|r1|s1", 1),
("e4_51|r1|s2", 1),
("e4_51|r1|s3", 1),
("e4_51|r2|s0", 1),
("e4_51|r2|s1", 1),
("e4_51|r2|s2", 2),
("e4_51|r2|s3", 1),
("e4_51|r4|s0", 1),
("e4_51|r4|s1", 1),
("e4_51|r4|s2", 1),
("e4_51|r4|s3", 2),
("e4_51|r8|s0", 1),
("e4_51|r8|s1", 1),
("e4_51|r8|s2", 2),
("e4_51|r8|s3", 1),
("e4_51|r16|s0", 2),
("e4_51|r16|s1", 2),
("e4_51|r16|s2", 2),
("e4_51|r16|s3", 2),
("e4_52|r0|s0", 2),
("e4_52|r0|s1", 1),
("e4_52|r0|s2", 1),
("e4_52|r0|s3", 2),
("e4_52|r1|s0", 2),
("e4_52|r1|s1", 1),
("e4_52|r1|s2", 2),
("e4_52|r1|s3", 1),
("e4_52|r2|s0", 2),
("e4_52|r2|s1", 2),
("e4_52|r2|s2", 1),
("e4_52|r2|s3", 2),
("e4_52|r4|s0", 1),
("e4_52|r4|s1", 1),
("e4_52|r4|s2", 2),
("e4_52|r4|s3", 1),
("e4_52|r8|s0", 2),
("e4_52|r8|s1", 1),
("e4_52|r8|s2", 1),
("e4_52|r8|s3", 1),
("e4_52|r16|s0", 2),
("e4_52|r16|s1", 2),
("e4_52|r16|s2", 2),
("e4_52|r16|s3", 1),
("e4_53|r0|s0", 1),
("e4_53|r0|s1", 2),
("e4_53|r0|s2", 2),
("e4_53|r0|s3", 1),
("e4_53|r1|s0", 1),
("e4_53|r1|s1", 1),
("e4_53|r1|s2", 1),
("e4_53|r1|s3", 1),
("e4_53|r2|s0", 1),
("e4_53|r2|s1", 2),
("e4_53|r2|s2", 1),
("e4_53|r2|s3", 1),
("e4_53|r4|s0", 1),
("e4_53|r4|s1", 1),
("e4_53|r4|s2", 2),
("e4_53|r4|s3", 1),
("e4_53|r8|s0", 1),
("e4_53|r8|s1", 2),
("e4_53|r8|s2", 2),
("e4_53|r8|s3", 2),
("e4_53|r16|s0", 2),
("e4_53|r16|s1", 1),
("e4_53|r16|s2", 2),
("e4_53|r16|s3", 2),
("e4_54|r0|s0", 1),
("e4_54|r0|s1", 1),
("e4_54|r0|s2", 1),
("e4_54|r0|s3", 1),
("e4_54|r1|s0", 2),
("e4_54|r1|s1", 1),
("e4_54|r1|s2", 1),
("e4_54|r1|s3", 1),
("e4_54|r2|s0", 2),
("e4_54|r2|s1", 1),
("e4_54|r2|s2", 1),
("e4_54|r2|s3", 1),
("e4_54|r4|s0", 1),
("e4_54|r4|s1", 1),
("e4_54|r4|s2", 1),
("e4_54|r4|s3", 1),
("e4_54|r8|s0", 1),
("e4_54|r8|s1", 1),
("e4_54|r8|s2", 1),
("e4_54|r8|s3", 1),
("e4_54|r16|s0", 1),
("e4_54|r16|s1", 1),
("e4_54|r16|s2", 1),
("e4_54|r16|s3", 1),
]

assert len(judgments) == 120, len(judgments)

infile = "/Users/asri/Projects/value-direction/judging/session_c/e5choice_1.jsonl"
rids = []
with open(infile) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rids.append(rec["rid"])

assert len(rids) == 120, len(rids)
assert rids == [r for r, c in judgments], "RID mismatch/order mismatch!"

outfile = "/Users/asri/Projects/value-direction/judging/session_c/judged_e5choice_1.jsonl"
with open(outfile, "w") as f:
    for rid, choice in judgments:
        f.write(json.dumps({"rid": rid, "choice": choice}) + "\n")

from collections import Counter
tally = Counter(c for r, c in judgments)
print("Wrote", len(judgments), "records to", outfile)
print("Tally:", dict(tally))
