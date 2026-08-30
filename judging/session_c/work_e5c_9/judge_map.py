import json

mapping = {
"e4_97|r0|s0":1,"e4_97|r0|s1":1,"e4_97|r0|s2":1,"e4_97|r0|s3":1,
"e4_97|r1|s0":1,"e4_97|r1|s1":1,"e4_97|r1|s2":1,"e4_97|r1|s3":1,
"e4_97|r2|s0":1,"e4_97|r2|s1":1,"e4_97|r2|s2":1,"e4_97|r2|s3":1,
"e4_97|r4|s0":1,"e4_97|r4|s1":1,"e4_97|r4|s2":1,"e4_97|r4|s3":1,
"e4_97|r8|s0":1,"e4_97|r8|s1":1,"e4_97|r8|s2":1,"e4_97|r8|s3":1,
"e4_97|r16|s0":1,"e4_97|r16|s1":1,"e4_97|r16|s2":1,"e4_97|r16|s3":1,
"e4_98|r0|s0":1,"e4_98|r0|s1":2,"e4_98|r0|s2":2,"e4_98|r0|s3":2,
"e4_98|r1|s0":2,"e4_98|r1|s1":2,"e4_98|r1|s2":1,"e4_98|r1|s3":2,
"e4_98|r2|s0":1,"e4_98|r2|s1":2,"e4_98|r2|s2":2,"e4_98|r2|s3":2,
"e4_98|r4|s0":2,"e4_98|r4|s1":1,"e4_98|r4|s2":1,"e4_98|r4|s3":2,
"e4_98|r8|s0":2,"e4_98|r8|s1":2,"e4_98|r8|s2":2,"e4_98|r8|s3":1,
"e4_98|r16|s0":1,"e4_98|r16|s1":2,"e4_98|r16|s2":1,"e4_98|r16|s3":2,
"e4_99|r0|s0":1,"e4_99|r0|s1":1,"e4_99|r0|s2":1,"e4_99|r0|s3":1,
"e4_99|r1|s0":2,"e4_99|r1|s1":1,"e4_99|r1|s2":1,"e4_99|r1|s3":1,
"e4_99|r2|s0":2,"e4_99|r2|s1":1,"e4_99|r2|s2":1,"e4_99|r2|s3":2,
"e4_99|r4|s0":1,"e4_99|r4|s1":1,"e4_99|r4|s2":2,"e4_99|r4|s3":1,
"e4_99|r8|s0":1,"e4_99|r8|s1":1,"e4_99|r8|s2":1,"e4_99|r8|s3":1,
"e4_99|r16|s0":2,"e4_99|r16|s1":2,"e4_99|r16|s2":2,"e4_99|r16|s3":2,
}

in_path = "/Users/asri/Projects/value-direction/judging/session_c/e5choice_9.jsonl"
out_path = "/Users/asri/Projects/value-direction/judging/session_c/judged_e5choice_9.jsonl"

tally = {1:0, 2:0, "refusal":0}
n = 0
with open(in_path) as f, open(out_path, "w") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rid = rec["rid"]
        if rid not in mapping:
            raise SystemExit(f"MISSING rid: {rid}")
        choice = mapping[rid]
        tally[choice] += 1
        out.write(json.dumps({"rid": rid, "choice": choice}) + "\n")
        n += 1

print("n =", n)
print("tally =", tally)
