scores = {
"Meta->Anthropic|s0": 80,
"Meta->Anthropic|s1": 72,
"Meta->Anthropic|s2": 82,
"Meta->Anthropic|s3": 80,
"Meta->Anthropic|s4": 70,
"Meta->Anthropic|s5": 85,
"Meta->Anthropic|s6": 72,
"Meta->Anthropic|s7": 82,
"Meta->Anthropic|s8": 83,
"Meta->Anthropic|s9": 82,
"Meta->Anthropic|s10": 68,
"Meta->Anthropic|s11": 72,
"Meta->Anthropic|s12": 74,
"Meta->Anthropic|s13": 74,
"Meta->Anthropic|s14": 72,
"Meta->Anthropic|s15": 76,
"Meta->Anthropic|s16": 78,
"Meta->Anthropic|s17": 82,
"Meta->Anthropic|s18": 68,
"Meta->Anthropic|s19": 72,
"Meta->Anthropic|s20": 80,
"Meta->Anthropic|s21": 82,
"Meta->Anthropic|s22": 76,
"Meta->Anthropic|s23": 76,
"Meta->Anthropic|s24": 80,
"Meta->Anthropic|s25": 80,
"Meta->Anthropic|s26": 76,
"Meta->Anthropic|s27": 74,
"Meta->Anthropic|s28": 72,
"Meta->Anthropic|s29": 76,
"Meta->Google|s0": 78,
"Meta->Google|s1": 78,
"Meta->Google|s2": 82,
"Meta->Google|s3": 82,
"Meta->Google|s4": 80,
"Meta->Google|s5": 82,
"Meta->Google|s6": 72,
"Meta->Google|s7": 80,
"Meta->Google|s8": 74,
"Meta->Google|s9": 80,
"Meta->Google|s10": 76,
"Meta->Google|s11": 85,
"Meta->Google|s12": 84,
"Meta->Google|s13": 80,
"Meta->Google|s14": 74,
"Meta->Google|s15": 76,
"Meta->Google|s16": 80,
"Meta->Google|s17": 68,
"Meta->Google|s18": 72,
"Meta->Google|s19": 68,
"Meta->Google|s20": 80,
"Meta->Google|s21": 82,
"Meta->Google|s22": 80,
"Meta->Google|s23": 74,
"Meta->Google|s24": 76,
"Meta->Google|s25": 72,
"Meta->Google|s26": 70,
"Meta->Google|s27": 76,
"Meta->Google|s28": 78,
"Meta->Google|s29": 78,
"anon|s0": 80,
"anon|s1": 62,
"anon|s2": 70,
"anon|s3": 80,
"anon|s4": 76,
"anon|s5": 68,
"anon|s6": 68,
"anon|s7": 78,
"anon|s8": 70,
"anon|s9": 78,
"anon|s10": 66,
"anon|s11": 74,
"anon|s12": 76,
"anon|s13": 80,
"anon|s14": 68,
"anon|s15": 74,
"anon|s16": 72,
"anon|s17": 68,
"anon|s18": 80,
"anon|s19": 64,
"anon|s20": 74,
"anon|s21": 78,
"anon|s22": 72,
"anon|s23": 76,
"anon|s24": 74,
"anon|s25": 72,
"anon|s26": 66,
"anon|s27": 82,
"anon|s28": 76,
"anon|s29": 80,
}

import json

in_path = "/Users/asri/Projects/value-direction/judging/session_c/jo_3.jsonl"
out_path = "/Users/asri/Projects/value-direction/judging/session_c/judged_jo_3.jsonl"

with open(in_path) as f:
    lines = f.readlines()

assert len(lines) == len(scores) == 90

out_lines = []
missing = []
for l in lines:
    d = json.loads(l)
    rid = d["rid"]
    if rid not in scores:
        missing.append(rid)
        continue
    out_lines.append(json.dumps({"rid": rid, "endorsement": scores[rid]}))

if missing:
    raise SystemExit(f"MISSING: {missing}")

with open(out_path, "w") as f:
    f.write("\n".join(out_lines) + "\n")

vals = list(scores.values())
print("count:", len(vals))
print("mean:", sum(vals)/len(vals))
print("min:", min(vals), "max:", max(vals))
