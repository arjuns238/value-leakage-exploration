import json

judgments = {
"e4:e4_13:-0.06:0": 2,
"e4:e4_90:-0.06:0": 2,
"e4:e4_64:0.0:1": 1,
"e4:e4_13:0.06:0": 1,
"e4:e4_14:-0.06:0": 2,
"e4:e4_27:-0.03:1": 2,
"e4:e4_27:0.0:0": 2,
"e4:e4_78:0.03:0": 1,
"e4:e4_66:-0.06:0": 2,
"e4:e4_2:0.0:1": 2,
"e4:e4_42:-0.06:1": 2,
"e4:e4_3:0.06:1": 1,
"e4:e4_29:0.03:0": 1,
"e4:e4_94:0.03:1": 1,
"e4:e4_57:-0.03:0": 2,
"e4:e4_31:0.06:0": 1,
"e4:e4_19:-0.06:1": 2,
"e4:e4_57:0.03:1": 1,
"e4:e4_70:0.03:1": 1,
"e4:e4_45:-0.03:0": 2,
"e4:e4_70:0.06:0": 1,
"e4:e4_96:0.03:0": "refusal",
"e4:e4_96:0.06:1": 1,
"e4:e4_97:-0.06:0": 2,
"e4:e4_97:0.03:1": 1,
"e4:e4_97:0.06:0": 1,
"e4:e4_61:-0.03:1": 2,
"e4:e4_74:0.03:1": 1,
"e4:e4_74:0.06:0": 1,
"e4:e4_11:-0.06:0": 2,
"e4:e4_11:-0.03:0": 2,
"e4:e4_24:0.0:0": 1,
"e4:e4_24:0.03:0": 1,
"e4:e4_88:0.06:0": 1,
"e4:e4_24:0.06:1": 1,
"e4:e4_51:-0.06:0": 2,
"e4:e4_76:0.03:1": 1,
}

in_path = "/Users/asri/Projects/value-direction/judging/chunk_e4choice.jsonl"
out_path = "/Users/asri/Projects/value-direction/judging/judged_e4choice.jsonl"

seen = []
with open(in_path) as f, open(out_path, "w") as out:
    for line in f:
        rec = json.loads(line)
        rid = rec["rid"]
        if rid not in judgments:
            raise SystemExit(f"MISSING judgment for {rid}")
        choice = judgments[rid]
        out.write(json.dumps({"rid": rid, "choice": choice}) + "\n")
        seen.append(rid)

print("total lines processed:", len(seen))
print("unique rids used:", len(set(seen)))
print("judgments dict size:", len(judgments))
