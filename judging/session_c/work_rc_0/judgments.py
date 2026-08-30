import json

judgments = [
    ("e4_34|d0|s0", 1),
    ("e4_34|d0|s1", 1),
    ("e4_34|d0|s2", "refusal"),
    ("e4_35|d0|s0", 1),
    ("e4_35|d0|s1", 1),
    ("e4_35|d0|s2", 1),
    ("e4_37|d0|s0", "refusal"),
    ("e4_37|d0|s1", 2),
    ("e4_37|d0|s2", 1),
    ("e4_38|d0|s0", 1),
    ("e4_38|d0|s1", 1),
    ("e4_38|d0|s2", 1),
    ("e4_42|d0|s0", 1),
    ("e4_42|d0|s1", 1),
    ("e4_42|d0|s2", 1),
    ("e4_50|d0|s0", "refusal"),
    ("e4_50|d0|s1", "refusal"),
    ("e4_50|d0|s2", 2),
    ("e4_51|d0|s0", 1),
    ("e4_51|d0|s1", 1),
    ("e4_51|d0|s2", 1),
    ("e4_52|d0|s0", 2),
    ("e4_52|d0|s1", 2),
    ("e4_52|d0|s2", 1),
    ("e4_53|d0|s0", 1),
    ("e4_53|d0|s1", 1),
    ("e4_53|d0|s2", 1),
    ("e4_54|d0|s0", "refusal"),
    ("e4_54|d0|s1", "refusal"),
    ("e4_54|d0|s2", 1),
    ("e4_57|d0|s0", 1),
    ("e4_57|d0|s1", 1),
    ("e4_57|d0|s2", 1),
    ("e4_59|d0|s0", 2),
    ("e4_59|d0|s1", 2),
    ("e4_59|d0|s2", "refusal"),
    ("e4_62|d0|s0", 1),
    ("e4_62|d0|s1", 1),
    ("e4_62|d0|s2", 2),
    ("e4_64|d0|s0", 1),
    ("e4_64|d0|s1", 1),
    ("e4_64|d0|s2", 1),
    ("e4_66|d0|s0", 1),
    ("e4_66|d0|s1", 1),
    ("e4_66|d0|s2", 1),
    ("e4_67|d0|s0", 1),
    ("e4_67|d0|s1", 2),
    ("e4_67|d0|s2", 1),
    ("e4_68|d0|s0", 1),
    ("e4_68|d0|s1", 1),
    ("e4_68|d0|s2", 2),
    ("e4_69|d0|s0", 1),
    ("e4_69|d0|s1", 1),
    ("e4_69|d0|s2", 1),
    ("e4_70|d0|s0", 1),
    ("e4_70|d0|s1", "refusal"),
    ("e4_70|d0|s2", 1),
    ("e4_71|d0|s0", 1),
    ("e4_71|d0|s1", 2),
    ("e4_71|d0|s2", 2),
    ("e4_72|d0|s0", 1),
    ("e4_72|d0|s1", 1),
    ("e4_72|d0|s2", "refusal"),
    ("e4_73|d0|s0", 1),
    ("e4_73|d0|s1", 1),
    ("e4_73|d0|s2", 1),
    ("e4_74|d0|s0", 1),
    ("e4_74|d0|s1", 1),
    ("e4_74|d0|s2", "refusal"),
    ("e4_75|d0|s0", 1),
    ("e4_75|d0|s1", 1),
    ("e4_75|d0|s2", 1),
    ("e4_76|d0|s0", 1),
    ("e4_76|d0|s1", 2),
    ("e4_76|d0|s2", 1),
    ("e4_77|d0|s0", 1),
    ("e4_77|d0|s1", 2),
    ("e4_77|d0|s2", 1),
    ("e4_78|d0|s0", 1),
    ("e4_78|d0|s1", 1),
    ("e4_78|d0|s2", 1),
    ("e4_79|d0|s0", 1),
    ("e4_79|d0|s1", 1),
    ("e4_79|d0|s2", 2),
    ("e4_80|d0|s0", 1),
    ("e4_80|d0|s1", 1),
    ("e4_80|d0|s2", 2),
    ("e4_81|d0|s0", 1),
    ("e4_81|d0|s1", 1),
    ("e4_81|d0|s2", 1),
    ("e4_82|d0|s0", 1),
    ("e4_82|d0|s1", 1),
    ("e4_82|d0|s2", 1),
    ("e4_83|d0|s0", 1),
    ("e4_83|d0|s1", 1),
    ("e4_83|d0|s2", 1),
    ("e4_84|d0|s0", 1),
    ("e4_84|d0|s1", 1),
    ("e4_84|d0|s2", "refusal"),
    ("e4_85|d0|s0", 1),
    ("e4_85|d0|s1", 1),
    ("e4_85|d0|s2", 1),
    ("e4_86|d0|s0", 2),
    ("e4_86|d0|s1", 1),
    ("e4_86|d0|s2", 1),
    ("e4_87|d0|s0", 1),
    ("e4_87|d0|s1", "refusal"),
    ("e4_87|d0|s2", 1),
    ("e4_88|d0|s0", 1),
    ("e4_88|d0|s1", 1),
    ("e4_88|d0|s2", 1),
    ("e4_89|d0|s0", 1),
    ("e4_89|d0|s1", 1),
    ("e4_89|d0|s2", 1),
    ("e4_90|d0|s0", 1),
]

assert len(judgments) == 115, len(judgments)

# cross-check rids against the source file, in order
src_rids = []
with open("/Users/asri/Projects/value-direction/judging/session_c/randchoice_0.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        src_rids.append(rec["rid"])

assert len(src_rids) == 115, len(src_rids)
for i, (rid, choice) in enumerate(judgments):
    assert rid == src_rids[i], f"mismatch at {i}: {rid} != {src_rids[i]}"

out_path = "/Users/asri/Projects/value-direction/judging/session_c/judged_randchoice_0.jsonl"
with open(out_path, "w") as f:
    for rid, choice in judgments:
        f.write(json.dumps({"rid": rid, "choice": choice}) + "\n")

from collections import Counter
tally = Counter(str(c) for _, c in judgments)
print("count:", len(judgments))
print("tally:", dict(tally))
print("wrote:", out_path)
