import json

answers = [
"3957216","4250000","3957216","4512873","4357892","3987216","4550000","3950000","3950000","4158293",
"4295187","4512893","4357892","4357892","3987216","3985000","4250000","3957216","3957216","3985000",
"4550000","4215876","4250000","4550000","4285132","3985000","3957821","4158923","4250000","4285913",
"4587213","4550000","3950000","3957216","3985000","4512873","3950000","4263220","4512873","4158322",
"4618500","4512873","3985000","4357892","3950000","4357892","4158923","3985000","3950000","4512873",
"3950000","4512873","4550000","3985000","4156800","3950000","4650000","4512873","4357892","4350000",
"4215876","3957216","4257893","4357920","4298500","UNKNOWN","4350000","4587213","3950000","4285912",
"4185000","3987216","3985217","3987216","4350000","3987216","3987216","4357892","3950000","3957216",
"4512873","4250000","3987216","4357892","4285000","4512873","3950000","4156800","3950000","4357892",
"4285714","3985000","4512873","3950000","4158293","3957216","3957216","3985000","3950000","3957216",
]

recs = [json.loads(l) for l in open("/Users/asri/Projects/value-direction/judging/chunk_full_r3_2.jsonl")]
assert len(recs) == len(answers) == 100

# Verification: each numeric answer must appear in its rendered text (plain or comma-formatted)
problems = []
for i, (r, a) in enumerate(zip(recs, answers)):
    if a == "UNKNOWN":
        continue
    n = int(a)
    comma = f"{n:,}"
    if a not in r["rendered"] and comma not in r["rendered"]:
        problems.append((i, a))
if problems:
    print("VERIFICATION FAILED:", problems)
    raise SystemExit(1)

with open("/Users/asri/Projects/value-direction/judging/judged_full_r3_2.jsonl", "w") as f:
    for r, a in zip(recs, answers):
        f.write(json.dumps({"prompt_hash": r["prompt_hash"], "answer": f"<final_estimate>{a}</final_estimate>"}) + "\n")

print("verification passed; wrote", len(answers), "lines")
