#!/usr/bin/env python3
"""Merge judged chunk files and validate against the pending file.

Usage: python3 merge_judged.py <pending.jsonl> <out.jsonl> <judged_chunk...>

Checks: every pending prompt_hash judged exactly once, every answer contains a
well-formed <final_estimate> tag. Exits nonzero listing problems otherwise.
"""
import json
import re
import sys

pending_path, out_path, *chunks = sys.argv[1:]
pending = {json.loads(l)["prompt_hash"] for l in open(pending_path)}

rows = {}
bad = []
for chunk in chunks:
    for line in open(chunk):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        h, a = r["prompt_hash"], r["answer"]
        if not re.search(r"<final_estimate>[\d,\.]+|<final_estimate>UNKNOWN</final_estimate>", a):
            bad.append((h, "malformed answer: " + a[:80]))
        if h in rows and rows[h] != a:
            bad.append((h, "conflicting duplicate"))
        rows[h] = r["answer"]

missing = pending - set(rows)
extra = set(rows) - pending
if missing:
    bad.append(("-", f"{len(missing)} pending hashes unjudged: {sorted(missing)[:3]}..."))
if extra:
    bad.append(("-", f"{len(extra)} judged hashes not in pending"))

if bad:
    for h, msg in bad[:10]:
        print(f"PROBLEM {h[:12]}: {msg}", file=sys.stderr)
    sys.exit(1)

with open(out_path, "w") as f:
    for h, a in rows.items():
        f.write(json.dumps({"prompt_hash": h, "answer": a}) + "\n")
unknown = sum("UNKNOWN" in a for a in rows.values())
print(f"OK: {len(rows)} judgements -> {out_path} ({unknown} UNKNOWN, "
      f"{100*(1-unknown/len(rows)):.0f}% parse rate)")
