"""Browser for the steering-generalization result: per topic, the model's
'which argument is better reasoned' response across steering coefficients."""
import json, re, html
from collections import defaultdict

rows = [json.loads(l) for l in open("artifacts/session_c/steer_judge.jsonl")]
topics = {t["id"]: t for t in json.load(open("session_b/stance_topics.json"))}
COEFFS = sorted(set(r["c"] for r in rows))

def verdict(t):
    m = re.search(r"Better:\s*\**\s*([12])", t)
    return m.group(1) if m else "?"

# one sample per (topic, c)
cell = {}
for r in rows:
    cell.setdefault((r["id"], r["c"]), r)

def esc(s): return html.escape(s)

parts = ["""<meta charset=utf-8><style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}
h1{padding:14px 20px;margin:0;background:#1a1a1a;color:#fff;font-size:17px}
.sub{padding:10px 20px;font-size:13px;background:#f0f0f0;color:#333;border-bottom:1px solid #ddd;line-height:1.5}
table.dr{border-collapse:collapse;margin:8px 20px}
table.dr td,table.dr th{border:1px solid #ccc;padding:5px 12px;font-size:13px;text-align:center}
.qhead{padding:9px 20px;background:#e8eef2;border-top:2px solid #b8c8d0;font-weight:600;font-size:13px}
.args{padding:6px 20px;font-size:12px;color:#555}.a1{color:#1565c0}.a2{color:#b8860b}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:8px 20px}
.col{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}
.chd{padding:6px;font-weight:700;font-size:12px;color:#fff;text-align:center}
.v{padding:4px 8px;font-size:12px;font-weight:700;border-top:1px solid #eee;text-align:center}
.v1{background:#e3f2fd;color:#0d47a1}.v2{background:#fff8e1;color:#8d6e00}
.resp{padding:8px;font-size:10.5px;line-height:1.35;white-space:pre-wrap;max-height:300px;overflow-y:auto;border-top:1px solid #eee}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}.col{background:#1c1c1c;border-color:#333}
.qhead{background:#1a2228}.sub,table.dr td{background:#181818;color:#ccc}.resp{border-color:#333}
.v1{background:#0d2438;color:#7bb5e8}.v2{background:#2a2410;color:#d9b34d}}
</style>
<h1>Does the Gilg preference vector generalize? Steering an "objective" argument-quality judgment</h1>
<div class=sub>
The model is asked to judge two matched-quality arguments <b>"PURELY on reasoning quality — not which conclusion you agree with"</b>, ending in "Better: 1" or "Better: 2".
We apply <b>the exact E4 steering</b> (Gilg vector, L25): +c on argument-1's tokens, &minus;c on argument-2's.
<b>Result: the verdict is fully controlled 0&rarr;1, and it flips on all 20/20 topics</b> — the vector overrides the model's stated objectivity. c=0 = unsteered.
</div>
<table class=dr><tr><th>c (steer &rarr; arg 1)</th><th>&minus;0.06</th><th>&minus;0.03</th><th>0</th><th>+0.03</th><th>+0.06</th></tr>
<tr><td><b>P(rates arg-1 "better")</b></td><td>0.00</td><td>0.00</td><td>0.58</td><td>1.00</td><td>1.00</td></tr></table>
"""]
def cbg(c):
    return {-0.06:"#8e24aa",-0.03:"#5e35b1",0.0:"#455a64",0.03:"#00897b",0.06:"#2e7d32"}.get(c,"#555")

for tid in sorted(topics):
    t = topics[tid]
    parts.append(f'<div class=qhead>{esc(t["topic"])}</div>')
    parts.append(f'<div class=args><span class=a1>Argument 1 (=side A): {esc(t["side_A"]["arg"])}</span><br>'
                 f'<span class=a2>Argument 2 (=side B): {esc(t["side_B"]["arg"])}</span></div>')
    parts.append('<div class=grid>')
    for c in COEFFS:
        r = cell.get((tid, c))
        parts.append(f'<div class=col><div class=chd style="background:{cbg(c)}">c = {c:+.2f}</div>')
        if r:
            v = verdict(r["response"])
            vc = "v1" if v == "1" else ("v2" if v == "2" else "")
            parts.append(f'<div class="v {vc}">Better: {v}' + (' (arg 1)' if v=="1" else (' (arg 2)' if v=="2" else '')) + '</div>')
            parts.append(f'<div class=resp>{esc(r["response"])}</div>')
        parts.append('</div>')
    parts.append('</div>')
open("judging/steer_browser.html", "w").write("".join(parts))
print("wrote judging/steer_browser.html")
