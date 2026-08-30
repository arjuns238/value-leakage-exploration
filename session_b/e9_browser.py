"""E9 browser: per question, donation & naked bets at rank 0 vs 16, with judged estimate + disclosure."""
import json, glob, re, html
from collections import defaultdict

rows = [json.loads(l) for l in open("artifacts/session_c/e9_gen.jsonl")]
JEST, JDISC = {}, {}
for fn in glob.glob("judging/session_c/judged_e9est_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); m = re.search(r"<final_estimate>(.*?)</final_estimate>", j["answer"])
        JEST[j["prompt_hash"]] = (m.group(1).strip() if m else "UNKNOWN")
for fn in glob.glob("judging/session_c/judged_e9disc_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); JDISC[j["prompt_hash"]] = j["category"]
def key(r): return f'{r["qkey"]}|{r["condition"]}|r{r["rank"]}|s{r["sample"]}'
def esc(s): return html.escape(s)

groups = defaultdict(lambda: defaultdict(list))  # qkey -> (cond,rank) -> rows
for r in rows:
    groups[r["qkey"]][(r["condition"], r["rank"])].append(r)
CELLS = [("donation",0),("donation",16),("naked",0),("naked",16)]

parts = ["""<meta charset=utf-8><style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}
h1{padding:16px 20px;margin:0;background:#1a1a1a;color:#fff;font-size:17px}
.sub{padding:8px 20px;font-size:13px;background:#f0f0f0;color:#444;border-bottom:1px solid #ddd}
.qhead{padding:10px 20px;background:#e8eef2;border-top:2px solid #b8c8d0;font-weight:600;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:10px 20px}
.col{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}
.chd{padding:6px 8px;font-weight:600;font-size:12px;color:#fff;text-align:center}
.meta{padding:5px 8px;font-size:11px;border-top:1px solid #eee;background:#fff8e1}
.INFLUENCED{color:#c62828;font-weight:700}.NO_MENTION{color:#2e7d32}.NO_STATEMENT{color:#888}
.resp{padding:8px;font-size:10.5px;line-height:1.35;white-space:pre-wrap;max-height:360px;overflow-y:auto;border-top:1px solid #eee}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}.col{background:#1c1c1c;border-color:#333}
.qhead{background:#1a2228}.sub{background:#181818;color:#bbb}.resp{border-color:#333}.meta{background:#2a2410}}
</style>
<h1>E9 — ablate the value subspace during the Donation Bet (does the bet's influence live there?)</h1>
<div class=sub>Same rank-16 ablation that drove the covert random-pick to chance (E5). Two bet types (regular donation · run-200m-naked) at <b>rank 0 (baseline)</b> vs <b>rank 16 (ablated)</b>. Result: ablation barely moves stated-intent or anchoring — the bet's influence is NOT in this subspace. Disclosure label + judged estimate shown per cell (first sample).</div>
"""]
def cbg(cond, rank):
    base = {"donation":(0x15,0x65,0xc0), "naked":(0x8e,0x24,0xaa)}[cond]
    f = 1.0 if rank==0 else 0.55
    return "#%02x%02x%02x" % tuple(int(x*f) for x in base)

for qkey in sorted(groups):
    ex = groups[qkey][CELLS[0]][0]
    parts.append(f'<div class=qhead>{esc(qkey)} · threshold {ex["threshold"]:,}</div>')
    parts.append('<div class=grid>')
    for (cond, rank) in CELLS:
        r = groups[qkey][(cond, rank)][0]
        jest = JEST.get(key(r), "UNKNOWN"); jdisc = JDISC.get(key(r), "?")
        side = f'est={float(jest):,.0f}' if jest.upper()!="UNKNOWN" else 'no committed est'
        parts.append(f'<div class=col><div class=chd style="background:{cbg(cond,rank)}">{cond} · rank {rank}</div>')
        parts.append(f'<div class=meta><span class="{esc(jdisc)}">{esc(jdisc)}</span> · {side}</div>')
        parts.append(f'<div class=resp>{esc(r["response"])}</div></div>')
    parts.append('</div>')
open("judging/e9_browser.html", "w").write("".join(parts))
print("wrote judging/e9_browser.html")
