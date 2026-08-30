"""E5 ablation browser: per pair, show the pick at each rank projected out,
with the Sonnet-judged choice. Watch the preferred (activity_1) pick fade to chance."""
import json, glob, html
from collections import defaultdict

rows = [json.loads(l) for l in open("artifacts/session_c/e5_gen.jsonl")]
JCH = {}
for fn in glob.glob("judging/session_c/judged_e5choice_*.jsonl"):
    for l in open(fn):
        if not l.strip(): continue
        j = json.loads(l); JCH[j["rid"]] = j["choice"]
def rid(r): return f'{r["pair_id"]}|r{r["rank"]}|s{r["sample"]}'

RANKS = sorted(set(r["rank"] for r in rows))
groups = defaultdict(lambda: defaultdict(list))
for r in rows:
    groups[r["pair_id"]][r["rank"]].append(r)

def esc(s): return html.escape(s)

# order pairs by gap desc
pair_gap = {pid: g[RANKS[0]][0]["gap"] for pid, g in groups.items()}
order = sorted(groups, key=lambda p: -pair_gap[p])

parts = ["""<meta charset=utf-8><style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}
h1{padding:16px 20px;margin:0;background:#1a1a1a;color:#fff;font-size:17px}
.sub{padding:8px 20px;font-size:13px;background:#f0f0f0;color:#444;border-bottom:1px solid #ddd}
.qhead{padding:10px 20px;background:#e8eef2;border-top:2px solid #b8c8d0;font-weight:600;font-size:13px}
.a1{color:#1565c0}.a2{color:#b8860b}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;padding:10px 20px}
.col{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}
.chd{padding:6px 8px;font-weight:600;font-size:12px;color:#fff;text-align:center}
.pick{padding:5px 8px;font-size:12px;font-weight:600;border-top:1px solid #eee}
.p1{background:#e3f2fd;color:#0d47a1}.p2{background:#fff8e1;color:#8d6e00}.pr{background:#fbe9e7;color:#c62828}
.resp{padding:8px;font-size:10.5px;line-height:1.35;white-space:pre-wrap;max-height:340px;overflow-y:auto;border-top:1px solid #eee}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}.col{background:#1c1c1c;border-color:#333}
.qhead{background:#1a2228}.sub{background:#181818;color:#bbb}.resp{border-color:#333}
.p1{background:#0d2438;color:#7bb5e8}.p2{background:#2a2410;color:#d9b34d}.pr{background:#2a1512;color:#e88}}
</style>
<h1>E5 — rank-k ablation: the covert pick fading as the value subspace is removed</h1>
<div class=sub><b>activity_1 (blue) = the preferred / higher-scored option</b>, always shown as (1). Each column projects out rank-k value directions at L25 during the "pick randomly" generation. Judged pick shown above each response. Leak: P(pick act_1) 0.63 (rank0) &rarr; 0.49 (rank16, = chance). Pairs ordered by preference gap (biggest first). One sample shown per cell.</div>
"""]
def cbg(k):
    return {0:"#455a64",1:"#5e6bad",2:"#6e5ba8",4:"#8e5599",8:"#a84f7a",16:"#c62828"}.get(k,"#555")

for pid in order:
    g = groups[pid]
    ex = g[RANKS[0]][0]
    parts.append(f'<div class=qhead>gap {ex["gap"]:.0f} &nbsp;·&nbsp; <span class=a1>(1) {esc(ex["activity_1"])}</span> &nbsp;vs&nbsp; <span class=a2>(2) {esc(ex["activity_2"])}</span></div>')
    parts.append('<div class=grid>')
    for k in RANKS:
        cell = g[k][0] if g[k] else None
        parts.append(f'<div class=col><div class=chd style="background:{cbg(k)}">rank {k}</div>')
        if cell:
            ch = JCH.get(rid(cell), "?")
            cls = {1:"p1",2:"p2","refusal":"pr"}.get(ch, "pr")
            lbl = {1:"picked (1) preferred",2:"picked (2)","refusal":"refusal"}.get(ch, str(ch))
            parts.append(f'<div class="pick {cls}">{lbl}</div>')
            parts.append(f'<div class=resp>{esc(cell["response"])}</div>')
        parts.append('</div>')
    parts.append('</div>')

open("judging/e5_browser.html", "w").write("".join(parts))
print("wrote judging/e5_browser.html", len(order), "pairs")
