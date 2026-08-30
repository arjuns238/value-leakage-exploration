import re, sys, unicodedata, itertools

def load(path):
    names = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*-\s*name:\s*(.+?)\s*$", line)
        if m:
            names.append(m.group(1))
    return names

STOP = {"a","an","the","at","in","on","of","to","for","and","with","up","out",
        "through","from","by","near","into","around","off","down","over","its"}

def norm_tokens(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    s = s.lower().replace("-"," ").replace("'s"," ")
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t not in STOP]

def jac(a, b):
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

orig = load("/Users/asri/Projects/value-direction/value_leakage/choosing_activities/activities.yaml")
new = load("/Users/asri/Projects/value-direction/judging/work_actgen/draft.yaml")
print(f"original: {len(orig)}  new: {len(new)}")

# exact dup check (normalized string)
def key(s): return " ".join(norm_tokens(s))
seen = {}
for n in new:
    k = key(n)
    if k in seen:
        print("EXACT DUP within new:", seen[k], "||", n)
    seen[k] = n
okeys = {key(o): o for o in orig}
for n in new:
    if key(n) in okeys:
        print("EXACT DUP vs original:", okeys[key(n)], "||", n)

# near-dup: jaccard >= 0.55, or shared>=3 tokens and jaccard>=0.45
def flag(pairs, label):
    out = []
    for x, y in pairs:
        tx, ty = norm_tokens(x), norm_tokens(y)
        j = jac(tx, ty)
        shared = len(set(tx) & set(ty))
        if j >= 0.55 or (shared >= 3 and j >= 0.45):
            out.append((round(j,2), shared, x, y))
    out.sort(reverse=True)
    print(f"\n--- near-dup candidates {label}: {len(out)}")
    for j, s, x, y in out:
        print(f"  [{j} shared={s}] {x}  ||  {y}")

flag(itertools.combinations(new, 2), "within new")
flag(((n, o) for n in new for o in orig), "new vs original")

# length stats
lens = sorted(len(n.split()) for n in new)
print(f"\nword counts: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
