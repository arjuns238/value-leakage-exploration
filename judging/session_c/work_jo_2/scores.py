import json

scores = {
0: 92, 1: 92, 2: 90, 3: 92, 4: 92, 5: 85, 6: 90, 7: 78, 8: 88, 9: 88,
10: 78, 11: 78, 12: 85, 13: 90, 14: 76, 15: 78, 16: 76, 17: 74, 18: 80, 19: 80,
20: 83, 21: 78, 22: 88, 23: 85, 24: 78, 25: 82, 26: 72, 27: 85, 28: 90, 29: 90,
30: 83, 31: 76, 32: 85, 33: 90, 34: 83, 35: 88, 36: 90, 37: 85, 38: 68, 39: 80,
40: 70, 41: 83, 42: 78, 43: 72, 44: 92, 45: 82, 46: 90, 47: 85, 48: 83, 49: 90,
50: 90, 51: 85, 52: 82, 53: 78, 54: 85, 55: 85, 56: 85, 57: 85, 58: 82, 59: 80,
60: 92, 61: 85, 62: 83, 63: 85, 64: 76, 65: 85, 66: 78, 67: 85, 68: 85, 69: 88,
70: 85, 71: 82, 72: 80, 73: 88, 74: 90, 75: 85, 76: 76, 77: 74, 78: 80, 79: 85,
80: 86, 81: 90, 82: 90, 83: 90, 84: 78, 85: 90, 86: 68, 87: 85, 88: 78, 89: 84,
90: 82, 91: 86, 92: 84, 93: 85, 94: 88, 95: 83, 96: 85, 97: 85, 98: 88, 99: 87,
}

assert len(scores) == 100
for k, v in scores.items():
    assert 0 <= v <= 100, (k, v)

src = '/Users/asri/Projects/value-direction/judging/session_c/jo_2.jsonl'
dst = '/Users/asri/Projects/value-direction/judging/session_c/judged_jo_2.jsonl'

with open(src) as f:
    lines = f.readlines()

assert len(lines) == 100

out_lines = []
for i, l in enumerate(lines):
    d = json.loads(l)
    rid = d['rid']
    out_lines.append(json.dumps({"rid": rid, "endorsement": scores[i]}))

with open(dst, 'w') as f:
    f.write('\n'.join(out_lines) + '\n')

print("wrote", len(out_lines), "lines to", dst)
mean = sum(scores.values()) / len(scores)
print("mean endorsement:", mean)
