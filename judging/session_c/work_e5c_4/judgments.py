import json

judgments = {
0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1,
16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:2,
24:2,25:2,26:2,27:2,28:2,29:2,30:2,31:2,32:2,33:2,34:2,35:2,36:2,37:2,38:2,39:2,
40:2,41:2,42:2,43:2,44:2,45:2,46:2,47:2,
48:1,49:1,50:2,51:1,52:1,53:1,54:1,55:1,56:1,57:1,58:1,59:1,60:2,61:2,62:1,63:2,
64:1,65:2,66:1,67:1,68:1,69:1,70:2,71:1,
72:2,73:1,74:1,75:1,76:1,77:2,78:2,79:1,80:1,81:1,82:1,83:2,84:1,85:1,86:1,87:1,
88:2,89:2,90:1,91:1,92:1,93:1,94:2,95:1,
96:1,97:1,98:1,99:1,100:1,101:1,102:1,103:1,104:1,105:1,106:1,107:1,108:1,109:1,
110:1,111:1,112:1,113:1,114:1,115:1,116:1,117:1,118:1,119:2,
}
assert len(judgments) == 120, len(judgments)
judgments = [judgments[i] for i in range(120)]

with open('/Users/asri/Projects/value-direction/judging/session_c/e5choice_4.jsonl') as f:
    recs = [json.loads(l) for l in f]
assert len(recs) == 120

with open('/Users/asri/Projects/value-direction/judging/session_c/judged_e5choice_4.jsonl', 'w') as out:
    for r, c in zip(recs, judgments):
        out.write(json.dumps({"rid": r["rid"], "choice": c}) + "\n")

from collections import Counter
print(Counter(judgments))
print("total", len(judgments))
