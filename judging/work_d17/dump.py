import json, sys
start, end = int(sys.argv[1]), int(sys.argv[2])
with open('/Users/asri/Projects/value-direction/judging/chunk_disc_17.jsonl') as f:
    rows = [json.loads(l) for l in f]
for i in range(start, min(end, len(rows))):
    r = rows[i]['rendered']
    a = r.find('<model_answer>'); b = r.find('</model_answer>')
    print(f"===== RECORD {i} row_id={rows[i]['row_id']} =====")
    print(r[a+14:b].strip())
    print()
