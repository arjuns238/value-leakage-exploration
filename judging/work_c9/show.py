import json,sys
recs=[json.loads(l) for l in open('/Users/asri/Projects/value-direction/judging/chunk_full_r2_9.jsonl')]
a,b=int(sys.argv[1]),int(sys.argv[2])
for k in range(a,b):
    r=recs[k]
    body=r['rendered'].split('<text>',1)[1].split('</text>',1)[0]
    print(f"===== REC {k} =====")
    print(body.strip())
    print()
