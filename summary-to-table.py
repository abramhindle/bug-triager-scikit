import json

with open('out/summary.json') as f:
    summary = json.loads(f.read())
    topic = 1
    for s in summary:        
        print("Topic %s & \\textit{%s} \\\\" % ( topic  , " ".join(s[0:10]) ))
        topic += 1

