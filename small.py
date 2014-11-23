#!/usr/bin/python3
import json
tsvfile = "/home/hindle1/projects/ali-study/DataSetsForTriaging/issues2-forTop20Projects.tsv"
lines = open(tsvfile).readlines()
ghids = set([line.split("\t")[0] for line in lines])
issues = json.load(open("issues.json"))
goodids = set()
for issue in issues:
    if str(issue["id"]) in ghids:
        goodids.add(issue["number"])

print(goodids)
# issues = None

large = json.load(open("large.json"))
newrows = list()
for issue in large["rows"]:
    if (issue["doc"]["_id"] in goodids):
        newrows.append(issue)

large["rows"] = newrows
json.dump(large,open("small.json","w"),indent=1)
