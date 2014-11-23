import arff
import json
large = json.load(open("large.json"))
def extract_row(issue):
    doc = issue["doc"]
    return [doc["_id"],doc["owner"],doc["content"]]
data = [extract_row(issue) for issue in large["rows"]]
datawo = [extract_row(issue) for issue in large["rows"] if issue["doc"]["owner"] != ""]
arff.dump('large.arff', data, relation="large.json", names=['id', 'owner', 'content'])
arff.dump('largewo.arff', datawo, relation="large.json", names=['id', 'owner', 'content'])

