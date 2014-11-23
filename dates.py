import json
import lda_from_json
import dateutil.parser
import time
import csv

docs, ids = lda_from_json.read_json_file("large.json")

with open('out/doc_date.csv', 'wb') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Doc","utime","date","day"])
    for id in ids:
        d = docs[id]["created_at"]
        dp = dateutil.parser.parse(d)
        ud = time.mktime(dp.timetuple())
        day = time.strftime("%Y-%m-%d",dp.timetuple())
        writer.writerow([id,ud,d,day])

    
