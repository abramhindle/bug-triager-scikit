import json
import lda
import os
import sys

ntopics = 20
alpha = 0.01
beta = 0.01
passes = 10

def length(text):
    return len(text)

def dumptojsonfile( filename, data):
    file = open(filename, "w")
    file.write(json.dumps(data, indent=2))
    file.close()

def read_json_file( filename ):
    jsonsdata = open(filename).read()
    data = json.loads(jsonsdata)
    ldocs = [ row["doc"] for row in data["rows"] ]
    docs = {}
    for doc in ldocs:
        docs[doc["_id"]] = doc
    ids = [ doc["_id"] for doc in ldocs ]
    return  docs, ids


def main():
    os.system( "mkdir out" )
    ldocs, lids = read_json_file("large.json")
    dumptojsonfile("out/lids.json", lids)
    docs, dicts = lda.load_lda_docs(ldocs, lids)
    dumptojsonfile("out/dicts.json", dicts)
    words = {v:k for k, v in x.items()}
    dumptojsonfile("out/words.json",words)
    filename, _ = lda.make_vr_lda_input( docs, dicts )
    command = lda.vm_lda_command(filename, ntopics, dicts, alpha=alpha, beta=beta, passes=passes)
    print(command)
    os.system( command )
    topics, summary = lda.summarize_topics_from_file( ntopics, dicts, ("out/topics-%s.dat" % ntopics) )
    document_topic_matrix = lda.summarize_document_topic_matrix_from_file( ntopics, ("out/predictions-%s.dat" % ntopics), passes=passes )
    doc_top_mat_map = dict( (lids[i], document_topic_matrix[i]) for i in range(0,len(lids)) )
    dumptojsonfile("out/lda-topics.json", topics)
    dumptojsonfile("out/summary.json", summary)
    dumptojsonfile("out/document_topic_matrix.json", document_topic_matrix)
    dumptojsonfile("out/document_topic_map.json", doc_top_mat_map)


if __name__ == "__main__":
    main()
