import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
import json
import re
import math
import csv

def dumptojsonfile( filename, data, indent=2):
    file = open(filename, "w")
    file.write(json.dumps(data, indent=indent))
    file.close()


def get_ids(db):
    ids = db['_all_docs']
    ids = [x["id"] for x in ids["rows"] if x["id"][0] != "_" ]
    return ids

def d2s(v):
    if (v == {}):
        return ""
    return v

def blank( d ):
    if (d == None):
        return ''
    else:
        return d

def d2sblank(doc, key):
    return blank(d2s(doc.get(key,"")))

def as_list(v):
    if (v.__class__ == [].__class__):
        return v
    return [v]

def reverse_dict( dicts ):
    return dict((v,k) for k, v in dicts.iteritems())

_stopwords = False
def stopwords():
    global _stopwords
    if (_stopwords == False):
        try:
            _stopwords = set(open("stop_words").read().splitlines())
        except:
            _stopwords = dict()
        _stopwords.add("")
        _stopwords.add("'s")
    return _stopwords

#stopwordp = re.compile("^[a-zA-Z0-9\\._\\/\\\\]+$",re.I)
stopwordp = re.compile("^[a-zA-Z]+$",re.I)

def filter_stopwords( tokens ):
    global stopwordp
    sw = stopwords()
    w1 = [x for x in tokens if not(x in sw)] 
    return [y for y in w1 if not(None == stopwordp.match(y))]

_tokenizer = RegexpTokenizer(r'\w+')
#_tokenizer = TreebankWordTokenizer()

def tokenize( text, tokenizer=_tokenizer):
    tokens = filter_stopwords( tokenizer.tokenize( text.lower() ) )
    return tokens


def filter_words_by_frequency(dicts, doc_db, filter_fun):
    newdict = dict()
    newdocs = dict()
    ndocs = dict()
    removedwords = set()
    # count words
    for key, doc in doc_db.iteritems():
        for w in doc:
            ndocs[w] = ndocs.get(w,0) + 1
    keptwords = dict((x,c) for x,c in ndocs.iteritems() if filter_fun(x,c))
    newmap = dict()
    cnt = 2
    for word, count in keptwords:
        newmap
    # drop the bad words
    for key, doc in doc_db.iteritems():
        ndocs[key] = [w for w in doc if w in keptwords]
    # map the words to new values
    for key, doc in doc_db.iteritems():
        ndocs[key] = [w for w in doc if w in keptwords]
    raise Exception("Not Done Don't Use")    

def filter_words_in_only_n_file(dicts, doc_db, n=1):
    ndocs = len(doc_db)
    mincount = n
    def thresh(word, count):
        return (count >= mincount)
    
    return filter_words_by_frequency( dicts, doc_db, thresh)

def filter_uncommon_word(dicts, doc_db, threshold=0.01):
    ndocs = len(doc_db)
    mincount = max(1,math.ceil(ndocs * threshold))
    def thresh(word, count):
        return (count >= mincount)    
    return filter_words_by_frequency( dicts, doc_db, thresh)

def filter_common_word(dicts, doc_db, threshold=0.8):
    ndocs = len(doc_db)
    mincount = max(1,math.ceil(ndocs * threshold))
    def thresh(word, count):
        return (count <= mincount)    
    return filter_words_by_frequency( dicts, doc_db, thresh)

def filter_and_uncommon_common_word(dicts, doc_db, lowthreshold=0.01, highthreshold=0.8):
    ndocs = len(doc_db)
    mincount = max(1,math.ceil(ndocs * lowthreshold))
    maxcount = max(1,math.ceil(ndocs * highthreshold))
    def thresh(word, count):
        return (count >= mincount and count <= maxcount)
    return filter_words_by_frequency( dicts, doc_db, thresh)



def add_doc( doc, dicts, counter, tokenizer=_tokenizer ):
    tokens = tokenize( doc )
    # count tokens and make dictionary from them
    counts = dict()
    for token in tokens:
        if (dicts.get(token, -1) == -1):
            dicts[token] = counter
            counter += 1
        id = dicts[token]
        oldcount = counts.get(id, 0)
        counts[id] = oldcount + 1
    return (counter, counts)



def extract_text_from_document( doc ):
    doc_comments = as_list(doc.get("comments",[]))
    comments = "\n".join([ d2s(x.get("what","")) + d2s(x.get("content",""))  or '' for x in doc_comments])
    outdoc = "\n".join([
            d2sblank(doc,"title"),
            d2sblank(doc,"description"),
            d2sblank(doc,"content"),
            blank(comments)
            ])
    return outdoc

def load_lda_docs(db, ids, extractor=extract_text_from_document, tokenizer=_tokenizer ):
    dicts  = {"_EMPTY_":1}
    counter = 2
    docs = dict()
    for id in ids:
        print(id)
        doc = db[id]
        #if (id == 180):
        #    pdb.set_trace()
        #    print(doc["content"])
        outdoc = extractor( doc )
        counter, counts = add_doc( outdoc, dicts, counter, tokenizer=tokenizer )
	if (len(counts) == 0):
            counts = {1:1}
        docs[id] = counts
    return docs, dicts


def doc_to_vr_lda( doc ):
    return "| " + " ".join([ str(key) + ":" + str(doc[key]) for key in doc ]) + "\n"

def make_vr_lda_input( docs, dicts, filename = "out/vr_lda_input.lda.txt", filenameid = "out/vr_lda_input.ids.txt"):
    file = open( filename, "w+")
    docids = docs.keys()
    docids.sort()
    for docid in docids:
        file.write( doc_to_vr_lda( docs[docid] ) )
        file.write
    file.close()
    
    ifile = open( filenameid, "w" )
    ifile.write( json.dumps([str(docid) for docid in docids], indent=2))
    ifile.close()
    # words
    filenamewords = "out/vr_lda_input.words.txt"
    wfile = open( filenamewords, "w" )
    wfile.write(json.dumps(dicts, indent=2))
    wfile.close()

    return filename, filenameid

def dict_bits( dicts ):
    return int(math.ceil(math.log(len(dicts),2)))
    
def vm_lda_command( filename, topics, dicts, alpha=0.01, beta=0.01, passes=1):
    stopics = str(topics)
    bits = dict_bits(dicts)
    # removed cache file
    try:
        os.remove("out/topic-%s.dat.cache" % (stopics))
    except:
        True
    #return " %s --lda %s --lda_alpha 0.1 --lda_rho 0.1 --minibatch 256 --power_t 0.5 --initial_t 1 -b %d --passes 2 -c  -p out/predictions-%s.dat --readable_model out/topics-%s.dat %s" % (
    return " %s --lda %s --lda_alpha %s --lda_rho %s --minibatch 256 --power_t 0.5 --initial_t 1 -b %d -c --passes %d -p out/predictions-%s.dat --readable_model out/topics-%s.dat  %s" % (
        "vw",
        stopics,
	alpha,
	beta,
        bits,
        passes,
        stopics,
        stopics,
        filename
        )


def summarize_topics( n, dicts, readable_model_lines, max_words = 256 ):
    '''  this returns a summary of the topic model as words and a matrix of terms '''
    nlines = len( readable_model_lines )
    topics = [[0 for x in range(0, nlines)] for y in range(0, n)] 
    word = 0
    # we need version 7 now
    if ("Version" in readable_model_lines[0]):
        readable_model_lines.pop(0)
    for line in readable_model_lines:
        if (not (":" in line) and not ("Version" in line)):            
            line = line.rstrip()
            topic = 0
            #print("["+line+"]")
            elms = [float(x) for x in line.split(" ")[1:]]
            for topic in range(0, n):
                topics[topic][word] = elms[topic]
            word += 1
    # now we have that matrix
    # per each topic find the most prevelant word
    summary = [[] for x in range(0, n)]
    revdict = reverse_dict( dicts )
    for topici in range(0, n):
        topic = topics[topici]
        #print("Topic Length %d" % (len(topic)))
        #print("RevDict Length %d" % (len(revdict)))
        #print("Dicts Length %d" % (len(dicts)))
        indices = range(0,len(topic))
        indices.sort( key = lambda i: topic[i], reverse = True )
        words = [ revdict.get(i,("NOTFOUND: %d" % i)) for i in indices[0:max_words] ]
        summary[ topici ] = words
    return topics , summary

def sorted_indices( x ):
    indices = range(0,len(x))
    indices.sort( key = lambda i: x[i] )
    return indices;
    

def summarize_topics_from_file(n, dicts, readable_model_filename ):
    file = open( readable_model_filename, "r" )
    text = file.readlines()
    file.close()
    return summarize_topics(n, dicts, text )

def summarize_document_topic_matrix(n, lines, passes=1):
    ''' each line is a set of numbers indicating the topic association '''
    if (passes > 1):
        lines = lines[len(lines) - len(lines)/passes:]
    nlines = len( lines )
    docs = [[0 for x in range(0, n)] for y in range(0, nlines)] 
    return [[float(x) for x in line.rstrip().split(" ")] for line in lines]


def summarize_document_topic_matrix_from_file( n, document_topic_matrix_filename, passes=1 ):
    file = open( document_topic_matrix_filename, "r" )
    text = file.readlines()
    file.close()
    return summarize_document_topic_matrix( n, text, passes=passes )

# LDA Object -- manages most state internally
class LDA(object):
    def __init__(self, params=None):
        global _tokenizer
        if (params == None):
            params = dict()
        self.alpha = params.get("alpha",0.1)
        self.beta = params.get("beta",0.1)
        self.passes = params.get("passes",1)
        self.ntopics = params.get("ntopics", params.get("topics", 20))
        self.tokenizer = params.get("tokenizer",_tokenizer)
        self.extractor = params.get("extractor", extract_text_from_document)
        self.init_params = params
        self.documents_loaded = False
        self.lda_prepared = False
        self.command = None
        self.lda_has_run = False

    def load_documents(self, document_db, ids):
        docs, dicts = load_lda_docs(document_db, ids, extractor=self.extractor, tokenizer=self.tokenizer)
        self.ids = ids
        self.docs = docs
        self.dicts = dicts
        self.words = {v:k for k, v in dicts.items()}
        self.documents_loaded = True
        return (self.docs, self.dicts, self.words)

    def prepare_lda(self):
        if (not self.documents_loaded):
            raise Exception("Documents are not loaded!")
        filename, _ = make_vr_lda_input( self.docs, self.dicts )
        self.filename = filename
        self.lda_prepared = True
        
    def lda_command(self):
        if (not self.lda_prepared):
            raise Exception("LDA Not prepared")
        command = vm_lda_command(self.filename, self.ntopics, self.dicts, alpha=self.alpha, beta=self.beta, passes=self.passes)
        self.command = command
        print("Command %s" % command)
        return command

    def run_lda(self):
        if (self.command == None):
            self.lda_command()
        print(self.command)
        os.system( self.command )
        self.lda_has_run = True
        return self.lda_has_run

    def summarize_topics(self):
        if (not self.lda_has_run):
            raise Exception("LDA Not Run -- Run LDA first")
        self.topic_file_name = ("out/topics-%s.dat" % self.ntopics)
        topics, summary = summarize_topics_from_file( self.ntopics, self.dicts, self.topic_file_name  )
        self.topics = topics
        self.summary = summary
        return (topics, summary)

    def summarize_document_topic_matrix(self):
        self.prediction_file = ("out/predictions-%s.dat" % self.ntopics)
        document_topic_matrix = summarize_document_topic_matrix_from_file( self.ntopics, self.prediction_file , passes=self.passes )
        doc_top_mat_map = dict( (self.ids[i], document_topic_matrix[i]) for i in range(0,len(self.ids)) )
        self.document_topic_matrix = document_topic_matrix
        self.doc_top_mat_map = doc_top_mat_map
        return (document_topic_matrix, doc_top_mat_map)

    def dump_doc_top_to_csv(self):
        try:
            ids = []
            try:
                ids = [ int(x) for x in self.doc_top_mat_map.keys() ]
                ids.sort()
                #ids = [str(x) for x in ids]
            except:
                ids = [x for x in self.doc_top_mat_map.keys()]
                ids.sort()
            doctop = self.doc_top_mat_map
            head = ["Doc"] + ["T%s" % x for x in range(1,self.ntopics+1)]
            with open('out/document_topic_map.csv', 'wb') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(head)
                for id in ids:
                    writer.writerow([id]+doctop[id])
            with open('out/document_topic_map_norm.csv', 'wb') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(head)
                for id in ids:
                    s = sum(doctop[id])
                    writer.writerow([id]+[x/s for x in doctop[id]])
        except AttributeError:
            raise Exception("Please run summarize_topics and summarize_document_topic_matrix")
                
    def dump_to_json(self):
        try:
            dumptojsonfile("out/lda-topics.json", self.topics)
            dumptojsonfile("out/summary.json", self.summary)
            dumptojsonfile("out/document_topic_matrix.json", self.document_topic_matrix)
            dumptojsonfile("out/document_topic_map.json", self.doc_top_mat_map)
            self.dump_doc_top_to_csv()
        except AttributeError:
            raise Exception("Please run summarize_topics and summarize_document_topic_matrix")
        
    def run(self):
        if (not self.documents_loaded):
            raise Exception("Documents are not loaded!")
        self.prepare_lda()
        self.lda_command()
        self.run_lda()
        self.summarize_topics()
        self.summarize_document_topic_matrix()
        self.dump_to_json()
        return True
