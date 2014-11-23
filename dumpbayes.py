import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from collections import Counter
import os
import os.path
import sklearn.svm
import random

prefixstr = os.path.basename(os.getcwd())

large = json.load(open("large.json"))
def extract_row(issue):
    doc = issue["doc"]
    return [doc["_id"],doc["owner"],doc["content"]]
data = [extract_row(issue) for issue in large["rows"]]
datawo = [extract_row(issue) for issue in large["rows"] if issue["doc"]["owner"] != ""]
#cv = CountVectorizer()
#counts = cv.fit_transform([x[2] for x in data])
#tf = TfidfTransformer(use_idf=False).fit(counts)
#counts2 = cv.fit_transform([x[2] for x in datawo])
#counttf = tf.transform(counts2)
## counttf is the counts of everything
#labels = [x[1] for x in datawo]
#clf = MultinomialNB().fit(counttf, labels)
#predicted = clf.predict(counttf)
#print(clf.predict_log_proba(counttf))
#np.mean(predicted == labels)
#print(metrics.classification_report(labels, predicted))
#p = clf.predict_log_proba(counttf)
#names = clf.classes_
#resrow = p[0]
#reslab = labels[0]

def rank(resrow, correct,names):
    namet = [(val,names[i]) for i, val in enumerate(resrow)]
    namet.sort()
    namet.reverse()
    l = [x[1] for x in namet]
    try:
        return l.index(correct) + 1
    except ValueError:
        return len(names) + 1

def topn(resrow, correct,names,n=5):
    namet = [(val,names[i]) for i, val in enumerate(resrow)]
    namet.sort()
    namet.reverse()
    l = [x[1] for x in namet]
    try:
        return l.index(correct) + 1 <= n
    except ValueError:
        return False

def mrr(p, labels, names):
    ranks = [ rank(row, labels[i], names) for i, row in enumerate(p)]
    rranks = [1.0/rank for rank in ranks]
    return np.mean(rranks)

def topns(p, labels, names, n=5):
    tops = [ topn(row, labels[i], names, n=n) for i, row in enumerate(p)]
    return np.mean(tops)

def eval_tuple(p, labels, names):
	return (mrr(p,labels,names), topns(p,labels,names,n=1), topns(p,labels,names,n=5))

# print("MRR %f\nTop 1 %f\nTop 5 %f" % eval_tuple(p,labels,names))

def run_learn(train_set, train_labels, test_set, test_labels, tfidfcounts=""):
    cv = CountVectorizer()
    counts = cv.fit_transform(train_set + [tfidfcounts])
    tf = TfidfTransformer(use_idf=False).fit(counts)
    counts2 = cv.fit_transform(test_set)
    counttf = tf.transform(counts2)
    # counttf is the counts of everything
    labels = test_labels
    clf = MultinomialNB().fit(counttf, labels)
    #print(metrics.classification_report(labels, predicted))
    p = clf.predict_log_proba(counttf)
    names = clf.classes_
    return eval_tuple(p, test_labels, names)

def run_onevsrest(train_set, train_labels, test_set, test_labels, tfidfcounts=""):
    cv = CountVectorizer()
    counts = cv.fit_transform(train_set + [tfidfcounts])
    tf = TfidfTransformer(use_idf=False).fit(counts)
    counts2 = cv.fit_transform(test_set)
    counttf = tf.transform(counts2)
    # counttf is the counts of everything
    labels = test_labels
    clf = OneVsRestClassifier(MultinomialNB()).fit(counttf, labels)
    #print(metrics.classification_report(labels, predicted))
    p = clf.predict_proba(counttf)
    names = clf.classes_
    return eval_tuple(p, test_labels, names)

#def run_linsvc(train_set, train_labels, test_set, test_labels, tfidfcounts=""):
#    cv = CountVectorizer()
#    counts = cv.fit_transform(train_set + [tfidfcounts])
#    tf = TfidfTransformer(use_idf=False).fit(counts)
#    counts2 = cv.fit_transform(test_set)
#    counttf = tf.transform(counts2)
#    # counttf is the counts of everything
#    labels = test_labels
#    clf = sklearn.svm.LinearSVC().fit(counttf, labels)
#    #print(metrics.classification_report(labels, predicted))
#    p = clf.predict_logproba(counttf)
#    names = clf.classes_
#    return eval_tuple(p, test_labels, names)

def run_svc(train_set, train_labels, test_set, test_labels, tfidfcounts=""):
    cv = CountVectorizer()
    counts = cv.fit_transform(train_set + [tfidfcounts])
    tf = TfidfTransformer(use_idf=False).fit(counts)
    counts2 = cv.fit_transform(test_set)
    counttf = tf.transform(counts2)
    # counttf is the counts of everything
    labels = test_labels
    clf = sklearn.svm.SVC(probability=True).fit(counttf, labels)
    #print(metrics.classification_report(labels, predicted))
    p = clf.predict_log_proba(counttf)
    names = clf.classes_
    return eval_tuple(p, test_labels, names)


def run_zeror(train_set, train_labels, test_set, test_labels, tfidfcounts=""):
    c = Counter(train_labels)
    pred = [x[0] for x in c.most_common()]
    predv = [-1*i for (i,name) in enumerate(pred)]
    names = pred
    return eval_tuple([predv for x in test_labels], test_labels, names)
    
def run_random(train_set, train_labels, test_set, test_labels, tfidfcounts=""):
    names = list(set(train_labels))
    pred = [[random.random() for x in names] for y in test_labels]
    return eval_tuple(pred, test_labels, names)
    

def split_learn(run_learner, dataset, labels):
   shuff = shuffle(range(0,len(dataset)))
   spliti = int(len(labels) / 10)
   test_set = [dataset[shuff[i]] for i in range(0, spliti)]
   test_labels = [labels[shuff[i]] for i in range(0, spliti)]
   train_set = [dataset[shuff[i]] for i in range(spliti,len(labels))]
   train_labels = [labels[shuff[i]] for i in range(spliti,len(labels))]
   return run_learner( train_set, train_labels, test_set, test_labels )


def multi_run(learner,n=50):
    res = [list(split_learn(learner, [x[2] for x in datawo],[x[1] for x in datawo])) for x in range(0,n)]
    return ( np.mean([x[0] for x in res]), 
             np.mean([x[1] for x in res]), 
             np.mean([x[2] for x in res]))

def csv_str(lname,t):
    return "%s,%s,MRR,%f\n%s,%s,Top1,%f\n%s,%s,Top5,%f" % (prefixstr,lname,t[0],prefixstr,lname,t[1],prefixstr,lname,t[2])

print(csv_str("Random",multi_run(run_random,50)))
print(csv_str("MultiNaiveBayesNB",multi_run(run_onevsrest,50)))
print(csv_str("NaiveBayesNB",multi_run(run_learn,50)))
print(csv_str("ZeroR",multi_run(run_zeror,50)))
print(csv_str("SVM",multi_run(run_svc,50)))
#print(csv_str("LinearSVM",multi_run(run_linsvc,50)))

# res = [list(split_learn(run_zeror, [x[2] for x in datawo],[x[1] for x in datawo])) for x in range(0,50)]
# print("MRR %f\nTop 1 %f\nTop 5 %f" % ( np.mean([x[0] for x in res]), 
#                                        np.mean([x[1] for x in res]),
#                                        np.mean([x[2] for x in res])))
# 
