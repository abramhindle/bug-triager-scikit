#!/bin/bash
RUNS=$1
for classifier in SVM ZeroR Random MultiNaiveBayesNB NaiveBayesNB 1NN 3NN 5NN 
do
	for evalm in MRR Top1 Top5
	do
		AVG=`fgrep $classifier $RUNS | fgrep $evalm | fgrep -v ,nan | awk -F, '{SUM=SUM+$4}END{print SUM/FNR}'`
		echo $classifier,$evalm,$AVG
	done
done
