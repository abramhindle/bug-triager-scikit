#!/bin/bash
for classifier in SVM ZeroR Random MultiNaiveBayesNB NaiveBayesNB
do
	for evalm in MRR Top1 Top5
	do
		AVG=`fgrep $classifier runs.csv | fgrep $evalm | awk -F, '{SUM=SUM+$4}END{print SUM/FNR}'`
		echo $classifier $evalm $AVG
	done
done
	
