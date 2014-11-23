#!/bin/bash
DIR=`dirname $1`
cd $DIR
python3 ../../dumpbayes.py | tee large.csv
cd ..
