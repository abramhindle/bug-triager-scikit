#!/bin/bash -x
DIR=`dirname $1`
cd $DIR
python3 ../../dumpbayes.py `basename $1` | tee `basename $2`
cd ..
