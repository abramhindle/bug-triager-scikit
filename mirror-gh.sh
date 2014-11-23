#!/bin/bash
USER=$1
PROJECT=$2
PROJECT2=$3
if [ -n $PROJECT2 ]
then
	PROJECT=${PROJECT2}
fi
echo USER: $USER
echo PROJECT: $PROJECT
GHUSER=$USER
GHPROJECT=$PROJECT
export GHUSER=$USER
export GHPROJECT=$PROJECT
mkdir data
cd data
mkdir ${PROJECT}
cd ${PROJECT} && \
( [ -e ${PROJECT}.git ] ||  git clone git://github.com/${USER}/${PROJECT}.git ${PROJECT}.git ) && \
( [ -e config.json ]|| ln -s ../../config.json . ) && \
ruby ../../github_issues_to_json.rb
