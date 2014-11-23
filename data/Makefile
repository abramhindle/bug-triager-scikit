.SUFFIXES: .json .csv

JSONS=$(shell find -maxdepth 2 -name 'large.json')
CSVS:= $(subst .json,.csv,$(JSONS))

.json.csv:
	bash learner.sh $< 

runs.csv: $(CSVS)
	cat $(CSVS) > runs.csv

means.csv: runs.csv
	bash means.sh | tee means.csv

all: runs.csv means.csv
