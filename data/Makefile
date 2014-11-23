.SUFFIXES: .json .csv

JSONS=$(shell find -maxdepth 2 -name 'large.json')
CSVS:= $(subst .json,.csv,$(JSONS))

.json.csv:
	bash learner.sh $< 

all: $(CSVS)
