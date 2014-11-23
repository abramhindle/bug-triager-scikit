.SUFFIXES: .json .csv

JSONS=$(shell find -maxdepth 2 -name 'large.json')
SJSONS=$(shell find -maxdepth 2 -name 'small.json')
CSVS:= $(subst .json,.csv,$(JSONS))
SCSVS:= $(subst .json,.csv,$(SJSONS))

.json.csv:
	bash learner.sh $< $@

runs.csv: $(CSVS)
	cat $(CSVS) > runs.csv

sruns.csv: $(SCSVS)
	cat $(SCSVS) > sruns.csv

means.csv: runs.csv
	bash means.sh runs.csv | tee means.csv

smeans.csv: sruns.csv
	bash means.sh sruns.csv | tee smeans.csv

all: runs.csv means.csv
