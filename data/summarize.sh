#!/bin/bash
rm runs.csv
find -name 'large.json' | sed -e 's/\/large.json//' | XARGS -i echo '( cd {} ; python3 ../../dumpbayes.py ; cd .. )' | bash | tee runs.csv
bash means.sh
