#!/bin/bash

source /etc/profile.d/rvm.sh
source $(rvm 1.9.2 do rvm env --path)

"$@"