#!/usr/bin/env bash

# Change directory to this script's directory
cd "$(dirname $(readlink -f $0))"

/usr/bin/env python3 ./src/model.py || exit 1
/usr/bin/env python3 ./src/test.py || exit 1

