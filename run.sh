#!/usr/bin/env bash

docker build -t pmldemo .
docker run -it -v "$(pwd)":/usr/src/app -v /tmp:/tmp -w /usr/src/app pmldemo python pml/pmldemo.py