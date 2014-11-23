#!/bin/bash
[[ -f .docker-built-already ]] || \
  docker build -t "lda-chapter-tutorial" . && \
  touch .docker-built-already
docker run -i --rm=true -t -v "$(pwd)/data:/lda-chapter-tutorial/data" lda-chapter-tutorial "$@"