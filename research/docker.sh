#!/bin/sh -x
# $1 - docker container name
# $2 - host port to use
#
#  -it --rm --network=host --ipc=host \
#  --runtime=nvidia \
docker run \
  -it --rm -p $2:8888 \
  -v $(pwd):/research \
  $1 \
    bash
