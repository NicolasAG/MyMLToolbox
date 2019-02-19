#!/usr/bin/env bash

borgy submit \
  --image images.borgy.elementai.lan/nicolasg/py37mymltoolbox \
  --volume /mnt/home/nicolasg/code:/network/home/angelarn/code \
  --volume /mnt/home/nicolasg/code/MyMLToolbox/external/subword_nmt:/network/home/angelarn/code/MyMLToolbox/external/subword_nmt \
  --volume /mnt/home/nicolasg/data/poetry:/network/home/angelarn/data/poetry \
  --cpu 4 \
  --gpu 1 \
  -- bash -c "while true; do sleep 60; done;"
