#!/usr/bin/env bash

# if subword_nmt is not installed yet, clone it from Git
if [ ! -d "./subword_nmt" ]
then
    echo "Dowloading subword_nmt ..."
    git clone https://github.com/NicolasAG/subword-nmt.git subword_nmt
fi

