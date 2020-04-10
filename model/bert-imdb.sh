#!/bin/bash

modelDir="$1"
trainDataDir="$2"
testDataDir="$3"

python3 model/bert-imdb.py \
    --model_dir="$modelDir" \
    --train_data="$trainDataDir" \
    --test_data="$testDataDir"
