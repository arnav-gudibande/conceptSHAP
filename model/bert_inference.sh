#!/bin/bash

# Parse arguments
batchSize="$1"
trainDir="$2"
modelDir="$3"
activationDir="$4"

python3 model/bert_inference.py \
    --batch_size=$batchSize \
    --activation_dir="$activationDir" \
    --train_dir="$trainDir" \
    --bert_weights="$modelDir"
