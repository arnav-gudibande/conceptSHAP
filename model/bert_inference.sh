#!/bin/bash

python3 bert_inference.py \
    --batch_size=256 \
    --activation_dir="../data/medium_activations.npy" \
    --train_dir="../data/sentences_medium.pkl" \
    --bert_weights="imdb_weights"
