#!/bin/bash

python3 bert_inference.py \
    --batch_size=1 \
    --activation_dir="../data/large_activations.npy" \
    --train_dir="../data/large_sentences.pkl" \
    --bert_weights="imdb_weights"
