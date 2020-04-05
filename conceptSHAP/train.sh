#!/bin/bash

python3 train_eval.py \
    --activation_dir="../data/small_activations.npy" \
    --cluster_dir="../data/small_clusters.npy" \
    --train_dir="../data/sentences_small.pkl" \
    --bert_weights="../model/imdb_weights" \
    --n_concepts=5 \
    --save_dir="./experiments" \
    --log_dir="./logs" \
    --lr=1e-3 \
    --batch_size=32 \
    --num_epochs=200 \
    --loss_reg_epoch=5 \
    --save_interval=50
