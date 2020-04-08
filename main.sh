#!/bin/bash

CUDA_VISIBLE_DEVICES=1

# Handle the imdb data set and the pre-processing
isDownloaded="$1"  # parse first argument

if [ $isDownloaded -eq 0 ]
then
    # Purely for imdb data set for now
    wget -P data/ "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar -xf data/aclImdb_v1.tar.gz  # extract the file
    rm data/aclImdb_v1.tar.gz  # delete the zipped downloaded file
    mv data/aclImdb data/imdb  # rename directory to imdb

    # Call the data loader to pre-process and do sliding windows with default directory
    python3 data/imdb-dataloader.py
fi

if [ #isDownloaded -eq 1 ]
then
    downloadedPath="$2"  # parse second argument
    # Call the data loader to pre-process and do sliding windows
    python3 data/imdb-dataloader.py --download_dir=$downloadedPath
fi

# Train BERT model on imdb data sets
# TODO: SSH to server port (TensorBoard) to plot the training curve
python3 model/bert-imdb.py

# Extract mid-layer activations
python3 bert_inference.py \
    --batch_size=256 \
    --activation_dir="../data/medium_activations.npy" \
    --train_dir="../data/sentences_medium.pkl" \
    --bert_weights="imdb_weights"

# Create clusters
python3 clustering/generateClusters.py

# Rest of conceptSHAP
# TODO: SSH to server port (TensorBoard) to plot the training curve
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
    --num_epochs=20 \
    --loss_reg_epoch=5 \
    --save_interval=50
