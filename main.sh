#!/bin/bash

CUDA_VISIBLE_DEVICES=1

# Handle the imdb data set and the pre-processing
isDownloaded="$1"  # parse first argument

if [ $isDownloaded -eq 0 ]
then
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
# TODO: Figure out the correct DATAPATH and SAVEPATH
python3 model/bert_inference.py

# Create clusters
python3 clustering/generateClusters.py

# Rest of conceptSHAP
# TODO: SSH to server port (TensorBoard) to plot the training curve
sh conceptSHAP/train.sh
