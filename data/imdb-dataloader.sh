#!/bin/bash

# Parse arguments
isDownloaded="$1"
downloadedPath="$2"
size="$3"
runOption="$4"
trainDir="$5"

if [ $isDownloaded -eq 0 ]
then
    # Purely for imdb data set for now
    wget -P data/ "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar -xf data/aclImdb_v1.tar.gz  # extract the file
    rm data/aclImdb_v1.tar.gz  # delete the zipped downloaded file
    mv data/aclImdb data/imdb  # rename directory to imdb

    isDownloaded=1
    downloadedPath="../data/imdb"
fi

if [ $isDownloaded -eq 1 ]
then
    python3 imdb-dataloader.py \
        --download_dir=$downloadedPath \
        --size=$size \
        --run_option=$runOption \
        --train_dir=$trainDir
fi
