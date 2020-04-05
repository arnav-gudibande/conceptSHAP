#!/bin/bash

CUDA_VISIBLE_DEVICES=1

# Call the dataloader to pre-process and do sliding windows
python3 data/imdb-dataloader.py

# Train BERT model on imdb data sets
# TODO: SSH to server port (TensorBoard) to plot the training curve
python3 model/bert-imdb.py

# Extract mid-layer activations
# TODO: Figure out the correct DATAPATH and SAVEPATH
python3 model/bert_inference.py

# Create clusters
python3 clustering/generateClusters.py

# Rest of conceptSHAP
sh conceptSHAP/train.sh
