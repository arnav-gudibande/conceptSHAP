#!/bin/bash


# Please check the settings below before running the script
# Notice that all directory paths should be relative to the scripts / python files
CUDA_VISIBLE_DEVICES=1
batchSize=64

# Preprocess arguments
isDownloaded=0  # whether we already have the data
downloadedPath="../data/imdb"  # path to the directory of original data
size=1000  # number of training sentences to run fragment extractions on
runOption=2  # 1 for both, 2 for sliding window, 3 for download
trainDir="../data/sentence_fragments.pkl"  # sliding window data for extracting inference

# Model saving arguments
modelDir="../model/imdb_weights"  # better to make an empty directory just for the BERT model

# Activation inference arguments
activationDir="../data/activations.npy"  # dir of .npy file to save dataset embeddings

# Clustering arguments
numConcepts=5
clusterDir="../data/clusters.npy"  # path to clustering results

# ConceptSHAP arguments
conceptSHAPModelDir="../conceptSHAP/models"  # saving directory for conceptSHAP model
logDir="../conceptSHAP/logs"
lr=1e-3
numEpochs=20
lossRegEpoch=5  # number of epochs to run without loss regularization
saveInterval=5


# Handle the imdb data set and the pre-processing
sh data/imdb-dataloader.sh $isDownloaded $downloadedPath $size $runOption $trainDir


# Train BERT model on imdb data sets
# TODO: SSH to server port (TensorBoard) to plot the training curve
python3 model/bert-imdb.py --model_dir=$modelDir


# Extract mid-layer activations
sh model/bert_inference.sh $batchSize $trainDir $modelDir $activationDir


# Create clusters
sh clustering/generateClusters.sh $numConcepts $clusterDir $activationDir


# Rest of conceptSHAP
# TODO: SSH to server port (TensorBoard) to plot the training curve
sh conceptSHAP/train_eval.sh $activationDir $clusterDir $trainDir $modelDir $numConcepts \
    $conceptSHAPModelDir $logDir $lr $batchSize $numEpochs $lossRegEpoch $saveInterval
