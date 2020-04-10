#!/bin/bash


# Please check the settings below before running the script
CUDA_VISIBLE_DEVICES=1
batchSize=64

# Preprocess arguments
isDownloaded=0  # whether we already have the data
downloadedPath="data/imdb"  # path to the directory of original data
size=1000  # number of training sentences to run fragment extractions on
runOption=2  # 1 for both, 2 for sliding window, 3 for download
trainDir="data/sentence_fragments.pkl"  # sliding window data for extracting inference

# Model saving arguments
modelDir="model/imdb_weights"  # better to make an empty directory just for the BERT model
trainDataDir="data/imdb-train.pkl"  # data to train the complete BERT model
testDataDir="data/imdb-test.pkl"

# Activation inference arguments
activationDir="data/activations.npy"  # dir of .npy file to save dataset embeddings

# Clustering arguments
n_clusters=7
clusterDir="data/clusters.npy"  # path to clustering results

# ConceptSHAP arguments
numConcepts=5
conceptSHAPModelDir="conceptSHAP/models"  # saving directory for conceptSHAP model
logDir="conceptSHAP/logs"
lr=1e-3
numEpochs=20
lossRegEpoch=5  # number of epochs to run without loss regularization
saveInterval=5


# Read command line running option
# six option flags available: 1 for "data preprocess", 2 for "BERT model training",
# 3 for "activation extraction", 4 for "clustering", 5 for "conceptSHAP", and 0 for "running all parts"
option="$1"

if [ $option -eq 1 ] || [ $option -eq 0 ]; then
    # Handle the imdb data set and the pre-processing
    sh data/imdb-dataloader.sh $isDownloaded $downloadedPath $size $runOption $trainDir
fi

if [ $option -eq 2 ] || [ $option -eq 0 ]; then
    # Train BERT model on imdb data sets
    # TODO: SSH to server port (TensorBoard) to plot the training curve
    sh model/bert-imdb.sh $modelDir $trainDataDir $testDataDir
fi

if [ $option -eq 3 ] || [ $option -eq 0 ]; then
    # Extract mid-layer activations
    sh model/bert_inference.sh $batchSize $trainDir $modelDir $activationDir
fi

if [ $option -eq 4 ] || [ $option -eq 0 ]; then
    # Create clusters
    sh clustering/generateClusters.sh $n_clusters $clusterDir $activationDir
fi

if [ $option -eq 5 ] || [ $option -eq 0 ]; then
    # Rest of conceptSHAP
    # TODO: SSH to server port (TensorBoard) to plot the training curve
    sh conceptSHAP/train_eval.sh $activationDir $clusterDir $trainDir $modelDir $numConcepts \
        $conceptSHAPModelDir $logDir $lr $batchSize $numEpochs $lossRegEpoch $saveInterval
fi
