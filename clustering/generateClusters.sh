#!/bin/bash

n_clusters="$1"
clusterDir="$2"
activationDir="$3"

python3 clustering/generateClusters.py \
    --activation_dir="$activationDir" \
    --cluster_dir="$clusterDir" \
    --n_clusters=$n_clusters
