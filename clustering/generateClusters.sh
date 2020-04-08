#!/bin/bash

numConcepts="$1"
clusterDir="$2"
activationDir="$3"

python3 generateClusters.py \
    --activation_dir=$activationDir \
    --cluster_dir=$clusterDir \
    --n_concepts=$numConcepts
