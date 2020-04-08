#!/bin/bash

python3 generateClusters.py \
    --activation_dir="../data/medium_activations.npy" \
    --cluster_dir="../data/medium_clusters.npy" \
    --n_concepts=5
