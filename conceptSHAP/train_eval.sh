#!/bin/bash

activationDir="$1"
clusterDir="$2"
trainDir="$3"
modelDir="$4"
numConcepts="$5"
conceptSHAPModelDir="$6"
logDir="$7"
lr="$8"
batchSize="$9"
numEpochs="${10}"
lossRegEpoch="${11}"
saveInterval="${12}"

python3 conceptSHAP/train_eval.py \
    --activation_dir=$activationDir \
    --cluster_dir=$clusterDir \
    --train_dir=$trainDir \
    --bert_weights=$modelDir \
    --n_concepts=$numConcepts \
    --save_dir=$conceptSHAPModelDir \
    --log_dir=$logDir \
    --lr=$lr \
    --batch_size=$batchSize \
    --num_epochs=$numEpochs \
    --loss_reg_epoch=$lossRegEpoch \
    --save_interval=$saveInterval
