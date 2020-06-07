This is an implementation for the paper "On Concept-Based Explanations in Deep Neural
Networks" https://arxiv.org/abs/1910.07969. This specific implementation applies the ConceptSHAP technique to BERT and other transformer-based language models via the [Huggingface Transformers](https://github.com/huggingface/transformers) library. This implementation was developed by members of [Machine Learning @ Berkeley](https://github.com/mlberkeley) for Intuit's Machine Learning Futures Group in Spring 2020.

## Installation & Requirements

* ```git clone https://github.com/arnav-gudibande/intuit-project.git```
* ```pip3 install -r requirements.txt```

## Pipeline Components

* ```data```
    * ```data/imdb-dataloader.py``` -- dataloader for the IMDB Movie Sentiment Dataset, contains options to format test/train data
    * ```data/20news-dataloder.py``` -- dataloader for 20NewsGroups dataset
* ```model```
    * ```bert-20news.py``` and ```bert-imdb.py``` -- training scripts for huggingface bert language model
    * ```bert_inference.py``` -- outputs embeddings generated from a trained transformer model for a target dataset
* ```clustering```
    * ```generateClusters.py``` -- k-means clustering of output embeddings
    * Note: this was discarded from the intitial conceptSHAP paper, but can still be used to test classical unsupervised methods against conceptSHAP
* ```conceptSHAP```
    * ```conceptNet.py``` -- trainable subclass that learns concepts
    * ```train_eval.py``` -- training script for ```conceptNet.py```
    * ```interpretConcepts.py``` -- post-training concept analysis and tensorboard plotting
    
## Example Usage

#### IMDB Sentiment Dataset
* Download and format IMDB Dataset: ```sh data/imdb-dataloader.sh```
* Train BERT model on IMDB: ```sh model/bert-imdb.sh```
* Generate and save BERT embeddings: ```sh model/bert-inference_imdb.sh```
* Run ConceptSHAP: ```sh conceptSHAP/train_eval_imdb.sh```

#### 20NewsGroups Dataset
* Download and format 20News: ```sh data/20news-dataloader.sh```
* Train BERT model on 20News: ```python3 model/bert-20news.py```
* Generate and save BERT embeddings: ```sh model/bert-inference_20news.sh```
* Run ConceptSHAP: ```sh conceptSHAP/train_eval_20news.sh```

### Tensorboard
* ```tensorboard --logdir=runs --port=6006```
