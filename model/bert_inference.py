import torch
from torch.utils.data import (TensorDataset, DataLoader, SequentialSampler)
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

import pandas as pd
import numpy as np

import os

device = torch.device('cuda')

def process_dataframe(_dframe, _tokenizer):
  sentences = _dframe.sentence.values
  sentences = ["[CLS] " + s for s in sentences] #causing error bc of data format
  labels = _dframe.polarity.values #WONT WORK, says no such thing as polarity

  tokenized = [_tokenizer.tokenize(s) for s in sentences]
  tokenized = [t[:(MAX_LEN_TRAIN-1)]+['SEP'] for t in tokenized]


  ids = [_tokenizer.convert_tokens_to_ids(t) for t in tokenized]
  ids = np.array([np.pad(i, (0, MAX_LEN_TRAIN-len(i)),
                             mode='constant') for i in ids])
  amasks = []
  for seq in ids:
    seq_mask = [float(i>0) for i in seq]
    amasks.append(seq_mask)
 
  inputs_reformatted = torch.tensor(ids)
  labels_reformatted = torch.tensor(labels)
  masks_reformatted = torch.tensor(amasks)

  data = TensorDataset(inputs_reformatted, masks_reformatted, labels_reformatted)
  sampler = SequentialSampler(data)

  dataloader = DataLoader(data, sampler=sampler, batch_size=1)
  return dataloader

def run_model(_model, loader):

  for batch in loader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

  with torch.no_grad():
      outputs = _model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)
      logits = outputs[0]
  return outputs

# BERT MODEL INITIALIZATION
# source code can be found here https://github.com/huggingface/transformers/blob/bb7c46852051f7d031dd4be0240c9c9db82f6ed9/src/transformers/modeling_bert.py#L1107
PATH = "/home/arnav/Documents/intuit-project/model/"
model = BertForSequenceClassification.from_pretrained(PATH + "imdb_weights") # load directly from checkpoint of imdb
model.to(device) # move to gpu

# PROCESS SIMPLE SENTENCE
BERTMODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERTMODEL, do_lower_case=True)
print("model loaded...")

EXTRACTED_ACTIVATIONS = []
RECORD = False
def extract_activation_hook(module, input, output):
  if RECORD:
    EXTRACTED_ACTIVATIONS.append(output)

def add_activation_hook(model, layer_idx):
  all_modules_list = list(model.modules())
  module = all_modules_list[layer_idx]
  module.register_forward_hook(extract_activation_hook)

add_activation_hook(model, layer_idx=-2)

"""Notice here: the function above added a forward hook to "layer_idx" layer of our model. You might want to google "register_forward_hook" to fully understand it but in short, everytime something is fed into the model and through the layer we specified, the function "extract_activation_hook" will get called. And "extract_activation_hook" will save the layer output to EXTRACTED_ACTIVATIONS when RECORD is true."""
MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

test_sentence = {}
test_sentence["sentence"] = ["this is a great movie, thumbs up!"]
train_df = pd.DataFrame.from_dict(test_sentence)
train_df["polarity"] = [0]
loader = process_dataframe(train_df, tokenizer)

RECORD=True
result = run_model(model, loader) # run the whole model
RECORD=False

def get_sentence_activation():
  return result, EXTRACTED_ACTIVATIONS[-1]


'''

"""# Sanity checks on trained model"""

# sanity checks


MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512
test_sentence = {}
test_sentence["sentence"] = ["This movie is great"]
train_df = pd.DataFrame.from_dict(test_sentence)
train_df["polarity"] = [0]
loader = process_dataframe(train_df, tokenizer)

print(run_model(model, loader))

MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512
test_sentence = {}
test_sentence["sentence"] = ["This movie is awesome"]
train_df = pd.DataFrame.from_dict(test_sentence)
train_df["polarity"] = [0]
loader = process_dataframe(train_df, tokenizer)

print(run_model(model, loader))

MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512
test_sentence = {}
test_sentence["sentence"] = ["This movie is shit"]
train_df = pd.DataFrame.from_dict(test_sentence)
train_df["polarity"] = [0]
loader = process_dataframe(train_df, tokenizer)

print(run_model(model, loader))

MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512
test_sentence = {}
test_sentence["sentence"] = ["This movie is messed up"]
train_df = pd.DataFrame.from_dict(test_sentence)
train_df["polarity"] = [0]
loader = process_dataframe(train_df, tokenizer)

print(run_model(model, loader))


print("the result make sense. Trained model shall be loaded successfully")

'''

