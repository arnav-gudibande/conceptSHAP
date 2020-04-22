import torch
from torch.utils.data import (TensorDataset, DataLoader, SequentialSampler)
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification

from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn
import IPython
e = IPython.embed

device = torch.device('cuda')


# LOAD DATA
# IN: filepath to directory where the movie review dataset is stored
# OUT: movie review sentences as a pandas dataframe with 0 polarity for every datapoint

def load_data(PATH):
  small_df = pd.read_pickle(PATH)
  labels = list(small_df["label"])
  polarity = [0 if l=="positive"  else 1 for l in labels]
  small_df["polarity"] = polarity
  return small_df


# LOAD MODEL
# IN: filepath to directory where the model weights are stored
# OUT: loaded model and sentence tokenizer object

def load_model(PATH):

  # config = BertConfig.from_pretrained(PATH + "/config.json", output_hidden_states=True)
  # bert_model = BertForSequenceClassification.from_pretrained(PATH + "/pytorch_model.bin", config=config)
  bert_model = BertForSequenceClassification.from_pretrained(PATH) # ../model/imdb_weights


  # possibly redundant
  bert_model.cuda()
  bert_model.to(device)

  BERTMODEL = "bert-base-uncased"
  tokenizer = BertTokenizer.from_pretrained(BERTMODEL, do_lower_case=True)
  print("model loaded...")
  
  return bert_model, tokenizer


# PROCESS THE DATA
# IN: dataframe of sentences, bert tokenizer
# OUT: dataloader

def process_dataframe(_dframe, _tokenizer, batch_size):

  sentences = _dframe.sentence.values
  sentences = [["[CLS]"] + s for s in sentences]
  
  labels = _dframe.polarity.values 

  # tokenized = [_tokenizer.tokenize(s) for s in sentences]
  # tokenized = [t[:(MAX_LEN_TRAIN-1)]+['SEP'] for t in tokenized]

  tokenized = [t[:(MAX_LEN_TRAIN-1)]+['SEP'] for t in sentences]

  ids = [_tokenizer.convert_tokens_to_ids(t) for t in tokenized]
  ids = np.array([np.pad(i, (0, MAX_LEN_TRAIN-len(i)),
                             mode='constant') for i in ids])
  
  amasks = []
  
  for seq in ids:
    seq_mask = [float(i>0) for i in seq]
    amasks.append(seq_mask)
 
  inputs_reformatted = torch.tensor(ids).cuda()
  labels_reformatted = torch.tensor(labels).cuda()
  masks_reformatted = torch.tensor(amasks).cuda()

  data = TensorDataset(inputs_reformatted, masks_reformatted, labels_reformatted)
  sampler = SequentialSampler(data)

  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  
  return dataloader


# RUN THE MODEL
# IN: dataloader, pre-loaded bert model
# OUT: side effects

def run_model(_model, loader):
  ce_loss = nn.CrossEntropyLoss()

  all_losses = []
  for batch in tqdm(loader):
    b_input_ids, b_input_mask, b_labels = batch
    # print(torch.sum(b_labels).item())
    # outputs doesn't need to be saved
    with torch.no_grad():
        logits, = _model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
        loss_val_list = ce_loss(logits, b_labels)
        pred_loss = torch.mean(loss_val_list).item()
        all_losses.append(pred_loss)
  print("inference loss:", np.mean(np.array(all_losses)))


"""Notice here: the function above added a forward hook to "layer_idx" layer of our model. You might want to google "register_forward_hook" to fully understand it but in short, everytime something is fed into the model and through the layer we specified, the function "extract_activation_hook" will get called. And "extract_activation_hook" will save the layer output to EXTRACTED_ACTIVATIONS when RECORD is true."""
MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

 
######################################################
# ALL STEPS PUT TOGETHER
######################################################

# IN: filepath to data directory, filepath to model weights
# OUT: embeddings

def get_sentence_activation(DATAPATH, MODELPATH, batch_size):

  sentence_df = load_data(DATAPATH)

  model, tokenizer = load_model(MODELPATH)

  loader = process_dataframe(sentence_df, tokenizer, batch_size)

  extracted_activations = []

  def extract_activation_hook(model, input, output):
    extracted_activations.append(output.cpu().numpy())

  def add_activation_hook(model, layer_idx):
    all_modules_list = list(model.modules())
    module = all_modules_list[layer_idx]
    module.register_forward_hook(extract_activation_hook)

  add_activation_hook(model, layer_idx=-2)

  print("running inference..")
  run_model(model, loader) # run the whole model

  #IPython.embed()
  return np.concatenate(extracted_activations, axis=0)


# IN: filepath to data directory, embeddings/activations
# OUT: side effect = writing the activations to a .npy file

def save_activations(activations, DATAPATH):
  np.save(DATAPATH, activations)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--activation_dir", type=str, required=True,
                        help="dir of .npy file to save dataset embeddings")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="path to .pkl file containing train preprocessed dataset")
    parser.add_argument("--bert_weights", type=str, required=True,
                        help="path to BERT config & weights directory")
    args = parser.parse_args()

    result = get_sentence_activation(args.train_dir, args.bert_weights, args.batch_size)
    save_activations(result, args.activation_dir)
