import torch
from conceptNet import ConceptNet
import os
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
from pathlib import Path

from interpretConcepts import eval_clusters, eval_concepts

def train(args, train_embeddings, train_y_true, clusters, h_x, n_concepts, device):
  '''
  :param train_embeddings: tensor of sentence embeddings => (# of examples, embedding_dim)
  :param train_y_true: the ground truth label for each of the embeddings => (# of examples)
  :param clusters: tensor of embedding clusters generated by k-means => (# of n_clusters, # of sentences per cluster, embedding_dim)
  :param h_x: final layers of the transformer
  :param n_concepts: number of concepts to generate
  :return: trained conceptModel
  '''

  # training parameters
  lr = args.lr
  batch_size = args.batch_size
  epochs = args.num_epochs
  save_interval = args.save_interval
  clusters = torch.from_numpy(clusters).to(device)
  train_embeddings = torch.from_numpy(train_embeddings).to(device)
  train_y_true = torch.from_numpy(train_y_true).to(device)
  model = ConceptNet(clusters, h_x, n_concepts).cuda()
  save_dir = Path(args.save_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  log_dir = Path(args.log_dir)
  log_dir.mkdir(exist_ok=True, parents=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  train_size = train_embeddings.shape[0]
  loss_reg_epoch = args.loss_reg_epoch
  writer = SummaryWriter(log_dir=str(log_dir))
  losses = []

  for i in tqdm(range(epochs)):
    if i < loss_reg_epoch:
      regularize = False
    else:
      regularize = True

    batch_start = 0
    batch_end = batch_size

    while batch_end < train_size:

      # generate training batch
      train_embeddings_narrow = train_embeddings.narrow(0, batch_start, batch_end - batch_start)
      train_y_true_narrow = train_y_true.narrow(0, batch_start, batch_end - batch_start)
      loss = model.loss(train_embeddings_narrow, train_y_true_narrow, regularize=regularize, l=5.)

      # update gradients
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      writer.add_scalar('concept_loss', loss.item(), i+1)

      # model saving
      if (i + 1) % save_interval == 0:
          state_dict = model.state_dict()
          for key in state_dict.keys():
              state_dict[key] = state_dict[key].to(torch.device('cpu'))
              torch.save(state_dict, save_dir /
                     'conceptNet_iter_{:d}.pth'.format(i + 1))

      # update batch indices
      batch_start += batch_size
      batch_end += batch_size

  writer.close()
  return model, losses


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Required dependencies
  parser.add_argument("--activation_dir", type=str, required=True,
                      help="path to .npy file containing dataset embeddings")
  parser.add_argument("--cluster_dir", type=str, required=True,
                      help="path to .npy file containing embedding clusters")
  parser.add_argument("--train_dir", type=str, required=True,
                      help="path to .pkl file containing train dataset")
  parser.add_argument("--bert_weights", type=str, required=True,
                      help="path to BERT config & weights directory")
  parser.add_argument("--n_concepts", type=int, default=5,
                      help="number of concepts to generate")

  # Training options
  parser.add_argument('--save_dir', default='./experiments',
                      help='directory to save the model')
  parser.add_argument('--log_dir', default='./logs',
                      help='directory to save the log')
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--loss_reg_epoch', type=int, default=5,
                      help="num of epochs to run without loss regularization")
  parser.add_argument('--num_epochs', type=int, default=20,
                      help="num of training epochs")
  parser.add_argument('--save_interval', type=int, default=5)
  args = parser.parse_args()

  if torch.cuda.is_available():
    device = torch.device('cuda')
    devicename = '[' + torch.cuda.get_device_name(0) + ']'
  else:
    device = torch.device('cpu')
    devicename = ""


  ###############################
  # Preparing data
  ###############################
  # Load assets
  print("Loading dataset embeddings...")
  small_activations = np.load(args.activation_dir)
  print("Shape: " + str(small_activations.shape))

  print("Loading clusters...")
  small_clusters = np.load(args.cluster_dir)
  print("Shape: " + str(small_clusters.shape))

  print("Loading dataset labels...")
  small_df = pd.read_pickle(args.train_dir)
  small_df["polarity"] = small_df.shape[0] * [0]
  senti_list = list(small_df['label'])
  senti_list = [1 if i == "positive" else 0 for i in senti_list]
  senti_list = np.array(senti_list)

  print("Loading model weights...")
  bert_model = BertForSequenceClassification.from_pretrained(args.bert_weights)
  bert_model.to(device)  # move to gpu

  print("Init training...\n")
  # get the embedding numpy array, convert to tensor
  train_embeddings = small_activations  # (4012, 768)

  # get the cluster numpy array
  clusters = small_clusters

  # get ground truth label
  train_y_true = senti_list

  # h_x
  h_x = list(bert_model.modules())[-1]

  # n_concepts
  n_concepts = args.n_concepts  # param

  # data_frame
  data_frame = small_df


  ###############################
  # Training model
  ###############################
  # init training
  concept_model, loss = train(args, train_embeddings, train_y_true, clusters, h_x, n_concepts, device)

  ###############################
  # Interpretation of results
  ###############################
  # evaluate clusters
  cluster_sentiments = eval_clusters(clusters, train_embeddings, train_y_true, data_frame)

  # evaluate concepts
  concept_idxs = list(range(n_concepts)) # the concepts of interest, set to all now
  concepts, saliency = eval_concepts(concept_model, clusters, cluster_sentiments, concept_idxs, train_embeddings, data_frame)
