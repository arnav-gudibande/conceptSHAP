# coding: utf-8

# Decompose the model and recompose some of them
# just for testing

import torch
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)

from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import BertForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule

from distutils.version import LooseVersion as LV

from sklearn.model_selection import train_test_split

import io

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch.nn as nn
import IPython
e = IPython.embed

if torch.cuda.is_available():
    device = torch.device('cuda')
    devicename = '['+torch.cuda.get_device_name(0)+']'
else:
    device = torch.device('cpu')
    devicename = ""

print('Using PyTorch version:', torch.__version__,
      'Device:', device, devicename)
assert(LV(torch.__version__) >= LV("1.0.0"))


BERTMODEL = "bert-base-uncased"

# BERT MODEL INITIALIZATION
#
# We now load a pretrained BERT model with a single linear
# classification layer added on top.

model = BertForSequenceClassification.from_pretrained(BERTMODEL,
                                                      num_labels=2)
model.cuda()
print('\nPretrained BERT model "{}" loaded'.format(BERTMODEL))


# notice notation:
# f(): the whole classifier
# theta(): the layer from input to the activation we are interested in
# h(): the rest of classifier from activation to classification result
# Thus f(x) = h(theta(x))

# classifier
f_net = model

# decompose into a list of nn.module
f_module_list = []
for idx, m in enumerate(f_net.modules()):
  f_module_list.append(m)

NUM_H_LAYERS = 1

# input -> activation net
# currently set as everything till last layer
theta_net = nn.Sequential(*f_module_list[:-NUM_H_LAYERS]) # everything except the last NUM_H_LAYERS

# activation -> result net
h_net = nn.Sequential(*f_module_list[-NUM_H_LAYERS:]) # the last NUM_H_LAYERS


# TODO: follow the conceptSHAP using these components

e()







