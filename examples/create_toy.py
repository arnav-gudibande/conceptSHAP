# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the main file to create the toy dataset and clusters."""
#  lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from concept_explanations.ipca import n, n0
import concept_explanations.toy_helper as toy_helper


def main(_):
  """Creates the toy dataset main."""
  # Prepares dataset
  print("creating dataset")
  skip = True # True if you have run create_dataset once already and have the npy files in directory
  if skip:
    print("Skipping generation and saving of data, make sure to have x_data.npy, y_data.npy, concept_data.npy in your directory")
  width, height = toy_helper.create_dataset(skip)
  # Loads dataset
  print("loading dataset")
  x, y, concept = toy_helper.load_xyconcept(n, pretrain=False)
  x_train = x[:n0, :]
  x_val = x[n0:, :]
  y_train = y[:n0, :]
  y_val = y[n0:, :]
  # Loads model
  print("loading model")
  trained = True # change to True if "conv_s13.h5" already exist in directory
  _, _, feature_dense_model = toy_helper.load_model(
      x_train, y_train, x_val, y_val, width=300,
               height=300, channel=3, pretrain=trained)
  print("creating feature")
  toy_helper.create_feature(x, width, height, feature_dense_model)
  # Runs after create_feature
  print("creating cluster")
  toy_helper.create_cluster(concept)


if __name__ == '__main__':
  app.run(main)
