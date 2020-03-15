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

"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import app

import keras
from keras.activations import sigmoid
import keras.backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Layer
from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import SGD
import numpy as np
from numpy import inf
from numpy.random import seed
from scipy.special import comb
import tensorflow as tf
from tensorflow import set_random_seed

import IPython
e = IPython.embed

seed(0)
set_random_seed(0)

# global variables
init = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
batch_size = 128

step = 200
min_weight_arr = []
min_index_arr = []
concept_arr = {}

n = 60000  # total dataset size
n0 = int(n * 0.8)  # training set size


class Weight(Layer):
  """Simple Weight class."""
  # Tony: This class is basically a place holder for variable "self.kernel", even though it inherits from "layer"

  def __init__(self, dim, **kwargs):
    self.dim = dim
    super(Weight, self).__init__(**kwargs)

  def build(self, input_shape):
    # creates a trainable weight variable for this layer.
    self.kernel = self.add_weight(
        name='proj', shape=self.dim, initializer=init, trainable=True)
    super(Weight, self).build(input_shape)

  def call(self, x):
    return self.kernel # Tony: Notice here the input x is not used at all, it just return the weight

  def compute_output_shape(self, input_shape):
    return self.dim


def reduce_var(x, axis=None, keepdims=False):
  """Returns variance of a tensor, alongside the specified axis."""
  m = tf.reduce_mean(x, axis=axis, keep_dims=True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def concept_loss(cov, cov0, i, n_concept, lmbd=5.):
  """Creates a concept loss based on reconstruction loss."""

  def loss(y_true, y_pred):
    if i == 0:
      return tf.reduce_mean(
          tf.keras.backend.binary_crossentropy(y_true, y_pred))
    else:
      return tf.reduce_mean(
          tf.keras.backend.binary_crossentropy(y_true, y_pred)
      ) + lmbd * K.mean(cov - np.eye(n_concept)) + lmbd * K.mean(cov0)

  return loss


def concept_variance(cov, cov0, i, n_concept):
  """Creates a concept loss based on reconstruction variance."""

  def loss(_, y_pred):
    if i == 0:
      return 1. * tf.reduce_mean(reduce_var(y_pred, axis=0))
    else:
      return 1. * tf.reduce_mean(reduce_var(y_pred, axis=0)) + 10. * K.mean(
          cov - np.eye(n_concept)) + 10. * K.mean(cov0)

  return loss


def ipca_model(concept_arraynew2,
               dense2,
               predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               epochs=20,
               metric='binary_accuracy'):
  """Returns main function of ipca."""

  """
  Tony's Note:
  
  This function essentially make a computational graph looks like a Model in tensorflow
  Where the ipca_model has input data phi_x, which is a batch of activations Phi(image), of size (bs, activation_dim) = (?, 200)
  phi_x are real data, like numpy arrays. We got them in another script where we feed images into the first half of Neural Net
  These activations will then be projected to the space spanned by concept vectors(v)
  Projected activation then go through the rest of neural net h(), to get the softmax_pr: classification result
  The loss is then calculated by cross_entropy(softmax_pr, ground_truth_y) <- the "completeness"
  and adding two regularization terms cov, cov0_abs <- the "sparsity"
  
  Variables:
  
  phi_x: The input to the ipca_model, activations of images
  cluster_input: shape (num_clusters, num_image_per_cluster, activation_dim) = (15, 300, 200)
  v: shape (activation_dim, num_concepts) = (200, 5): THE THING WE WANT TO TRAIN: concept vectors!
  v_normalized: make v normal vector of l2norm=1
  eye: identity matrix of shape (num_concepts, num_concepts) = (5, 5)
  first_half_proj_matrix: shape (activation_dim, num_concepts) = (200, 5), refer to v*inv(v.T*v) in formula (1), page 3, old version
  proj: The projection of phi_x onto space spanned by concepts. whole formula (1), page 3, old version
  
  cov1: Mean(cluster_input * v_normalized), which is the same as mean_of_cluster[i] * v_normalized[j] for each i, j
        Has shape (15, 5): Saliency scores of each concept to each cluster
  cov0: let saliency scores(refer to cov1) related to same cluster center around 0
  cov0_abs: cluster-sparsity regularization loss
            let saliency scores related to same cluster sum to 1 after element-wise squaring e.g. if all_scores_for_cluster_i = [x1, .. xn], sum(x1^2, ... xn^2) = 1
            The also take the absolute value: author consider "same direction" and "opposite direction" equally salient
            ^ is kinda sketchy, or I did not understand it correctly
  cov: concept-sparsity regularization loss. Notice inside the function "concept_loss", they are going to minus Identity on this variable, for the i!=j part in formula
  
  ^ both concept and cluster sparsity score can be found page 5 old paper.
  
  dense2, predict: the last two layers of the original classifier: they are h() in f(x) = h(theta(x))
  softmax_pr: shape (bs, y_dim) = (?, 15): the output of classifier, after its activation got projected onto concepts
  
  
  """


  phi_x = Input(shape=(f_train.shape[1],), name='pool1_input') # input layer that takes on the size of the embedding # (?, 200) where ? is bs
  cluster_input = K.variable(concept_arraynew2) # init of the clusters (15, 300, 200) # TODO make sure this is not trainable
  v = Weight((f_train.shape[1], n_concept))(phi_x) # returns the self.kernel (the weight itself) # our concepts # (200, 5) ####### THIS IS WHAT WE ARE TRYING TO LEARN, IT IS THE Concept C!!!!!
  v_normalized = Lambda(lambda x: K.l2_normalize(x, axis=0))(v) # normalized proj_weight # (200, 5)
  eye = K.eye(n_concept) * 1e-5 # id matrix # (5, 5)
  first_half_proj_matrix = Lambda(
      lambda x: K.dot(x, tf.linalg.inv(K.dot(K.transpose(x), x) + eye)))(v) # (200, 5)
  proj = Lambda(lambda x: K.dot(K.dot(x[0], x[2]), K.transpose(x[1]))) (
      [phi_x, v, first_half_proj_matrix]) # v * (v^T *v)^-1 * phi(x)  # phi_x * proj_matrix * v.T # (?, 200)

  cov1 = Lambda(lambda x: K.mean(K.dot(x[0], x[1]), axis=1))(
      [cluster_input, v_normalized]) # Mean(cluster_input * v_normalized) # (15, 5), saliency score numerator
  cov0 = Lambda(lambda x: x - K.mean(x, axis=0, keepdims=True))(cov1) # (15, 5) # Normalize -1
  cov0_abs = Lambda(lambda x: K.abs(K.l2_normalize(x, axis=0)))(cov0) # (15, 5) # Normalize -1
  cov0_abs_flat = Lambda(lambda x: K.reshape(x, (-1, n_concept)))(cov0_abs) # (15, 5)
  cov = Lambda(lambda x: K.dot(K.transpose(x), x))(cov0_abs_flat) # (5, 5)
  # passing the projected activations through the rest of model
  fc2_pr = dense2(proj) # (?, 100)
  softmax_pr = predict(fc2_pr) # (?, 15) # notice 15 here means the output dimension, not the num_clusters. They just happen to be the same

  finetuned_model_pr = Model(inputs=phi_x, outputs=softmax_pr) # instantiates the model
  finetuned_model_pr.layers[-1].activation = sigmoid # make sure the output is in the correct range (0 to 1), since label y is 0 or 1
  finetuned_model_pr.layers[-1].trainable = False # predict (a dense layer), which is from our trained model, freeze it
  finetuned_model_pr.layers[-2].trainable = False # dense2 (a dense layer), which is from our trained model, freeze it
  finetuned_model_pr.layers[-3].trainable = False # a lambda layer, not exactly sure why freeze: there shall be no param at all
  finetuned_model_pr.compile(
      loss=concept_loss(cov, cov0_abs, 0, n_concept), # get the objective # concept_loss is a important function to look at
      optimizer=Adam(lr=0.001), # get the optimizer
      metrics=[metric])
  finetuned_model_pr.fit( ### THIS IS WARM STARTING with ONLY COMPLETENESS SCORE, NO REGULARIZATION
      f_train,
      y_train,
      batch_size=50,
      epochs=epochs,
      validation_data=(f_val, y_val),
      verbose=verbose)

  finetuned_model_pr.layers[-1].trainable = False
  finetuned_model_pr.layers[-2].trainable = False
  finetuned_model_pr.layers[-3].trainable = False
  finetuned_model_pr.compile( # compile the model again, this time with the full loss described in paper
      loss=concept_loss(cov, cov0_abs, 1, n_concept), # the 1 here make loss function include the regularization: last two term in the objective
      optimizer=Adam(lr=0.001),
      metrics=[metric])

  return finetuned_model_pr


def ipca_model_shap(dense2, predict, n_concept, input_size, concept_matrix):
  """returns model that calculates of SHAP."""
  pool1f_input = Input(shape=(input_size,), name='cluster1')
  concept_mask = Input(shape=(n_concept,), name='mask')
  proj_weight = Weight((input_size, n_concept))(pool1f_input)
  concept_mask_r = Lambda(lambda x: K.mean(x, axis=0, keepdims=True))(
      concept_mask)
  proj_weight_m = Lambda(lambda x: x[0] * x[1])([proj_weight, concept_mask_r])
  eye = K.eye(n_concept) * 1e-10
  proj_recon_t = Lambda(
      lambda x: K.dot(x, tf.linalg.inv(K.dot(K.transpose(x), x) + eye)))(
          proj_weight_m)
  proj_recon = Lambda(lambda x: K.dot(K.dot(x[0], x[2]), K.transpose(x[1])))(
      [pool1f_input, proj_weight_m, proj_recon_t])
  fc2_pr = dense2(proj_recon)
  softmax_pr = predict(fc2_pr)
  finetuned_model_pr = Model(
      inputs=[pool1f_input, concept_mask], outputs=softmax_pr)
  finetuned_model_pr.compile(
      loss='categorical_crossentropy',
      optimizer=SGD(lr=0.000),
      metrics=['accuracy'])
  finetuned_model_pr.summary()
  finetuned_model_pr.layers[-7].set_weights([concept_matrix])
  return finetuned_model_pr


def get_acc(binary_sample, f_val, y_val_logit, shap_model, verbose=False):
  """Returns accuracy."""
  acc = shap_model.evaluate(
      [f_val, np.tile(np.array(binary_sample), (f_val.shape[0], 1))],
      y_val_logit,
      verbose=verbose)[1]
  return acc


def shap_kernel(n, k):
  """Returns kernel of shapley in KernelSHAP."""
  return (n-1)*1.0/((n-k)*k*comb(n, k))


def get_shap(nc, f_val, y_val_logit, shap_model, full_acc, null_acc, n_concept):
  """Returns ConceptSHAP."""
  inputs = list(itertools.product([0, 1], repeat=n_concept))
  outputs = [(get_acc(k, f_val, y_val_logit, shap_model)-null_acc)/
             (full_acc-null_acc) for k in inputs]
  kernel = [shap_kernel(nc, np.sum(ii)) for ii in inputs]
  x = np.array(inputs)
  y = np.array(outputs)
  k = np.array(kernel)
  k[k == inf] = 0
  xkx = np.matmul(np.matmul(x.transpose(), np.diag(k)), x)
  xky = np.matmul(np.matmul(x.transpose(), np.diag(k)), y)
  expl = np.matmul(np.linalg.pinv(xkx), xky)
  return expl


def main(_):
  return


if __name__ == '__main__':
  app.run(main)
