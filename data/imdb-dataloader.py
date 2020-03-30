import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os
import re
import IPython

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

def download():
    directory = "./imdb"
    train_df = load_dataset(directory + "/train")
    test_df = load_dataset(directory + "/test")

    return train_df, test_df

def make_sliding_window_pkl():
  data = pd.read_csv('imdb.csv', encoding='latin-1')
  #data['review'][0], see a review
  indicies = []
  windows = []
  labels = []

  for i in range(25):
      split_review = data['review'][i].split()
      label = data['sentiment'][i]
      for j in range(10, len(split_review)):
          sliding_window = split_review[j-10:j]
          #print(sliding_window)
          windows.append(sliding_window)
          labels.append(label)

  indicies = [i for i in range(len(windows))]
  d = {'index': indicies, 'sentence': windows, 'label': labels}

  df = pd.DataFrame(data=d)
  df.to_pickle("./sentenes.pkl")


if __name__ == "__main__":
    train_df, test_df = download()
    train_df.to_pickle('imdb-train.pkl')
    test_df.to_pickle('imdb-test.pkl')
