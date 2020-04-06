import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os
import re
import IPython
import argparse

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

def download(args):
    directory = args.download_dir
    train_df = load_dataset(directory + "/train")
    test_df = load_dataset(directory + "/test")

    return train_df, test_df

def make_sliding_window_pkl(size, dir):
  data = pd.read_pickle(dir)
  #data['review'][0], see a review
  windows = []
  labels = []

  for i in range(size):
      split_review = data.sentence.values[i].split()
      label = data.polarity.values[i]
      for j in range(10, len(split_review)):
          sliding_window = split_review[j-10:j]
          #print(sliding_window)
          windows.append(sliding_window)
          labels.append(label)

  indices = [i for i in range(len(windows))]
  d = {"index": indices, "sentence": windows, "polarity": labels}

  df = pd.DataFrame.from_dict(d)
  df.to_pickle("sentence_fragments.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, default="./imdb",
                        help="path to original imdb data files")
    parser.add_argument("--size", type=int, default=1000,
                        help="how many training and test sentences to run fragment extraction on")
    parser.add_argument("--run_option", type=int, default=2,
                        help="0 for download, 1 for sliding window, 2 for both")
    args = parser.parse_args()
    run_option = args.run_option


    if run_option == 3:
        train_df, test_df = download(args.download_dir)
        train_df.to_pickle('./imdb-train.pkl')
        test_df.to_pickle('./imdb-test.pkl')
        make_sliding_window_pkl(args.size, './imdb-train.pkl')
        make_sliding_window_pkl(args.size, './imdb-test.pkl')
    elif run_option == 2:
        make_sliding_window_pkl(args.size, './imdb-train.pkl')
    else:
        train_df, test_df = download(args.download_dir)
        train_df.to_pickle('./imdb-train.pkl')
        test_df.to_pickle('./imdb-test.pkl')
