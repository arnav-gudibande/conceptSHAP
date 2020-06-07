import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import argparse

def save_train_data(SAVEDIR):
    # Note: uses ~7-12GB of memory
    news_train = fetch_20newsgroups(subset="train", shuffle=True,
                                    remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    n_train = np.c_[np.array(news_train.data), np.array(news_train.target)]
    col_train = np.append(['news'], ['label'])
    pd.DataFrame(n_train, columns=col_train).to_pickle(SAVEDIR)

def save_test_data(SAVEDIR):
    news_test = fetch_20newsgroups(subset="test", shuffle=True,
                                   remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    n_test = np.c_[np.array(news_test.data), np.array(news_test.target)]
    col_train = np.append(['news'], ['label'])
    pd.DataFrame(n_test, columns=col_train).to_pickle(SAVEDIR)

def make_sliding_window_pkl(size, dir, savedir):
    data = pd.read_pickle(dir)
    windows = []
    labels = []

    for i in range(size):
        split_review = data.news.values[i].split()
        label = data.label.values[i]
        for j in range(10, len(split_review)):
            sliding_window = split_review[j - 10:j]
            windows.append(sliding_window)
            labels.append(label)

    d = {"sentence": windows, "polarity": labels}
    pd.DataFrame.from_dict(d).to_pickle(savedir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000,
                        help="how many training and test sentences to run fragment extraction on")
    parser.add_argument("--run_option", type=int, default=2,
                        help="3 for download, 2 for sliding window, 1 for both")
    parser.add_argument("--train_dir", type=str, default="train_fragments.pkl",
                        help="path to save extracted sentence fragments")
    parser.add_argument("--test_dir", type=str, default="test_fragments.pkl",
                        help="path to save extracted sentence fragments")
    args = parser.parse_args()
    run_option = args.run_option

    if run_option == 3:
        save_train_data('data/news-train.pkl')
        save_test_data('data/news-test.pkl')
        print("loaded & saved news files")
    elif run_option == 2:
        make_sliding_window_pkl(args.size, 'data/news-train.pkl', args.train_dir)
        make_sliding_window_pkl(args.size, 'data/news-test.pkl', args.test_dir)
        print("saved fragment files")
    else:
        save_train_data('data/news-train.pkl')
        save_test_data('data/news-test.pkl')
        print("loaded & saved news files")
        make_sliding_window_pkl(args.size, 'data/news-train.pkl', args.train_dir)
        make_sliding_window_pkl(args.size, 'data/news-test.pkl', args.test_dir)
        print("saved fragment files")