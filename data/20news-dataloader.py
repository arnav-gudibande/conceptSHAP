import pandas as pd
import os

#To use, download 20news groups from http://qwone.com/~jason/20Newsgroups/ (18828 samples), and have the directory called 20news in the working
#directory, also may have to remove .DS_Store file if on mac
if __name__ == "__main__":
    indicies = []
    windows = []
    labels = []

    for label in os.listdir("./20news"):
        for text in os.listdir("./20news/" + label):
            try:
                file = open("./20news/" + label + "/" + text, "r", encoding='utf-8')
                contents = file.read()
                split_contents = contents.split()
                for j in range(10, len(split_contents)):
                    sliding_window = split_contents[j-10:j]
                    windows.append(sliding_window)
                    labels.append(label)
            except:
                continue


    indicies = [i for i in range(len(windows))]
    d = {'index': indicies, 'sentence': windows, 'label': labels}
    df = pd.DataFrame(data=d)
    df.to_pickle("./20news.pkl")
