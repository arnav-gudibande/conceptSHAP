import numpy as np
import pandas as pd
import torch
from collections import Counter

# DEBUG
import IPython
e = IPython.embed

def concept_analysis(train_embeddings, train_data):
    # concepts: (n_concepts, dim)
    # train_embeddings: (n_embeddings, dim)
    # train_data: df => (n_sentences, label)

    concepts = torch.from_numpy(np.transpose(np.load('conceptSHAP/concepts.npy')))
    train_embeddings = torch.from_numpy(train_embeddings)

    i = 0
    for concept in concepts:
        i+=1
        distance = torch.norm(train_embeddings - concept, dim=1)
        knn = distance.topk(150, largest=False).indices

        words = []
        for idx in knn:
          words += train_data.iloc[int(idx)]['sentence']

        cx = Counter(words)
        most_occur = cx.most_common(25)
        print("Concept " + str(i) + " most common words:")
        print(most_occur)
        print("\n")

def plot_embeddings(train_activations, train_data, senti_list, writer):
  concepts = np.load('conceptSHAP/concepts.npy')

  # plot training activations
  NUM_PLOT=10000
  sentences = [(senti_list[i], ' '.join(train_data.iloc[i]['sentence'])) for i in range(0, NUM_PLOT)]

  # plot clusters & concepts
  embed_met = sentences + ["concept_" + str(i) for i in range(concepts.shape[1])]
  embed = np.vstack((train_activations[:NUM_PLOT], np.transpose(concepts)))
  writer.add_embedding(embed, metadata=embed_met, tag="embeddings")

def save_concepts(concept_model):
  concepts = concept_model.concept.detach().cpu().numpy()
  np.save('conceptSHAP/concepts.npy', concepts)


if __name__=="__main__":
    ### Concept Analysis
    concept_analysis(np.load('data/large_news.npy'), pd.read_pickle('data/news_tr_sentences.pkl'))