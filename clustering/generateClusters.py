import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

EMBEDDING_PATH = '../data/small_activations.npy'
SAVE_PATH = '../data/small_clusters.npy'

def generate_clusters(path, num_clusters):

    # load embeddings from file
    embeddings = np.load(path) # embeddings.shape = (4012, 768)

    # run k-means clustering
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(embeddings)
    cluster_labels = clusters.labels_

    # calculate dimensions for output clusters
    cluster_count = Counter(cluster_labels)
    num_sentences_per_cluster = min(cluster_count.values())
    embedding_dim = embeddings.shape[1]

    # save clusters
    clusters = [np.empty((0, embedding_dim)) for i in range(num_clusters)]

    print(clusters[0].shape)

    for i in range(len(embeddings)):
        if len(clusters[cluster_labels[i]]) < num_sentences_per_cluster:
            clusters[cluster_labels[i]] = \
                np.append(clusters[cluster_labels[i]], np.array([embeddings[i]]), axis=0)

    return np.array(clusters)

if __name__=="__main__":

    clusters = generate_clusters(EMBEDDING_PATH, 5)
    np.save(SAVE_PATH, clusters)
