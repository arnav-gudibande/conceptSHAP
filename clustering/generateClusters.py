import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import argparse

def generate_clusters(path, num_clusters):

    print("loading embeddings...")
    embeddings = np.load(path) # embeddings.shape = (4012, 768)

    print("running k-means...")
    clusters = KMeans(n_clusters=num_clusters, n_jobs=-1)
    clusters.fit(embeddings)
    cluster_labels = clusters.labels_

    # calculate dimensions for output clusters
    cluster_count = Counter(cluster_labels)
    num_sentences_per_cluster = min(cluster_count.values())
    embedding_dim = embeddings.shape[1]

    # save clusters
    clusters = [np.empty((0, embedding_dim)) for i in range(num_clusters)]

    print("saving clusters...")
    for i in range(len(embeddings)):
        if len(clusters[cluster_labels[i]]) < num_sentences_per_cluster:
            clusters[cluster_labels[i]] = \
                np.append(clusters[cluster_labels[i]], np.array([embeddings[i]]), axis=0)

    return np.array(clusters)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # Required dependencies
    parser.add_argument("--activation_dir", type=str, required=True,
                        help="path to .npy file containing dataset embeddings")
    parser.add_argument("--cluster_dir", type=str, required=True,
                        help="path to .npy file to save embedding clusters")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="number of concepts to generate")
    args = parser.parse_args()

    print("generating clusters...")
    clusters = generate_clusters(args.activation_dir, args.n_clusters)
    np.save(args.cluster_dir, clusters)
