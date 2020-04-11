import numpy as np

# DEBUG
import IPython
e = IPython.embed

def eval_clusters(clusters, activations, senti_list, df):
  # for loop: need refactor this part in future: not scaling well
  # find index of each vector in cluster, in the original activation list
  # TODO This part shall be done in the extracting cluster, needs to return the sentiments of each cluster
  non_single_count = 0
  clusters_idx = []
  for cluster in clusters:
    # cluster: (303, 768)
    cluster_idx = []
    for activation in cluster:
      idxs = np.where((activation == activations).all(axis=1))[0]
      if len(idxs) > 1:
        non_single_count += 1
      cluster_idx.append(idxs[0])
    clusters_idx.append(cluster_idx)
  clusters_idx = np.array(clusters_idx)
  # sanity check
  for i in range(len(clusters)):
    for j in range(len(clusters[i])):
      if not (clusters[i][j] == activations[clusters_idx[i][j]]).all():
        print(clusters[i][j] == activations[clusters_idx[i][j]])
        assert False, "mismatch index and activation"

  cluster_labels = []
  for idxs in clusters_idx:
    labels = []
    for idx in idxs:
      labels.append(senti_list[idx])
    cluster_labels.append(labels)
  cluster_labels = np.array(cluster_labels) # shape (num_clusters, num_activations_per_cluster),
  # the sentiment(binary) for each activation in cluster matrix
  # TODO This part shall be done in the extracting cluster, needs to return the sentiments of each cluster


  cluster_sentiment = np.count_nonzero(cluster_labels, axis=1) / (cluster_labels.shape[1])
  print("good sentiment ratio for each of", len(cluster_sentiment), "clusters:")
  print(cluster_sentiment)

  cluster_idx = 0
  print("\nexamples from cluster {cluster_idx}\n")
  print_cluster(df, clusters_idx, which_cluster=cluster_idx, n=20)

  return cluster_sentiment

def eval_concepts(concept_model, clusters, concept_idxs, activations, df):
  concepts = concept_model.concept.detach().cpu().numpy() # (activation_dim, num_concepts)
  clusters_mean = np.mean(clusters, axis=1) # (num_clusters, activation_dim)
  # TODO shall we normalize both concept and cluster vectors to norm=1?
  corr = clusters_mean @ concepts # (num_clusters, num_concepts)
  print("correlation of all_cluster<->all_concept, shape (num_clusters, num_concepts)")
  print(corr)

  for concept_idx in concept_idxs:
    # evaluate the concept of interest
    print("\n\nlooking at concept", concept_idx)
    concept = concepts[:, concept_idx]
    concept = np.expand_dims(concept, axis=0) # (1, activation_dim)

    # find nearest neighbour of concept in small activations
    diff = np.abs(activations - concept)
    distance = np.linalg.norm(diff, axis=1) # shape (dataset_size,)

    k = 5 # number of nearest neighbours to choose
    near_idxs = distance.argsort()[:k]  # take first k when ranking from smallest dist to largest

    print("top", k, " nearest neighbours of concept", concept_idx)
    polarity=0
    for idx in near_idxs:
      s = list2str(df["sentence"][idx])
      polarity += df["polarity"][idx]
      print(s)
    print("avg polarity: " + str(polarity/k))

  return concepts, corr

### Utils

def list2str(l):
  s = ""
  for w in l:
    s += w + " "
  return s

def print_cluster(df, clusters_idx, which_cluster, n=20):
  print("picking first", n, "sentences from cluster", which_cluster)
  for idx in clusters_idx[which_cluster][:n]:
    s = list2str(df["sentence"][idx])
    print(idx, s)
  print("\n")