import torch
import torch.nn as nn
import torch.nn.functional as F

# DEBUG
import IPython
e = IPython.embed

class ConceptNet(nn.Module):

    def __init__(self, clusters, h_x, n_concepts):

        super(ConceptNet, self).__init__()

        # note: clusters.shape = (num_clusters, num_sentences_per_cluster, embedding_dim)
        embedding_dim = clusters.shape[2]

        self.clusters = nn.Parameter(clusters, requires_grad=False)

        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)

        self.h_x = h_x # final layers of the transformer
        self.n_concepts = n_concepts

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    def forward(self, train_embedding):
        """
        :param train_embedding: shape (bs, embedding_dim)
        :return:
        """

        concept_normalized = F.normalize(self.concept, p=2, dim=0) # (embedding_dim x n_concepts)

        # calculating projection of train_embedding onto the concept vector space
        proj_matrix = (self.concept @ torch.inverse((self.concept.T @ self.concept))) \
                      @ self.concept.T # (embedding_dim x embedding_dim)
        proj = proj_matrix @ train_embedding.T # (embedding_dim x batch_size)

        # calculating the saliency score between the concept and the cluster
        cluster_mean = torch.mean(self.clusters, dim=1).type(concept_normalized.dtype) # (n_clusters x embedding_dim)
        score_matrix = torch.abs(cluster_mean @ concept_normalized) # (n_clusters x n_concepts)
        score_norm = F.normalize(score_matrix, p=2, dim=0) # (n_clusters x n_concepts)

        L_sparse_1 = torch.sum(score_norm)  # maximize this
        L_sparse_2 = 0 # minimize this

        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                if i != j: L_sparse_2 += torch.dot(score_norm[:, i], score_norm[:, j])

        # score_flat = torch.reshape(score_norm, (-1,)) # ((n_clusters * n_concepts) x 1)
        # saliency_score = score_flat.T @ score_flat

        # passing projected activations through rest of model
        y_pred = self.h_x(proj.T)

        return y_pred, L_sparse_1, L_sparse_2

    def loss(self, train_embedding, train_y_true, regularize=False, l_1=5, l_2=5):
        """
        This function will be called externally to feed data and get the loss
        :param train_embedding:
        :param train_y_true:
        :param regularize:
        :param l: lambda weights
        :return:
        """
        y_pred, L_sparse_1, L_sparse_2 = self.forward(train_embedding)

        loss = nn.CrossEntropyLoss()
        loss_val_list = loss(y_pred, train_y_true)
        loss_val = torch.mean(loss_val_list)

        if regularize:
            # reg_loss_1 = torch.mean(saliency_score - torch.eye(self.n_concepts))
            # reg_loss_2 = torch.mean(score_abs)
            return loss_val + (l_1 * L_sparse_1) + (l_2 * L_sparse_2)
        else:
            return loss_val