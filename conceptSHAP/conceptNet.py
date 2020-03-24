import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptNet(nn.Module):

    def __init__(self, clusters, h_x, n_concepts):

        super(ConceptNet, self).__init__()

        # note: clusters.shape = (num_clusters, num_sentences_per_cluster, embedding_dim)
        # Tony(1): one caveat here is that different clusters might have different num_sentences_per_cluster,
        # it holds in their case but not sure about our k-means. We might need to manually enforce this by truncation
        embedding_dim = clusters.shape[2]

        self.clusters = nn.Parameter(torch.tensor(clusters))
        self.clusters.requires_grad = False

        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(n_concepts, embedding_dim), requires_grad=True) # the trainable concept

        self.h_x = h_x # final layers of the transformer # Tony(3): make sure weights are frozen
        self.n_concepts = n_concepts

    def init_concept(self, n_concepts, embedding_dim):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(n_concepts, embedding_dim) + r_1
        return concept

    # moved to loss()
    # def concept_loss_fn(self, y_true, y_pred, saliency_score, score_abs):
    #     loss = nn.BCEWithLogitsLoss()
    #     output = loss(y_pred, y_true)
    #
    #     # TODO: add concept-sparsity regularization
    #
    #     return output

    def forward(self, train_embedding):
        """
        :param train_embedding: shape (bs, embedding_dim)
        :return:
        """

        concept_normalized = F.normalize(self.concept, p=2, dim=0)

        # calculating projection of train_embedding onto the concept vector space
        eye = torch.eye(self.n_concepts) * 1e-5
        first_half_proj_matrix = \
            torch.dot(self.concept, torch.inverse(torch.dot(torch.t(self.concept), self.concept) + eye))
        proj = torch.dot(torch.dot(train_embedding, first_half_proj_matrix), torch.t(self.concept))

        # calculating the saliency score between the concept and the cluster
        score_numerator = torch.dot(torch.mean(self.clusters, dim=1), concept_normalized)
        score_numerator_normalized = torch.sub(score_numerator, torch.mean(score_numerator, dim=1, keepdim=True))
        score_abs = torch.abs(F.normalize(score_numerator_normalized, p=2, dim=0)) # cov0_abs
        score_flat = torch.reshape(score_abs, (-1, self.n_concepts))
        saliency_score = torch.dot(torch.t(score_flat), score_flat) # cov

        # passing projected activations through rest of model
        y_pred = self.h_x(proj)

        return y_pred, saliency_score, score_abs # all the stuff that will be used to calculate the loss

    def loss(self, train_embedding, train_y_true, regularize=False, l=5.):
        """
        This function will be called externally to feed data and get the loss
        :param train_embedding:
        :param train_y_true:
        :param regularize:
        :param l: lambda weights
        :return:
        """
        # TODO Tony: check dimension of each tensor
        y_pred, saliency_score, score_abs = self.forward(train_embedding)

        loss = nn.BCEWithLogitsLoss()
        loss_val = torch.mean(loss(y_pred, train_y_true))

        if regularize:
            reg_loss_1 = torch.mean(saliency_score - torch.eye(self.n_concepts))
            reg_loss_2 = torch.mean(score_abs)
            return loss_val + l * (reg_loss_1 + reg_loss_2)
        else:
            return loss_val
