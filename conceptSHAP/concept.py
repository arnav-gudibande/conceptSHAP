import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptModel(nn.Module):

    def __init__(self, clusters, h_x, n_concepts):

        super(ConceptModel, self).__init__()

        # note: clusters.shape = (num_clusters, num_sentences_per_cluster, embedding_dim)
        embedding_dim = clusters.shape[2]

        self.clusters = nn.Parameter(torch.tensor(clusters))
        self.clusters.requires_grad = False

        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(n_concepts, embedding_dim)) # the trainable concept

        self.h_x = h_x # final layers of the transformer
        self.n_concepts = n_concepts

    def init_concept(self, n_concepts, embedding_dim):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(n_concepts, embedding_dim) + r_1
        return concept

    def concept_loss(self):
        pass

    def forward(self, train_embedding, train_label, validation_embedding, validation_label):

        concept_normalized = F.normalize(self.concept, p=2, dim=0)

        # calculating projection of train_embedding onto the concept vector space
        eye = torch.eye(self.n_concepts) * 1e-5
        first_half_proj_matrix = \
            torch.dot(self.concept, torch.inverse(torch.dot(torch.t(self.concept), self.concept) + eye))
        proj = torch.dot(torch.dot(train_embedding, first_half_proj_matrix), torch.t(self.concept))

        # calculating the saliency score between the concept and the cluster
        score_numerator = torch.dot(torch.mean(self.clusters, dim=1), concept_normalized)
        score_numerator_normalized = torch.sub(score_numerator, torch.mean(score_numerator, dim=1, keepdim=True))
        score_abs = torch.abs(F.normalize(score_numerator_normalized, p=2, dim=0))
        score_flat = torch.reshape(score_abs, (-1, self.n_concepts))
        saliency_score = torch.dot(torch.t(score_flat), score_flat)

        # passing projected activations through rest of model
        output = self.h_x(proj)

        # TODO: calculate and return concept loss
