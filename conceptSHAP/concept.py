import torch
import torch.nn as nn

class ConceptModel(nn.Module):

    def __init__(self, clusters, decoder, train_embeddings,
                 train_labels, validation_embeddings,
                 validation_labels, n_concepts, epochs=20):

        super(ConceptModel, self).__init__()

        # note: clusters.shape = (num_clusters, num_sentences_per_cluster, embedding_dim)
        embedding_dim = clusters.shape[2]

        self.clusters = nn.Parameter(torch.tensor(clusters))
        self.clusters.requires_grad = False

        concept = self.init_concept(embedding_dim, n_concepts)
        self.V = nn.Parameter(concept)
        self.V_norm = nn.Parameter(nn.functional.normalize(concept))

        # TODO: eye -> softmax_pr

    def init_concept(self, embedding_dim, n_concepts):
        '''
        :param n_concepts: number of concepts
        :return: concept: uniformly distributed tensor of size (embedding_dim, n_concepts)
        '''
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept


    def forward(self):
        pass

    def concept_loss(self):
        pass

    def concept_variance(self):
        pass

    def tensor_variance(self):
        pass

