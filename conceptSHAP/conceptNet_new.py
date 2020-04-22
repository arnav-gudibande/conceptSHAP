import torch
import torch.nn as nn
import torch.nn.functional as F

# DEBUG
import IPython
e = IPython.embed

class ConceptNet_New(nn.Module):

    def __init__(self, clusters, h_x, n_concepts, train_embeddings):

        super(ConceptNet_New, self).__init__()

        # note: clusters.shape = (num_clusters, num_sentences_per_cluster, embedding_dim)
        embedding_dim = clusters.shape[2]

        self.clusters = nn.Parameter(clusters, requires_grad=False)

        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)

        self.h_x = h_x # final layers of the transformer
        self.n_concepts = n_concepts

        self.train_embeddings = train_embeddings.transpose(0, 1) # (dim, all_data_size)

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

        L_sparse_1_old = torch.sum(score_norm)  # maximize this
        # Notes: try to optimize this part, since L_sparse_1 will also be small if none of the concepts are salient
        L_sparse_2_old = 0 # minimize this

        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                if i != j: L_sparse_2_old += torch.dot(score_norm[:, i], score_norm[:, j])

        # score_flat = torch.reshape(score_norm, (-1,)) # ((n_clusters * n_concepts) x 1)
        # saliency_score = score_flat.T @ score_flat

        # passing projected activations through rest of model

        y_pred = self.h_x(proj.T)
        # y_pred = self.h_x(train_embedding)



        ###### Calculate the regularization terms in second version of paper
        # new parameters
        k = 10
        ### calculate first regularization term
        # 1. find the top k nearest neighbour
        all_concept_knns = []
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
            distance = torch.norm(self.train_embeddings - c, dim=0) # (num_total_activations)
            knn = distance.topk(k, largest=False)
            indices = knn.indices # (k)
            knn_activations = self.train_embeddings[:, indices] # (activation_dim, k)
            all_concept_knns.append(knn_activations)

        # 2. calculate the avg dot product for each concept with each of its knn
        L_sparse_1_new = 0.0
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
            c_knn = all_concept_knns[concept_idx] # knn for c
            dot_prod = torch.sum(c * c_knn) / k # avg dot product on knn
            L_sparse_1_new += dot_prod
        L_sparse_1_new = L_sparse_1_new / self.n_concepts

        ### calculate Second regularization term
        all_concept_dot = self.concept.T @ self.concept
        mask = torch.eye(self.n_concepts).cuda() * -1 + 1 # mask the i==j positions
        L_sparse_2_new = torch.mean(all_concept_dot * mask)




        return y_pred, L_sparse_1_old, L_sparse_2_old, L_sparse_1_new, L_sparse_2_new

    def loss(self, train_embedding, train_y_true, regularize=False, l_1=5., l_2=5.):
        """
        This function will be called externally to feed data and get the loss
        :param train_embedding:
        :param train_y_true:
        :param regularize:
        :param l: lambda weights
        :return:
        """
        y_pred, L_sparse_1_old, L_sparse_2, L_sparse_1_new, L_sparse_2_new = self.forward(train_embedding)

        ce_loss = nn.CrossEntropyLoss()
        loss_val_list = ce_loss(y_pred, train_y_true)
        pred_loss = torch.mean(loss_val_list)
        #print(pred_loss)

        if regularize:
            # reg_loss_1 = torch.mean(saliency_score - torch.eye(self.n_concepts))
            # reg_loss_2 = torch.mean(score_abs)
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        return final_loss, pred_loss, L_sparse_1_new * -1, L_sparse_2