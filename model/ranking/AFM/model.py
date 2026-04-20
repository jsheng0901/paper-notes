import torch.nn as nn
import torch
import torch.nn.functional as F

from notes.ranking.layer.layer import EmbeddingsInteraction, FeaturesEmbedding, FeaturesLinear


class AttentionalFactorizationMachineNet(nn.Module):
    """
    Attention layer for AFM. Model second order part = p^T * Sum(i,j)∈R (α_ij(v_i ⊙ v_j)x_ix_j + b),
    α_ij = h^T * ReLU(W(v_i ⊙ v_j)x_ix_j + b)
    """
    def __init__(self, embed_dim, attn_size, dropout):
        super().__init__()
        # inner product, interaction layer
        self.interact = EmbeddingsInteraction()
        # linear attention layer [embed_dim, attn_size]
        self.attention = nn.Linear(embed_dim, attn_size)
        # project attention layer [attn_size, 1]
        self.projection = nn.Linear(attn_size, 1)
        # linear layer [embed_dim, 1]
        self.p = nn.Linear(embed_dim, 1)
        # dropout parameter
        self.dropout = dropout

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embed_dim]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields, embed_dim] after embedding
        # interact -> [batch_size, num_fields * (num_fields) // 2, embed_dim]
        interactions = self.interact(x)
        # [batch_size, num_fields * (num_fields) // 2, embed_dim] * [embed_dim, attn_size]
        # -> [batch_size, num_fields * (num_fields) // 2, attn_size] -> relu
        attn_scores = F.relu(self.attention(interactions))
        # [batch_size, num_fields * (num_fields) // 2, attn_size] * [attn_size, 1]
        # -> [batch_size, num_fields * (num_fields) // 2, 1] -> softmax on feature cross dim
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        # dropout -> [batch_size, num_fields * (num_fields) // 2, 1], so far we got attention score α_ij
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)
        # [batch_size, num_fields * (num_fields) // 2, 1] * [batch_size, num_fields * (num_fields) // 2, embed_dim]
        # -> [batch_size, num_fields * (num_fields) // 2, embed_dim] -> sum -> [batch_size, embed_dim]
        # multipy each feature interaction attention score on each interaction and then sum cross all interactions
        attn_output = torch.sum(attn_scores * interactions, dim=1)
        # dropout -> [batch_size, embed_dim], so far we got attention score α_ij
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        # linear transfer layer p for output size [batch_size, embed_dim] * [embed_dim, 1] -> [batch_size, 1]
        attn_output = self.p(attn_output)

        return attn_output


class AttentionalFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropout):
        super().__init__()

        self.num_fields = len(field_dims)
        # bias parameter -> [1, ]
        self.w_0 = nn.Parameter(torch.zeros((1,)))
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # linear layer, same as embedding layer but output is 1, ex: [sum(fields_dims), 1]
        self.linear = FeaturesLinear(field_dims)
        # afm second order net, get attention score and multipy on feature cross
        self.afm = AttentionalFactorizationMachineNet(embed_dim, attn_size, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embeddings = self.embedding(x)
        # w_0 -> [1, ] + linear -> [batch_size, 1] + afm -> [batch_size, 1] -> [batch_size, 1]
        # sum three parts as output, bias + first order + second order
        output = self.w_0 + self.linear(x) + self.afm(embeddings)

        return torch.sigmoid(output)
