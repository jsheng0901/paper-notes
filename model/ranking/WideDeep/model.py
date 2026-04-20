import torch.nn as nn
import torch

from notes.ranking.layer.layer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron


class WideAndDeepModel(nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Wide design for memorization, and deep design for generalization.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # wide linear layer, build general linear model
        self.wide = FeaturesLinear(field_dims)
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for mlp layer input
        self.embed_output_dim = len(field_dims) * embed_dim
        # deep layer dims, ex: [embed_output_dim, 256, 128, 64, 16, 1]
        self.deep = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embed_x = self.embedding(x)
        # view -> [batch_size, num_fields, embed_dim] -> [batch_size, num_fields * embed_dim]
        # wide -> [batch_size, 1] + deep -> mlp -> [batch_size, num_fields * embed_dim] -> [batch_size, 1]
        output = self.wide(x) + self.deep(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(output)