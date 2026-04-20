import torch.nn as nn
from notes.ranking.layer.layer import FeaturesCross, FeaturesLinear, FeaturesEmbedding
import torch


class FactorizationMachine(nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    This version linear and cross layer use embedding layer before pass into model.
    Input id is one-hot encoding id, so get input embedding output equal to feature * weight.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear_layer = FeaturesLinear(field_dims)
        self.cross_layer = FeaturesCross(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embeddings = self.embedding(x)
        # linear -> [batch_size, 1]
        linear = self.linear_layer(x)
        # cross -> [batch_size, 1]
        cross = self.cross_layer(embeddings)
        # output [batch_size, 1]
        output = linear + cross

        # apply sigmoid to transfer to probability, no squeeze here since target is same [batch_size, 1] size
        return torch.sigmoid(output)
