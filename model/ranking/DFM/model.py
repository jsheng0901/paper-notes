import torch.nn as nn
import torch

from notes.ranking.layer.layer import FeaturesLinear, FeaturesCross, FeaturesEmbedding, MultiLayerPerceptron


class DeepFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of DeepFM.
    From Wide & Deep model, use FM model for wide part. Deep and wide share same input embedding.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # linear layer, same as embedding layer but output is 1, ex: [sum(fields_dims), 1]
        self.linear = FeaturesLinear(field_dims)
        # feature interaction layer same as FM cross layer
        self.fm = FeaturesCross(reduce_sum=True)
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # embed output dim same as mlp input dime
        self.embed_output_dim = len(field_dims) * embed_dim
        # mlp layer dims, ex: [embed_output_dim, 128, 64, 32, 16, 1]
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embedding(x) -> [batch_size, num_fields, embed_dim]
        embeddings = self.embedding(x)
        # linear -> [batch_size, 1] + fm -> [batch_size, 1] + view -> [batch_size, num_fields * embed_dim]
        # -> mlp -> [batch_size, mlp_dims[-1] ex: 1] -> [batch_size, 1]
        output = self.linear(x) + self.fm(embeddings) + self.mlp(embeddings.view(-1, self.embed_output_dim))

        return torch.sigmoid(output)
