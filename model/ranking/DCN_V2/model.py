import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron, CrossNetworkMix


class DeepCrossNetworkMixModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network Mix.
    Same as DCN v1, only wide part cross net change w matrix.
    Original w is [input_dims, input_dims] size, but input_dims is embedding_output_dims, which is too large.
    Then we can use w = u * v^T, split into v, u two matrix with low rank dim projection,
    which can reduce lot parameters to train, and also keep close to original w matrix feature interaction information.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout, low_rank):
        super().__init__()
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for mlp and cross layer input
        self.embed_output_dim = len(field_dims) * embed_dim
        # crate cross layer, each layer have v, u -> [embed_output_dim, low_rank], b -> [embed_output_dim, ]
        self.cross = CrossNetworkMix(self.embed_output_dim, num_layers, low_rank)
        # deep layer dims, ex: [embed_output_dim, 256, 128, 64, 16], here we close output layer will add next layer
        self.deep = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        # add one linear layer for concat deep and cross for output, [mlp_dims[-1] + embed_output_dim, 1]
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        # view -> [batch_size, num_fields * embed_dim]
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        # cross -> [batch_size, num_fields * embed_dim] -> [batch_size, num_fields * embed_dim]
        cross_output = self.cross(embed_x)
        # deep -> mlp -> [batch_size, num_fields * embed_dim] -> [batch_size, mlp_dims[-1] ex: 16]
        deep_output = self.deep(embed_x)
        # horizontal stack -> [batch_size, num_fields * embed_dim + mlp_dims[-1] ex: 16]
        stacked = torch.cat([cross_output, deep_output], dim=1)
        # [batch_size, num_fields * embed_dim + mlp_dims[-1]] * [mlp_dims[-1] + embed_output_dim, 1] -> [batch_size, 1]
        output = self.linear(stacked)

        return torch.sigmoid(output)
