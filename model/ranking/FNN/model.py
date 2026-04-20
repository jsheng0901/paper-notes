import torch.nn as nn
from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron
import torch


class FactorizationSupportedNeuralNetworkModel(nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Use FM model get linear and cross embedding output, then stack together as input to mlp layer.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # FM linear embedding, embedding -> [sum(field_dims), 1] will be assign by linear layer after FM model
        self.linear_embedding = FeaturesEmbedding(field_dims, 1)
        # FM cross embedding, embedding -> [sum(field_dims), embed_dim] will be assign by cross layer after FM model
        self.cross_embedding = FeaturesEmbedding(field_dims, embed_dim)
        # after FM model, mlp embedding input, here need add 1 since we stack cross and linear embedding output
        self.embed_output_dim = len(field_dims) * (embed_dim + 1)
        # mlp layer, input will be FM linear and cross stack output
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields] -> [batch_size, num_fields, 1] -> squeeze -> [batch_size, num_fields]
        w = self.linear_embedding(x).squeeze(-1)
        # x -> [batch_size, num_fields] -> [batch_size, num_fields, embed_dim] -> [batch_size, num_fields * embed_dim]
        # transfer each sample to one embedding vector by horizontal concat all features embedding
        v = self.cross_embedding(x).reshape(x.shape[0], -1)
        # stack on dim 1 by horizontal concat both linear and cross weight output
        # stack(w, v) -> [batch_size, num_fields * (embed_dim + 1)]
        stacked = torch.hstack([w, v])

        output = self.mlp(stacked)

        return torch.sigmoid(output)
