import itertools

import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron


class SENETLayer(nn.Module):
    """SENETLayer used in FiBiNET.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **reduction_ratio** : Positive integer, dimensionality of the attention network output space.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, reduction_ratio=3):
        super().__init__()
        self.filed_size = filed_size
        # reduction size, size = f // r
        self.reduction_size = max(1, self.filed_size // reduction_ratio)
        # two mlp layers, both no bias
        self.excitation = nn.Sequential(
            # linear layer -> [filed_size, reduction_size]
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            # linear layer -> [reduction_size, filed_size]
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_size]
        :return: [batch_size, num_fields, embedding_size]
        """
        # x -> [batch_size, num_fields, embedding_size] ex: [64, 39, 8]
        # mean -> [batch_size, num_fields] ex: [64, 39]
        # mean will sum all information for each feature (field)
        Z = torch.mean(x, dim=-1, out=None)
        # [batch_size, num_fields] * [num_fields, reduction_size] * [reduction_size, num_fields]
        # -> [batch_size, num_fields] ex: [64, 39] * [39, 18] * [18, 39] -> [64, 39]
        # first mlp will calculate each features interaction weight, second mlp transfer to same x dims
        A = self.excitation(Z)
        # A -> [batch_size, num_fields, 1], x mul A -> [batch_size, num_fields, embedding_size]
        # each feature (field) times attention weight from two mlp layers output
        V = torch.mul(x, torch.unsqueeze(A, dim=2))

        return V


class BilinearInteraction(nn.Module):
    """BilinearInteraction Layer used in FiBiNET.
          Arguments
            - **filed_size** : Positive integer, number of feature groups.
            - **embedding_size** : Positive integer, embedding size of sparse features.
            - **bilinear_type** : String, types of bilinear functions used in this layer.
          References
            - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """
    def __init__(self, filed_size, embedding_size, bilinear_type='interaction'):
        super().__init__()
        self.filed_size = filed_size
        # three different bilinear types
        self.bilinear_type = bilinear_type
        # record bilinear layer
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == 'all':
            # all feature share one weight matrix
            # linear -> [embedding_size, embedding_size]
            self.bilinear = nn.Linear(embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == 'each':
            # each feature embedding use one weight matrix
            # each layer is linear -> [embedding_size, embedding_size]
            for _ in range(self.filed_size):
                self.bilinear.append(nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == 'interaction':
            # each feature interaction use one weight matrix, we have num_fields * (num_fields - 1) / 2 interaction
            # each layer is linear -> [embedding_size, embedding_size]
            for _, _ in itertools.combinations(range(self.filed_size), 2):
                self.bilinear.append(nn.Linear(embedding_size, embedding_size, bias=False))

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_size]
        :return: [batch_size, num_fields * (num_fields - 1) / 2, embedding_size]
        """
        # x -> [batch_size, num_fields, embedding_size]
        # split -> list([batch_size, embedding_size] * num_fields)
        x = torch.split(x, 1, dim=1)
        # all feature share one weight matrix
        if self.bilinear_type == 'all':
            # combinations -> all x list combination 2 elements among fields dim
            # which have num_fields * (num_fields - 1) / 2 number of combination subset
            # v_i, v_j -> [batch_size, embedding_size]
            # bilinear * v_i -> [batch_size, embedding_size] * [embedding_size, embedding_size]
            # -> [batch_size, embedding_size]
            # [batch_size, embedding_size] mul [batch_size, embedding_size] -> [batch_size, embedding_size]
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(x, 2)]
        # each feature embedding use one weight matrix
        elif self.bilinear_type == 'each':
            # combinations -> all x list combination 2 elements among fields dim
            # which have num_fields * (num_fields - 1) / 2 number of combination subset
            # x[i], x[j] -> [batch_size, embedding_size]
            # bilinear[i] * x[i] -> [batch_size, embedding_size] * [embedding_size, embedding_size]
            # -> [batch_size, embedding_size]
            # [batch_size, embedding_size] mul [batch_size, embedding_size] -> [batch_size, embedding_size]
            p = [torch.mul(self.bilinear[i](x[i]), x[j])
                 for i, j in itertools.combinations(range(len(x)), 2)]
        # each feature interaction use one weight matrix
        elif self.bilinear_type == 'interaction':
            # combinations -> all x list combination 2 elements among fields dim
            # which have num_fields * (num_fields - 1) / 2 number of combination subset
            # v -> ([batch_size, embedding_size], [batch_size, embedding_size]), v[0] -> [batch_size, embedding_size]
            # bilinear * v[0] -> [batch_size, embedding_size] * [embedding_size, embedding_size]
            # -> [batch_size, embedding_size]
            # [batch_size, embedding_size] mul [batch_size, embedding_size] -> [batch_size, embedding_size]
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(x, 2), self.bilinear)]

        # cat -> list([batch_size, embedding_size] * (num_fields * (num_fields - 1) / 2))
        # among first dim -> [batch_size, num_fields * (num_fields - 1) / 2, embedding_size]
        return torch.cat(p, dim=1)


class FiBiNETModel(nn.Module):
    """
    A pytorch implementation of FiBiNET. Here we ignore dense feature linear layer and treat as sparse feature.
    """
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, reduction_ratio, bilinear_type):
        super().__init__()
        self.field_size = len(field_dims)
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # SENET layer -> [batch_size, num_fields, embedding_size]
        self.SE = SENETLayer(self.field_size, reduction_ratio)
        # Bilinear layer -> [batch_size, num_fields * (num_fields - 1) / 2, embedding_size]
        self.Bilinear = BilinearInteraction(self.field_size, embed_dim, bilinear_type)
        # two embedding concat output dims will be mlp layer input
        self.embed_output_dim = self.field_size * (self.field_size - 1) * embed_dim
        # deep layer dims, ex: [embed_output_dim, 256, 128, 64, 16, 1], here we have output layer
        self.dnn = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embed_x = self.embedding(x)
        # get senet output, calculate each feature weight
        # [batch_size, num_fields, embedding_size] -> SE -> [batch_size, num_fields, embedding_size]
        senet_output = self.SE(embed_x)
        # get bilinear output for weighted embedding
        # [batch_size, num_fields, embedding_size] -> [batch_size, num_fields * (num_fields - 1) / 2, embedding_size]
        senet_bilinear_out = self.Bilinear(senet_output)
        # get bilinear output for original embedding
        # [batch_size, num_fields, embedding_size] -> [batch_size, num_fields * (num_fields - 1) / 2, embedding_size]
        bilinear_out = self.Bilinear(embed_x)
        # horizontal stack -> [batch_size, num_fields * (num_fields - 1), embedding_size]
        # view -> [batch_size, num_fields * (num_fields - 1) * embedding_size]
        dnn_input = torch.cat([senet_bilinear_out, bilinear_out], dim=1).view(-1, self.embed_output_dim)
        # mlp -> [batch_size, num_fields * (num_fields - 1) * embedding_size] -> [batch_size, 1]
        output = self.dnn(dnn_input)

        return torch.sigmoid(output)
