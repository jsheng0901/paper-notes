import torch.nn as nn
import torch

from notes.ranking.layer.layer import FeaturesLinear, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron, \
    FeaturesCross


class NeuralFactorizationMachineModelV1(nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Model use FM logic with Bi-Interaction Pooling layer as intput then MLP layer.
    Same as wide and deep model, only deep part input become to cross layer, cross layer like FM model logic,
    Here use PNN inner product layer, which take longer in creating feature combination.
    """
    def __init__(self, field_dims, mlp_dims, dropout, embed_dim):
        super().__init__()

        # bias parameter -> [1, ]
        self.w_0 = nn.Parameter(torch.zeros((1,)))
        # linear layer, same as embedding layer but output is 1, ex: [sum(fields_dims), 1]
        self.linear = FeaturesLinear(field_dims)
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # feature interaction layer same as PNN inner method
        self.interaction = EmbeddingsInteraction()
        # mlp layer dims, ex: [embed_dim, 256, 128, 64, 16, 1]
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embeddings = self.embedding(x)
        # [batch_size, num_fields, embed_dim] -> interact -> [batch_size, num_fields*(num_fields)//2, embed_dim]
        # sum -> [batch_size, num_fields*(num_fields)//2, embed_dim] -> [batch_size, embed_dim]
        # each sample sum across all feature interaction dims
        bi_output = self.interaction(embeddings).sum(dim=1)
        # mlp -> [batch_size, embed_dim] -> [batch_size, mlp_dims[-1] ex: 1]
        f_output = self.mlp(bi_output)
        # w_0 -> [1, ] + linear -> [batch_size, 1] + f -> [batch_size, mlp_dims[-1] ex: 1] -> [batch_size, 1]
        output = self.w_0 + self.linear(x) + f_output

        return torch.sigmoid(output)


class NeuralFactorizationMachineModelV2(nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Here use FM cross layer which take linear time much faster than PNN inner method.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # bias parameter -> [1, ]
        self.w_0 = nn.Parameter(torch.zeros((1,)))
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # linear layer, same as embedding layer but output is 1, ex: [sum(fields_dims), 1]
        self.linear = FeaturesLinear(field_dims)
        # feature interaction layer same as FM cross layer
        self.fm = torch.nn.Sequential(
            FeaturesCross(reduce_sum=False),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout)
        )
        # mlp layer dims, ex: [embed_dim, 128, 64, 32, 16, 1]
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embeddings = self.embedding(x)
        # [batch_size, num_fields, embed_dim] -> FM cross layer no reduce sum -> [batch_size, embed_dim]
        # each sample same across all feature interaction dims in FM cross layer
        cross_term = self.fm(embeddings)
        # mlp -> [batch_size, embed_dim] -> [batch_size, mlp_dims[-1] ex: 1]
        f_output = self.mlp(cross_term)
        # w_0 -> [1, ] + linear -> [batch_size, 1] + f -> [batch_size, mlp_dims[-1] ex: 1] -> [batch_size, 1]
        output = self.w_0 + self.linear(x) + f_output

        return torch.sigmoid(output)
