import torch.nn as nn
import torch

from notes.ranking.layer.layer import EmbeddingsInteraction, FeaturesEmbedding, MultiLayerPerceptron


class InnerProduct(nn.Module):
    """
    Inner Product Layer. Same as embedding product cross each filed (feature)
    """
    def __init__(self):
        super().__init__()
        # get embedding interaction layer
        self.interaction = EmbeddingsInteraction()

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_dim]
        :return: [batch_size, num_fields * (num_fields - 1) // 2]
        """
        # x -> [batch_size, num_fields, embedding_dim]
        # interaction -> [batch_size, num_fields * (num_fields - 1) // 2, embedding_dim]
        # sum -> [batch_size, num_fields * (num_fields - 1) // 2]
        # each sample feature interaction with all others feature and then sum cross on all embedding dims
        p = torch.sum(self.interaction(x), dim=2)

        return p


class OuterProduct(nn.Module):
    """
    Outer Product Layer. Same as field (feature) product cross each embedding
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_dim]
        :return: [batch_size, field_dims^2]
        """
        field_dims = x.shape[2]

        # each sample sum over all fields in paper is sum pooling
        # x -> [batch_size, num_fields, embedding_dim] -> sum -> [batch_size, embedding_dim]
        sum_f = x.sum(dim=1)

        # loop through all field dims combination also as embedding dims.
        # each embedding dims need calculate outer product with all other embedding dims cross different filed (feature)
        row, col = [], []
        for i in range(field_dims):
            for j in range(field_dims):
                row.append(i)
                col.append(j)
        # x1: [batch_size, 1] * x2: [batch_size, 1] -> [batch_size, 1]
        # we have field_dims * field_dims combination, mul will keep same output shape as input
        # then we have [batch_size, field_dims^2] output
        p = torch.mul(sum_f[:, row], sum_f[:, col])

        return p


class ProductNeuralNetworkModel(nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.
    """
    def __init__(self, field_dims, mlp_dims, embed_dim=4, dropout=0.2, method='inner'):
        super().__init__()
        mlp_input_size = 0
        num_fields = len(field_dims)
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embed = FeaturesEmbedding(field_dims, embed_dim)

        if method == 'inner':
            # use inner product layer
            self.pn = InnerProduct()
            # mlp input dims = embed layer output dims + inner product layer output dims
            mlp_input_size = num_fields * embed_dim + num_fields * (num_fields - 1) // 2
        elif method == 'outer':
            # use outer product layer
            self.pn = OuterProduct()
            # mlp input dims = embed layer output dims + outer product layer output dims
            mlp_input_size = num_fields * embed_dim + embed_dim ** 2

        # bias layer will pass into mlp input, ex: [num_fields * embed_dim, ]
        self.bias = nn.Parameter(torch.zeros((num_fields * embed_dim,)))
        # init bias layer
        nn.init.xavier_uniform_(self.bias.unsqueeze(0).data)

        # mlp layer dims, ex: [mlp_input_size, 256, 256, 128, 1]
        self.mlp = MultiLayerPerceptron(mlp_input_size, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        x = self.embed(x)
        # x -> reshape -> [batch_size, num_fields * embed_dim]
        # each sample concat all fields embed dims into one vector, for later add bias
        z = x.reshape(x.shape[0], -1)
        # inner: [batch_size, num_fields * (num_fields - 1) // 2]
        # outer: [batch_size, field_dims^2]
        p = self.pn(x)

        # mlp input: l1 = relu(z, p, bias)
        # z + bias -> [batch_size, num_fields * embed_dim] + [batch_size, num_fields * embed_dim]
        # cat -> [batch_size, num_fields * embed_dim + (num_fields * (num_fields - 1) // 2) or (field_dims^2)]
        output = torch.cat([z + self.bias, p], dim=1)
        # [batch_size, mlp_input_size] -> [batch_size, 1]
        output = self.mlp(output)

        return torch.sigmoid(output)
