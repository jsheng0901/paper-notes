import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron


class CrossNetwork(nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # each cross layer have one linear layer -> [input_dim, 1] without bias, input_dim will be embed output dims
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        # each cross layer have one bias -> [input_dim, ]
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: [batch_size, num_fields * embed_dim]
        :return: [batch_size, num_fields * embed_dim]
        """
        # x -> [batch_size, num_fields * embed_dim]
        # save init input x value to make sure after cross many layer not far away from init input x
        x_0 = x
        # create cross layer
        for i in range(self.num_layers):
            # num_fields * embed_dim == embed_output_dim = input_dim
            # [batch_size, num_fields * embed_dim] * [input_dim, 1] -> [batch_size, 1]
            x_w = self.w[i](x)
            # x_0 * x_w: [batch_size, num_fields * embed_dim] * [batch_size, 1] -> [batch_size, num_fields * embed_dim]
            # [batch_size, num_fields * embed_dim] + [input_dim, ] -> [batch_size, num_fields * embed_dim]
            # [batch_size, num_fields * embed_dim] + [batch_size, num_fields * embed_dim]
            x = x_0 * x_w + self.b[i] + x

        return x


class DeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Same as wide and deep model, only wide part become to cross layer, cross layer like FM model logic,
    have many feature interaction calculation.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for mlp and cross layer input
        self.embed_output_dim = len(field_dims) * embed_dim
        # crate cross layer, each layer have w -> [embed_output_dim, 1], b -> [embed_output_dim, ]
        self.cross = CrossNetwork(self.embed_output_dim, num_layers)
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
