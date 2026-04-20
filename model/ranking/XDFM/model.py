import torch.nn as nn
import torch
import torch.nn.functional as F

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear


class CompressedInteractionNetwork(nn.Module):
    """
    CIN layer CNN way implemented.
    """
    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = nn.ModuleList()
        # init prev dim and CIN output dim which is fc input dim
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            # input CNN channel is input_dim * prev_dim, output dim is cross_layer_size
            self.conv_layers.append(nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                              stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            # next layer input channel h_k update, since x_0 dim is always input_dim not change
            prev_dim = cross_layer_size
            # sum each layer output dim, final linea layer will use each cross layer output as input
            fc_input_dim += prev_dim

        # linear layer [fc_input_dim, 1]
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embed_dim]
        :return: [batch_size, 1]
        """
        # save each cross layer output
        xs = []
        # x0 -> [batch_size, num_fields, 1, embed_dim], h -> [batch_size, num_fields, embed_dim]
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            # first h -> [batch_size, 1, num_fields, embed_dim]
            # second h -> [batch_size, 1, cross_layer_size // 2, embed_dim]
            # first x0 * h -> multipy -> [batch_size, num_fields, 1, embed_dim] * [batch_size, 1, num_fields, embed_dim]
            # -> [batch_size, num_fields, num_fields, embed_dim]
            # second -> [batch_size, num_fields, cross_layer_size // 2, embed_dim]
            # here is hadamard product, output keep embedd dim
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            # view -> [batch_size, num_fields * num_fields, embed_dim]
            # flatten each sample all h_k * m (h_k is k layer x, m is num_fields) into 1-dim
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            # conv1D -> [batch_size, num_fields * num_fields, embed_dim] -> [batch_size, cross_layer_size, embed_dim]
            # conv1D will weighted sum each embedd dim cross all flatten feature
            # like conv1D in nlp, input dim is sentence length here is flattened feature length ex: h_k * m
            # output will be number of cross_layer_size embedd dim vector vertical stacked
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                # if split, then x, h will be equal along cross_layer_size dim split,
                # x, h -> [batch_size, cross_layer_size, embed_dim] -> [batch_size, cross_layer_size // 2, embed_dim]
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            # save each cross layer output x, will be used in later fc linear layer
            xs.append(x)
        # cat -> [batch_size, cross_layer_size, embed_dim] + [batch_size, cross_layer_size // 2, embed_dim] on dim 1
        # -> [batch_size, cross_layer_size + ... + cross_layer_size // 2 , embed_dim]
        # sum -> [batch_size, cross_layer_size + ... + cross_layer_size // 2]
        # linear -> [batch_size, cross_layer_size + ... + cross_layer_size // 2] * [fc_input_dim, 1] -> [batch_size, 1]
        # fc_input_dim == sum of each layer output x cross layer output
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # embed output dim same as mlp input dime
        self.embed_output_dim = len(field_dims) * embed_dim
        # CIN layer
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        # mlp layer dims, ex: [embed_output_dim, 128, 64, 32, 16, 1]
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        # linear layer, same as embedding layer but output is 1, ex: [sum(fields_dims), 1]
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embedding(x) -> [batch_size, num_fields, embed_dim] ex: [64, 39, 4]
        embed_x = self.embedding(x)
        # linear -> [batch_size, 1] + cin -> [batch_size, 1] + view -> [batch_size, num_fields * embed_dim]
        # -> mlp -> [batch_size, mlp_dims[-1] ex: 1] -> [batch_size, 1]
        output = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(output)
