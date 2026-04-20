import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron


class GateCrossNetwork(nn.Module):
    """
    Here we inherit from DCN-V2 apply low rank projection implementation. No MoE logic in here.
    """
    def __init__(self, input_dim, num_layers, low_rank):
        super().__init__()
        self.num_layers = num_layers
        # each cross layer have one low rank projection linear layer -> [input_dim, low_rank] without bias
        # input_dim will be embed output dims
        self.v = nn.ModuleList([
            nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_layers)
        ])
        # each cross layer have one projection back linear layer -> [low_rank, input_dim] without bias
        self.u = nn.ModuleList([
            nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_layers)
        ])
        # each cross layer have one bias -> [input_dim, ]
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])
        # each cross layer have one gate matrix -> [input_dim, input_dim] without bias
        self.w_gate = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)
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
            # num_fields * embed_dim == embed_output_dim = input_dim, projection into low rank dims
            # [batch_size, num_fields * embed_dim] * [input_dim, low_rank] -> [batch_size, low_rank]
            v_x = self.v[i](x)
            # nonlinear activation in low rank space, tanh -> [batch_size, low_rank]
            # here we not implement with MoE, not really need activation
            v_x = torch.tanh(v_x)
            # projection from low rank dims back to input dims
            # [batch_size, low_rank] * [low_rank, input_dim] -> [batch_size, input_dim = num_fields * embed_dim]
            uv_x = self.u[i](v_x)
            # add information gate matrix to filter out input x noise
            # [batch_size, num_fields * embed_dim] * [input_dim, input_dim] -> [batch_size, input_dim]
            gate_x = torch.sigmoid(self.w_gate[i](x))
            # [batch_size, num_fields * embed_dim] + [input_dim, ] -> [batch_size, num_fields * embed_dim]
            # x_0 * x_w * gate_x: [batch_size, num_fields * embed_dim] * [batch_size, num_fields * embed_dim]
            # -> [batch_size, num_fields * embed_dim] Hadamard-product keep same dims
            # -> [batch_size, num_fields * embed_dim] gate_w same as Hadamard-product keep same dims
            # [batch_size, num_fields * embed_dim] + [batch_size, num_fields * embed_dim]
            # -> [batch_size, num_fields * embed_dim]
            x = x_0 * (uv_x + self.b[i]) * gate_x + x

        return x


class GateDeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Gate Deep & Cross Network.
    Same as DCN v2, only wide part add gate w matrix.
    We can do the same lower rank on gate w, but from original paper not use.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout, low_rank):
        super().__init__()
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for mlp and cross layer input
        self.embed_output_dim = len(field_dims) * embed_dim
        # crate cross layer, each layer have v, u -> [embed_output_dim, low_rank], b -> [embed_output_dim, ]
        self.cross = GateCrossNetwork(self.embed_output_dim, num_layers, low_rank)
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
