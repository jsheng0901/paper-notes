import torch
import torch.nn as nn


class FactorizationMachineLinearLayer(nn.Module):
    """
    Linear layer of FM model. output = w_i * x + w_0
    """

    def __init__(self, n_features):
        super().__init__()
        # simpler linear layer -> [n_features, 1]
        # bias -> [n_samples, ]
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        # x -> [n_samples, n_features]
        # x * linear + bias -> [n_samples, n_features] * [n_features, 1] + [n_samples, ] -> [n_samples, ]
        return self.linear(x)


class FactorizationMachineCrossLayer(nn.Module):
    """
    Cross layer of FM model. output = 0.5 * sum_f((sum_i(vi_f * x_i))^2 - sum_i(vi_f^2 * x_i^2))
    Calculate each feature cross interaction weight.
    """

    def __init__(self, n_features, k_dims):
        super().__init__()
        # cross layer for calculate vi_k -> [n_features, k]
        self.cross = nn.Parameter(torch.empty(n_features, k_dims), requires_grad=True)
        # init parameter data
        # according to doc, need input shape as [fan_out, fan_in], so we need transposed weight matrix
        torch.nn.init.xavier_uniform(self.cross.T)

    def forward(self, x):
        # x -> [n_samples, n_features]
        # (vi_f * x_i)^2 -> x * cross -> ([n_samples, n_features] * [n_features, k])^2 -> [n_samples, k]^2
        square_of_sum = torch.pow(x @ self.cross, 2)
        # (vi_f^2 * x_i^2) -> x^2 * cross^2 -> ([n_samples, n_features])^2 * ([n_features, k])^2 -> [n_samples, k]
        sum_of_square = torch.pow(x, 2) @ torch.pow(self.cross, 2)
        # each sample sum along k dims, sum([n_samples, k] - [n_samples, k]) -> [n_samples, 1]
        output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=-1, keepdim=True)

        return output


class FactorizationMachineModel(nn.Module):
    def __init__(self, n_features, k_dims):
        super().__init__()

        self.linear_layer = FactorizationMachineLinearLayer(n_features)
        self.cross_layer = FactorizationMachineCrossLayer(n_features, k_dims)

    def forward(self, x):
        # x -> [n_samples, n_features]
        # linear -> [n_samples, ]
        linear = self.linear_layer(x)
        # cross -> [n_samples, ]
        cross = self.cross_layer(x)
        # output [n_samples, 1]
        output = linear + cross

        return output.view(-1, 1)
