import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding


class ResidualUnit(nn.Module):
    """
    Single residual unit, contains two linear layers and two activation layers.
    l1 = relu(w_0 * x + b_0), l2 = relu(w_1 * l1 + b_1 + x)
    """
    def __init__(self, input_size):
        super().__init__()
        # linear layer [input_size, input_size], input output size same
        self.fc1 = nn.Linear(input_size, input_size)
        # linear layer [input_size, input_size]
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields * embed_dims]
        :return: [batch_size, input_size]
        """
        # x -> [batch_size, num_fields * embed_dims], here input_size == num_fields * embed_dims
        # [batch_size, num_fields * embed_dims] * [input_size, input_size] -> [batch_size, input_size]
        output = self.fc1(x)
        # relu -> [batch_size, input_size]
        output = torch.relu(output)
        # [batch_size, input_size] * [input_size, input_size] -> [batch_size, input_size]
        output = self.fc2(output)
        # [batch_size, input_size] + [batch_size, num_fields * embed_dims] -> [batch_size, input_size]
        output = output + x
        # relu -> [batch_size, input_size]
        output = torch.relu(output)

        return output


class DeepCrossingModel(nn.Module):
    """
    A pytorch implementation of deep crossing model.
    Deep crossing use residual connect unit to capture information between input and output information.
    """
    def __init__(self, field_dims, embed_dim=4, num_res=1):
        super().__init__()

        # get residual input size which is equal after embedding layer, concat all field embed into one vector
        residual_input_size = len(field_dims) * embed_dim
        # residual layer, sequential of [input_size, input_size] linear layer
        self.res = nn.Sequential(*[ResidualUnit(residual_input_size) for _ in range(num_res)])
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embed = FeaturesEmbedding(field_dims, embed_dim)
        # output linear layer, [residual_input_size, 1]
        self.fc = nn.Linear(residual_input_size, 1)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        x = self.embed(x)
        # x -> reshape -> [batch_size, num_fields * embed_dim]
        x = x.reshape(x.shape[0], -1)
        # [batch_size, num_fields * embed_dim] * [input_size, input_size] ... -> [batch_size, input_size]
        x = self.res(x)
        # [batch_size, input_size] * [residual_input_size, 1] -> [batch_size, 1]
        output = self.fc(x)

        return torch.sigmoid(output)
