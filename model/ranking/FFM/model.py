import torch.nn as nn
from notes.ranking.layer.layer import FeaturesEmbedding, FeaturesLinear
import torch


class FieldAwareCross(nn.Module):
    """
    Field feature cross layer, output = sum_i(sum_j(<v_i_fj, v_j_fi> xi, xj))
    Calculate each feature cross interaction weight.
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # get number of fields which same as number of features
        self.num_fields = len(field_dims)
        # build embedding layer for each field, ex: [sum(fields_dims), embed_dim] * num_fields
        self.embeddings_layer = torch.nn.ModuleList([
            FeaturesEmbedding(field_dims, embed_dim) for _ in range(self.num_fields)
        ])

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embedding -> [batch_size, num_fields, embedding_dim] -> [4096, 39, 10]
        # embeddings -> [batch_size, num_fields, embedding_dim] * num_fields
        # each element will be calculate all sample in batch each feature to ith field embedding
        embeddings = [self.embeddings_layer[i](x) for i in range(self.num_fields)]

        # calculate <v_i_fj, v_j_fi> xi, xj among all fields
        # here only (num_fields - 1 + num_fields - 2 + num_fields - 3 + ... + 1) interaction
        # since feature don't need to interact with themselves and interaction pair only show once
        interaction = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                # embeddings[j][:, i] means all sample in this batch feature i to field j -> same as v_i_fj
                # embeddings[i][:, j] means all sample in this batch feature j to field i -> same as v_j_fi
                # [batch_size, 1, embedding_dim] * [batch_size, 1, embedding_dim] -> [batch_size, 1, embedding_dim]
                interaction.append(embeddings[j][:, i] * embeddings[i][:, j])
        # stack all outputs on feature dims
        # means for each sample stack all feature interaction output, each output is embedding dim
        # stack -> [batch_size, 1, embedding_dim] -> [batch_size, (num_fields - 1 + ... + 1), embedding_dim]
        interaction = torch.stack(interaction, dim=1)
        # sum through interaction and then through embedding dim
        # sum -> [batch_size, (num_fields - 1 + ... + 1), embedding_dim] -> [batch_size, embedding_dim]
        # sum -> [batch_size, embedding_dim] -> [batch_size, 1], here need keep same dim for calculate loss
        output = torch.sum(torch.sum(interaction, dim=1), dim=1, keepdim=True)
        return output


class FieldAwareFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of Field Aware Factorization Machine.
    This version linear and cross layer use embedding layer before pass into model.
    Input id is one-hot encoding id, so get input embedding output equal to feature * weight.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear_layer = FeaturesLinear(field_dims)
        self.cross_layer = FieldAwareCross(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # linear -> [batch_size, 1]
        linear = self.linear_layer(x)
        # cross -> [batch_size, 1]
        cross = self.cross_layer(x)
        # output [batch_size, 1]
        output = linear + cross

        # apply sigmoid to transfer to probability, no squeeze here since target is same [batch_size, 1] size
        return torch.sigmoid(output)
