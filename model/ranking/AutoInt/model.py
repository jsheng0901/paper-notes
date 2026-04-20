import torch
import torch.nn as nn
import torch.nn.functional as F

from notes.ranking.layer.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class AutomaticFeatureInteractionModel(nn.Module):
    """
    A pytorch implementation of AutoInt.
    """

    def __init__(self, field_dims, embed_dim, atten_embed_dim, num_heads, num_layers,
                 mlp_dims, dropouts, has_residual=True):
        super().__init__()
        # number of features
        self.num_fields = len(field_dims)
        # linear layer, same as embedding layer but output is 1, ex: [sum(fields_dims), 1]
        self.linear = FeaturesLinear(field_dims)
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # attention embedding layer -> [embed_dim, atten_embed_dim]
        self.atten_embedding = nn.Linear(embed_dim, atten_embed_dim)
        # embed_x output after flatten shape, number of features * embed_dim
        self.embed_output_dim = len(field_dims) * embed_dim
        # interacting layer output after flatten shape, number of features * atten_embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        # use residual connection or not
        self.has_residual = has_residual
        # mlp layer dims, ex: [embed_output_dim, 128, 64, 32, 16, 1]
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])
        # multi heads self attention layers, here we use pytorch version
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        # linear layer after interacting layer, [number of features * atten_embed_dim, 1]
        self.attn_fc = nn.Linear(self.atten_output_dim, 1)
        # apply residual connection
        if self.has_residual:
            # linear layer use as residual connection input embed [embed_dim, atten_embed_dim]
            self.res_embedding = nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embed_x = self.embedding(x)
        # apply attention embedding after feature embedding
        # [batch_size, num_fields, embed_dim] * [embed_dim, atten_embed_dim] = [batch_size, num_fields, atten_embed_dim]
        atten_x = self.atten_embedding(embed_x)
        # [batch_size, num_fields, atten_embed_dim] -> transpose -> [num_fields, batch_size, atten_embed_dim]
        cross_term = atten_x.transpose(0, 1)
        for self_attn in self.self_attns:
            # since batch fist set as false, so input will be [length, batch, embed_dim] which is cross term as q, k ,v
            # [num_fields, batch_size, atten_embed_dim] -> self_atten -> [num_fields, batch_size, atten_embed_dim]
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        # [num_fields, batch_size, atten_embed_dim] -> transpose -> [batch_size, num_fields, atten_embed_dim]
        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            # [batch_size, num_fields, embed_dim] * [embed_dim, atten_embed_dim]
            # -> [batch_size, num_fields, atten_embed_dim], make sure residual connection input sharp sample as cross
            res = self.res_embedding(embed_x)
            # residual connection multi head output + embed_x to keep original x value information
            # [batch_size, num_fields, atten_embed_dim] += [batch_size, num_fields, atten_embed_dim]
            cross_term += res
        # here cross_term is interacting layer output
        # [batch_size, num_fields, atten_embed_dim] -> relu -> [batch_size, num_fields, atten_embed_dim]
        # contiguous will optimize data save in memory after data apply view or transpose
        # [batch_size, num_fields, atten_embed_dim] -> view -> [batch_size, num_fields * atten_embed_dim]
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        # concat linear(original x) + linear(cross_term) + mlp three part output as final output
        # linear original x layer keep lower interaction information, cross and mlp keep high interaction information
        # x -> [batch_size, num_fields], linear(x) -> [batch_size, 1]
        # attn_fc -> [batch_size, num_fields * atten_embed_dim] * [num_fields * atten_embed_dim, 1] -> [batch_size, 1]
        # embed_x -> [batch_size, num_fields, embed_dim] -> view -> [batch_size, num_fields * embed_dim]
        # mlp -> [batch_size, num_fields * embed_dim] * [num_fields * embed_dim, mlp[0]]
        # -> [mlp[1], 1] -> [batch_size, 1]
        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(x)
