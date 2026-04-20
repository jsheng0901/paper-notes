import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, Dice


class ActivationUnit(nn.Module):
    """
    Activation unit, each ad will have different attention score on all behaviors
    formular: sum(i, n)(g(v_i, v_a) * v_a), v_a -> ad embedd, v_i -> behavior embedd
    """
    def __init__(self, embed_dim, activation_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            # mlp layer -> [embed_dim * (embed_dim + 2), activation_dim]
            nn.Linear(embed_dim * (embed_dim + 2), activation_dim),
            # activation layer not change shape, [embed_dim * (embed_dim + 2), activation_dim]
            Dice(),
            # mlp layer -> [activation_dim, 1]
            nn.Linear(activation_dim, 1),
        )

    def forward(self, x):
        """
        :param x: [batch_size, num_behaviors + 1, embedding_dim]
        :return: [batch_size, num_behaviors, 1]
        """
        # x -> [batch_size, num_behaviors + 1, embedding_dim]
        # behaviors -> [batch_size, num_behaviors, embedding_dim]
        # extract all history behaviors embedd, ex: [64, 40, 8], same as v_i in formular
        behaviors = x[:, :-1]
        num_behaviors = behaviors.shape[1]
        # ads -> [batch_size, num_behaviors, embedding_dim], same as v_a in formular
        # extract last field ads embedd and repeat number of behaviors times, ex: [64, 40, 8], each 40 embedd is same
        ads = x[:, [-1] * num_behaviors]

        # outer product
        embed_dim = x.shape[-1]
        i1, i2 = [], []
        for i in range(embed_dim):
            for j in range(embed_dim):
                i1.append(i)
                i2.append(j)

        # behaviors: [batch_size, num_behaviors, 1] * ads: [batch_size, num_behaviors, 1]
        # -> [batch_size, num_behaviors, 1] we have embed_dim^2 combination
        # mul will keep same output shape as input then we have [batch_size, num_behaviors, embed_dim^2] output
        # -> reshape -> [batch_size, num_behaviors, embed_dim^2]
        # here use outer product, each behavior will multipy ad in all embedd dim combination
        p = behaviors[:, :, i1].mul(ads[:, :, i2]).reshape(behaviors.shape[0], behaviors.shape[1], -1)

        # cat -> [batch_size, num_behaviors, embed_dim] + [batch_size, num_behaviors, embed_dim^2
        # + [batch_size, num_behaviors, embed_dim] -> [batch_size, num_behaviors, embed_dim^2 + 2 * embed_dim]
        # -> [batch_size, num_behaviors, embed_dim * (2 + embed_dim)]
        # mlp -> [batch_size, num_behaviors, embed_dim * (embed_dim + 2)]
        # * [embed_dim * (embed_dim + 2), activation_dim] * [activation_dim, 1] -> [batch_size, num_behaviors, 1]
        # here use mlp layer to calculate behaviors, product, ads cross interaction raw attention score
        att = self.mlp(torch.cat([behaviors, p, ads], dim=2))

        return att


class DeepInterestNetworkModel(nn.Module):
    """
    A pytorch implementation of DIN.
    """
    def __init__(self, field_dims, embed_dim, activation_dim, mlp_dims):
        super().__init__()
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # attention layer
        self.attention = ActivationUnit(embed_dim, activation_dim)
        # mlp layer
        self.mlp = nn.Sequential(
            # input will be user interest and behavior horizontal concat result
            # mlp layer -> [embed_dim * 2, mlp_dims[0] ex: 200]
            nn.Linear(embed_dim * 2, mlp_dims[0]),
            Dice(),
            # mlp layer -> [mlp_dims[0] ex: 200, mlp_dims[1] ex: 100]
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            Dice(),
            # mlp layer -> [mlp_dims[1] ex: 100, 1]
            nn.Linear(mlp_dims[1], 1)
        )

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # create mask for padding behaviors and add one dim for embedding multipy -> [batch_size, num_behaviors + 1, 1]
        mask = (x > 0).float().unsqueeze(-1)
        # embedding(x) -> [batch_size, num_behaviors + 1, embed_dim] ex: [64, 40 + 1, 8]
        # mul [batch_size, num_behaviors + 1, 1] -> [batch_size, num_behaviors + 1, embed_dim]
        behaviors_ad_embeddings = self.embedding(x).mul(mask)
        # [batch_size, num_behaviors + 1, embed_dim] -> attention -> [batch_size, num_behaviors, 1]
        att = self.attention(behaviors_ad_embeddings)

        # raw attention score multipy all behaviors embedd to get weighted behaviors
        # behaviors: [batch_size, num_behaviors, embed_dim] * behavior mask * att: [batch_size, num_behaviors, 1]
        # ->  [batch_size, num_behaviors, embed_dim]
        weighted_behaviors = behaviors_ad_embeddings[:, :-1].mul(mask[:, :-1]).mul(att)
        # sum pooling across all behaviors dim, sum -> [batch_size, embed_dim]
        user_interest = weighted_behaviors.sum(dim=1)

        # concat weighted sum behavior and ad embedd along embedd dim
        # [batch_size, embed_dim] + [batch_size, embed_dim] -> [batch_size, embed_dim * 2]
        concated = torch.hstack([user_interest, behaviors_ad_embeddings[:, -1]])
        # [batch_size, embed_dim * 2] * [embed_dim * 2, mlp_dims] * ... * [mlp_dims, 1] -> [batch_size, 1]
        output = self.mlp(concated)

        return torch.sigmoid(output)
