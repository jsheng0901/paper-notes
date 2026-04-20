import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron, CrossNetworkMix


class MultiHeadFeatureEmbedding(nn.Module):
    """
    Split embedding dim into multi head, same like multi head self attention
    """

    def __init__(self, field_dims, embed_dim, num_heads=2):
        super().__init__()
        # number of head want to split embedding into
        self.num_heads = num_heads
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, num_heads, num_fields * embed_dim / num_heads]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embed_x = self.embedding(x)
        # split -> [batch_size, num_fields, embed_dim] -> [batch_size, num_fields, embed_dim / num_heads] * num_heads
        multi_head_embed_x = torch.tensor_split(embed_x, self.num_heads, dim=-1)
        # stacked -> [batch_size, num_heads, num_fields, embed_dim / num_heads]
        multi_head_embed_x = torch.stack(multi_head_embed_x, dim=1)
        # split -> [batch_size, num_heads, num_fields, embed_dim / num_heads] ->
        # [batch_size, num_heads, num_fields, embed_dim / 2 * num_heads] * 2
        # split each head embedding into 2 part
        multi_head_embed_x1, multi_head_embed_x2 = torch.tensor_split(multi_head_embed_x, 2, dim=-1)
        # flatten -> [batch_size, num_heads, num_fields, embed_dim / 2 * num_heads] ->
        # [batch_size, num_heads, num_fields * embed_dim / 2 * num_heads]
        # for each split head and each field embedding flatten into one vector (dims)
        multi_head_embed1 = multi_head_embed_x1.flatten(start_dim=2)
        # same as above output -> [batch_size, num_heads, num_fields * embed_dim / 2 * num_heads]
        multi_head_embed2 = multi_head_embed_x2.flatten(start_dim=2)
        # concat -> [batch_size, num_heads, num_fields * embed_dim / 2 * num_heads * 2] ->
        # [batch_size, num_heads, num_fields * embed_dim / num_heads]
        # concat each head both split embed into one vector again, need to flatten first then concat back
        # otherwise, split without flatten first, will do same before split by 2
        # then we don't have two split well flatten embedding into one vector
        multi_head_embed = torch.cat([multi_head_embed1, multi_head_embed2], dim=-1)

        return multi_head_embed


class ExponentialCrossNetwork(nn.Module):
    """
    This is exp cross network without projection dim and MoE.
    """

    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True, batch_norm=False, net_dropout=0.1, num_heads=1):
        super().__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()

        # add parameter for each layer
        for i in range(num_cross_layers):
            # w -> [input_dim, input_dim / 2]
            # here we don't apply projection dim, and will mask half of parameter for LN
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            # each cross layer have one bias -> [input_dim, ]
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                # LN only apply on half of the layer output
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                # BN on head dim
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                # apply dropout layer
                self.dropout.append(nn.Dropout(net_dropout))
            # init as uniform distribution
            nn.init.uniform_(self.b[i].data)

        # activate function layer
        self.masker = nn.ReLU()
        # linear output layer -> [input_dim, 1]
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: [batch_size, num_heads, num_fields * embed_dim / num_heads]
        :return: [batch_size, num_heads, 1]
        """
        # create exp cross layer
        for i in range(self.num_cross_layers):
            # [batch_size, num_heads, num_fields * embed_dim / num_heads] * [input_dim, input_dim / 2]
            # num_fields * embed_dim / num_heads = input_dim
            head = self.w[i](x)
            # apply BN layer
            if len(self.batch_norm) > i:
                head = self.batch_norm[i](head)
            # apply LN
            if len(self.layer_norm) > i:
                # LN -> [batch_size, num_heads, input_dim / 2]
                norm_head = self.layer_norm[i](head)
                # add mask after LN
                mask = self.masker(norm_head)
            else:
                # add mask
                mask = self.masker(head)
            # cat -> [batch_size, num_heads, input_dim / 2 + input_dim / 2] -> [batch_size, num_heads, input_dim]
            # cat two type of head, one is normal head, another is mask head with 0 - 1 activation function
            head = torch.cat([head, head * mask], dim=-1)
            # [batch_size, num_heads, input_dim] + [input_dim, ] -> [batch_size, num_heads, input_dim]
            # x * x_w: [batch_size, num_heads, input_dim] * [batch_size, num_heads, input_dim]
            # -> [batch_size, num_heads, input_dim] Hadamard-product keep same dims
            # [batch_size, num_heads, input_dim] + [batch_size, num_heads, input_dim]
            # -> [batch_size, num_heads, input_dim]
            # here is there core formular, no x_0 product instead of x from last layer output, achieve exp cross
            x = x * (head + self.b[i]) + x
            # add dropout layer
            if len(self.dropout) > i:
                x = self.dropout[i](x)

        # apply linear output layer
        # [batch_size, num_heads, input_dim] * [input_dim, 1] -> [batch_size, num_heads, 1]
        logit = self.output(x)

        return logit


class LinearCrossNetwork(nn.Module):
    """
    This is linear cross network without projection dim and MoE.
    """

    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True, batch_norm=False, net_dropout=0.1, num_heads=1):
        super().__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: [batch_size, num_heads, num_fields * embed_dim / num_heads]
        :return: [batch_size, num_heads, 1]
        """
        x0 = x
        for i in range(self.num_cross_layers):
            head = self.w[i](x)
            if len(self.batch_norm) > i:
                head = self.batch_norm[i](head)
            if len(self.layer_norm) > i:
                norm_head = self.layer_norm[i](head)
                mask = self.masker(norm_head)
            else:
                mask = self.masker(head)
            head = torch.cat([head, head * mask], dim=-1)
            # everything same as exp cross network, except this line
            # here is same as DCN_V2
            x = x0 * (head + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.output(x)

        return logit


class DeepCrossNetworkV3Model(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network v3.
    Same as DCN v2, but no projection dim and no MoE, add exp cross layer with multi head embedding.
    """

    def __init__(self,
                 field_dims,
                 embed_dim=10,
                 num_linear_cross_layers=4,
                 num_exp_cross_layers=4,
                 exp_net_dropout=0.1,
                 linear_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1):
        super().__init__()
        # init multi head embedding, head embedding dim set as times num_heads, each head has same embedding dim
        # here after embedding we will get embedding layer -> [sum(fields_dims), embed_dim * num_heads]
        self.embedding_layer = MultiHeadFeatureEmbedding(field_dims, embed_dim * num_heads, num_heads)

        # get embedding layer output dims, this will be ECN and  LCN input dim as well
        # here embedding layer output will be same as flatten input x for each head
        input_dim = len(field_dims) * embed_dim

        # init exp cross layer
        self.ECN = ExponentialCrossNetwork(input_dim=input_dim,
                                           num_cross_layers=num_exp_cross_layers,
                                           net_dropout=exp_net_dropout,
                                           layer_norm=layer_norm,
                                           batch_norm=batch_norm,
                                           num_heads=num_heads)
        # init linear cross layer
        self.LCN = LinearCrossNetwork(input_dim=input_dim,
                                      num_cross_layers=num_linear_cross_layers,
                                      net_dropout=linear_net_dropout,
                                      layer_norm=layer_norm,
                                      batch_norm=batch_norm,
                                      num_heads=num_heads)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: {y_pred: [batch_size, 1], y_exp: [batch_size, 1], y_linear: [batch_size, 1]}
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_heads, num_fields * embed_dim]
        # here each head has same embedding dim
        embed_x = self.embedding_layer(x)
        # [batch_size, num_heads, num_fields * embed_dim] -> exp -> [batch_size, num_heads, 1]
        # mean -> [batch_size, num_heads, 1] -> [batch_size, 1]
        # take mean on each head
        exp_logit = self.ECN(embed_x).mean(dim=1)
        # [batch_size, num_heads, num_fields * embed_dim] -> exp -> [batch_size, num_heads, 1]
        # mean -> [batch_size, num_heads, 1] -> [batch_size, 1]
        linear_logit = self.LCN(embed_x).mean(dim=1)
        # take mean of two layer outputs [batch_size, 1] + [batch_size, 1] -> [batch_size, 1]
        logit = (exp_logit + linear_logit) * 0.5
        # sigmoid final predict logit
        y_pred = torch.sigmoid(logit)

        # record each layers output logit
        output_dict = {"y_pred": y_pred,
                       "y_exp": torch.sigmoid(exp_logit),
                       "y_linear": torch.sigmoid(linear_logit)
                       }

        return output_dict
