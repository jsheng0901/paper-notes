import torch
import torch.nn as nn

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron, CrossNetworkMix, BridgeModule, \
    RegulationModule


class EnhancedDeepCrossNetworkMixModel(nn.Module):
    """
    A pytorch implementation of Enhancing Deep & Cross Network.
    Same as DCN v2 Mix, we will inherit DCN_V2 architecture.
    Here we only support deep and wide same number of layers and each layer output keep same dims.
    """

    def __init__(self, field_dims, embed_dim, num_layers, dropout, low_rank, bridge_type, tau):
        super().__init__()
        self.num_layers = num_layers
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for mlp and cross layer input
        self.embed_output_dim = self.num_fields * embed_dim

        # build cross and deep layer separate one by one, since we need add bridge/regulation module between each layer
        self.cross_layers = []
        self.deep_layers = []
        self.bridge_layers = []
        self.regulation_layers = []
        for i in range(num_layers):
            # crate cross layer, each layer have v, u -> [embed_output_dim, low_rank], b -> [embed_output_dim, ]
            cross = CrossNetworkMix(self.embed_output_dim, 1, low_rank)
            self.cross_layers.append(cross)
            # deep layer dims, here deep layer each layer output dim same as cross layer output
            deep = MultiLayerPerceptron(self.embed_output_dim, [self.embed_output_dim],
                                        dropout, output_layer=False)
            self.deep_layers.append(deep)
            # build bridge module according to bridge type,
            bridge = BridgeModule(self.embed_output_dim, bridge_type)
            self.bridge_layers.append(bridge)
            # build regulation module for each layer, here each layer have two towers output
            regulation_deep = RegulationModule(self.num_fields, embed_dim, tau)
            regulation_wide = RegulationModule(self.num_fields, embed_dim, tau)
            self.regulation_layers.append([regulation_deep, regulation_wide])

        # add one linear layer for concat deep and cross for output, [embed_output_dim * 3, 1]
        self.linear = torch.nn.Linear(self.embed_output_dim * 3, 1)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        embed_x = self.embedding(x)
        # deep for embed: [batch_size, num_fields, embed_dim] -> regulation -> [batch_size, num_fields * embed_dim]
        deep_in = self.regulation_layers[0][0](embed_x)
        # wide for embed: [batch_size, num_fields, embed_dim] -> regulation -> [batch_size, num_fields * embed_dim]
        cross_in = self.regulation_layers[0][1](embed_x)

        # run cross and deep parallel layer by layer
        for i in range(self.num_layers):
            # [batch_size, num_fields * embed_dim] -> cross -> [batch_size, num_fields * embed_dim]
            cross_out = self.cross_layers[i](cross_in)
            # [batch_size, num_fields * embed_dim] -> deep -> [batch_size, num_fields * embed_dim]
            deep_out = self.deep_layers[i](deep_in)
            # ([batch_size, num_fields * embed_dim], [batch_size, num_fields * embed_dim])
            # -> [batch_size, num_fields * embed_dim]
            bridge_out = self.bridge_layers[i](cross_out, deep_out)

            # run regulation module for each layer output except last layer
            if i + 1 < self.num_layers:
                # view -> [batch_size, num_fields * embed_dim] -> [batch_size, num_fields, embed_dim]
                bridge_out = bridge_out.view(-1, self.num_fields, self.embed_dim)
                # [batch_size, num_fields, embed_dim] -> regulation -> [batch_size, num_fields * embed_dim]
                deep_in = self.regulation_layers[i + 1][0](bridge_out)
                # [batch_size, num_fields, embed_dim] -> regulation -> [batch_size, num_fields * embed_dim]
                cross_in = self.regulation_layers[i + 1][1](bridge_out)

        # horizontal stack -> [batch_size, num_fields * embed_dim * 3], each output has same output dim
        stacked = torch.cat([cross_out, deep_out, bridge_out], dim=1)
        # [batch_size, num_fields * embed_dim * 3] * [num_fields * embed_dim * 3, 1] -> [batch_size, 1]
        output = self.linear(stacked)

        return torch.sigmoid(output)
