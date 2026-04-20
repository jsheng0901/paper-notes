import torch.nn as nn
import torch

from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron, EmbeddingsInteraction


class DLRM(nn.Module):
    """
    A pytorch implementation of DLRM.
    Here we skip the dense feature bottom MLP layer which original from paper to transfer all dense feature into one
    embedding vector with same size as category feature embedding.
    For dense feature how to transfer, reference: https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DLRM/src/DLRM.py
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, interaction_op):
        super().__init__()
        # get number of fields
        self.num_fields = len(field_dims)
        # get interaction layer operation type
        self.interaction_op = interaction_op

        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # build interaction layer
        if self.interaction_op == "dot":
            # get embedding interaction layer, same as inner product layer
            self.interact = EmbeddingsInteraction()
            # got inner product output dim same as top mlp input dim
            top_input_dim = (self.num_fields * (self.num_fields - 1)) // 2
        elif self.interaction_op == "cat":
            # flatten layer to concat all dims after start_dim, same as view but more useful when too many dims
            self.interact = nn.Flatten(start_dim=1)
            # got concat output dim same as top mlp input dim
            top_input_dim = self.num_fields * embed_dim
        else:
            raise ValueError(f"interaction_op={self.interaction_op} not supported.")

        # mlp layer dims, ex: [top_input_dim, 128, 64, 32, 16, 1]
        self.mlp = MultiLayerPerceptron(top_input_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # embedding(x) -> [batch_size, num_fields, embed_dim]
        embeddings = self.embedding(x)
        # transfer the embedding with feature interaction
        # dot: [batch_size, num_fields, embed_dim] -> [batch_size, num_fields * (num_fields - 1) // 2, embed_dim]
        # cat: [batch_size, num_fields, embed_dim] -> [batch_size, num_fields * embed_dim]
        interact_out = self.interact(embeddings)
        if self.interaction_op == "dot":
            # sum -> [batch_size, num_fields * (num_fields - 1) // 2]
            interact_out = torch.sum(interact_out, dim=2)
        # mlp -> [batch_size, top_input_dim] -> [batch_size, 1]
        output = self.mlp(interact_out)

        return torch.sigmoid(output)
