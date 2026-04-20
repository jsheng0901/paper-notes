import torch
import torch.nn as nn
from ..layer.layer import FeaturesEmbedding, MultiLayerPerceptron


class EntireSpaceMultitaskModel(nn.Module):
    """
    A pytorch implementation of Entire Space Multi-Task Model architecture.
    Here we only support transfer all input features into embedding,
    no difference between sparse or dense features.
    The model only support classification task, will calculate sigmoid as output.
    For two towers, we simply use same hyperparameters.
    Core idea from P(z=1|y=1,x) = P(z=1,y=1|x) / P(y=1|x), cvr = ctcvr / ctr.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for later mlp input
        self.embed_output_dim = len(field_dims) * embed_dim
        # ctr mlp layer ex: [embed_output_dim, 256, 128, 64, 16, 1]
        self.ctr_dnn = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        # cvr mlp layer ex: [embed_output_dim, 256, 128, 64, 16, 1]
        self.cvr_dnn = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, num_tasks]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        # view -> [batch_size, num_fields * embed_dim]
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)

        # ctr -> mlp -> [batch_size, num_fields * embed_dim] -> [batch_size, mlp_dims[-1] ex: 16] -> [batch_size, 1]
        ctr_output = self.ctr_dnn(embed_x)
        # cvr -> mlp -> [batch_size, num_fields * embed_dim] -> [batch_size, mlp_dims[-1] ex: 16] -> [batch_size, 1]
        cvr_output = self.cvr_dnn(embed_x)

        # calculate each task prediction probability value
        # sigmoid -> [batch_size, 1] -> [batch_size, 1]
        ctr_pred = torch.sigmoid(ctr_output)
        cvr_pred = torch.sigmoid(cvr_output)

        # here we have CTCVR = CTR * CVR
        # [batch_size, 1] * [batch_size, 1] -> [batch_size, 1]
        ctcvr_pred = ctr_pred * cvr_pred

        # horizontal cat -> [batch_size, 1 + 1] -> [batch_size, 2]
        # here we only need pred ctcvr and ctr to use calculate loss, since both under same impression sample space
        outputs = torch.cat([ctr_pred, ctcvr_pred], -1)

        return outputs
