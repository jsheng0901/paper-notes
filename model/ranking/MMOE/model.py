import torch
import torch.nn as nn

from ..layer.layer import MultiLayerPerceptron, FeaturesEmbedding


class MultigateMixtureOfExpertsModel(nn.Module):
    """
    A pytorch implementation of the Multi-gate Mixture-of-Experts architecture.
    Here we only support transfer all input features into embedding,
    no difference between sparse or dense features.
    The model only support classification task, will calculate sigmoid as output.
    """

    def __init__(self, field_dims, embed_dim, num_experts, num_tasks, expert_mlp_dims,
                 gate_mlp_dims, tower_mlp_dims, dropout):
        super().__init__()

        # number of tasks
        self.num_tasks = num_tasks
        # number of experts as shared layer
        self.num_experts = num_experts
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for later mlp input
        self.embed_output_dim = len(field_dims) * embed_dim

        # expert dnn layer, ex: [embed_output_dim, 256, 128, 64, 16], no output layer here
        self.expert_dnn = nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, expert_mlp_dims, dropout,
                                                              output_layer=False) for _ in range(self.num_experts)])

        # gate dnn layer, ex: [embed_output_dim, 256, 128, 64, 16], no output layer here
        self.gate_dnn = nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, gate_mlp_dims, dropout,
                                                            output_layer=False) for _ in range(self.num_tasks)])

        # add final gate output linear layer, ex: [gate_mlp_dims[-1], num_experts], no bias
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_mlp_dims[-1], self.num_experts, bias=False) for _ in range(self.num_tasks)])

        # tower dnn (task-specific), ex: [expert_mlp_dims[-1], 256, 128, 64, 16], no output layer here
        # here input is expert_mlp_dims[-1] since after gate control, output dim will keep same expert last layer output
        self.tower_dnn = nn.ModuleList([MultiLayerPerceptron(expert_mlp_dims[-1], tower_mlp_dims, dropout,
                                                             output_layer=False) for _ in range(self.num_tasks)])

        # add final task tower output linear layer, ex: [tower_mlp_dims[-1], 1], no bias
        self.tower_dnn_final_layer = nn.ModuleList(
            [nn.Linear(tower_mlp_dims[-1], 1, bias=False) for _ in range(self.num_tasks)])

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, num_tasks]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        # view -> [batch_size, num_fields * embed_dim] -> [batch_size, embed_output_dim]
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)

        # expert dnn
        expert_outs = []
        # each embed output will run one expert
        for i in range(self.num_experts):
            # [batch_size, embed_output_dim] * [embed_output_dim, expert_mlp_dims[-1] ex: 16]
            # -> [batch_size, expert_mlp_dims[-1] ex: 16]
            expert_out = self.expert_dnn[i](embed_x)
            # [batch_size, expert_mlp_dims[-1]] * num_experts
            expert_outs.append(expert_out)
        # stack -> [batch_size, expert_mlp_dims[-1]] * num_experts -> [batch_size, num_experts, expert_mlp_dims[-1]]
        # for each sample, each expert will have one output dense vector
        expert_outs = torch.stack(expert_outs, 1)

        # gate dnn
        mmoe_outs = []
        # each embed output will run one gate, and number of gate equal to number of tasks
        for i in range(self.num_tasks):
            # [batch_size, embed_output_dim] * [embed_output_dim, gate_mlp_dims[-1] ex: 16]
            # -> [batch_size, gate_mlp_dims[-1] ex: 16]
            gate_dnn_out = self.gate_dnn[i](embed_x)
            # [batch_size, gate_mlp_dims[-1]] * [gate_mlp_dims[-1], num_experts] -> [batch_size, num_experts]
            gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            # softmax -> [batch_size, num_experts] on experts dim, means calculate each expert weight on each sample
            # unsqueeze -> [batch_size, 1, num_experts]
            # matmul -> [batch_size, 1, num_experts] * [batch_size, num_experts, expert_mlp_dims[-1]]
            # -> [batch_size, 1, expert_mlp_dims[-1]]
            # in paper is g_x = softmax(w * x), f_gate_out = sum(g_x * f_expert_out)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)
            # each sample on each expert gate sum weight all expert output on same dims
            # then we got sum weighted expert_mlp_dims[-1] on each vector dim
            # squeeze -> [batch_size, 1, expert_mlp_dims[-1]] -> [batch_size, expert_mlp_dims[-1]]
            mmoe_outs.append(gate_mul_expert.squeeze(1))

        # tower dnn (task-specific)
        task_outs = []
        # each gate weighted output will run one task tower
        for i in range(self.num_tasks):
            # [batch_size, expert_mlp_dims[-1]] * [expert_mlp_dims[-1], tower_mlp_dims[-1] ex: 16]
            # -> [batch_size, tower_mlp_dims[-1] ex: 16]
            tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
            # [batch_size, tower_mlp_dims[-1]] * [tower_mlp_dims[-1], 1] -> [batch_size, 1]
            # get final logit for each sample on each tasks
            tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            # calculate each task prediction probability value, here we only support classification task
            # sigmoid -> [batch_size, 1] -> [batch_size, 1]
            tower_pred = torch.sigmoid(tower_dnn_logit)
            # [batch_size, 1] * num_tasks
            task_outs.append(tower_pred)

        # cat -> [batch_size, 1] * num_tasks -> [batch_size, num_tasks]
        # finally we horizontal concat all tasks output logit for each sample
        outputs = torch.cat(task_outs, -1)

        return outputs
