import torch
import torch.nn as nn

from ..layer.layer import FeaturesEmbedding, MultiLayerPerceptron


class ProgressiveLayeredExtractionModel(nn.Module):
    """
    A pytorch implementation the multi level of Customized Gate Control of Progressive Layered Extraction architecture.
    Here we only support transfer all input features into embedding,
    no difference between sparse or dense features.
    The model only support classification task, will calculate sigmoid as output.
    """

    def __init__(self, field_dims, embed_dim, shared_expert_num, specific_expert_num, num_levels,
                 expert_mlp_dims, gate_mlp_dims, tower_mlp_dims, dropout, num_tasks):
        super().__init__()

        # number of tasks
        self.num_tasks = num_tasks
        # number of experts for specific task not share expert
        self.specific_expert_num = specific_expert_num
        # number of share experts
        self.shared_expert_num = shared_expert_num
        # number of level for MLP layers
        self.num_levels = num_levels
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # get embedding output dims, will concat all field embed into one vector, for later mlp input
        self.embed_output_dim = len(field_dims) * embed_dim

        # create multi mlp module, for each level, each task, and each expert
        def multi_module_list(num_level, num_tasks, expert_num, inputs_dim_level0, inputs_dim_not_level0, hidden_units):
            return nn.ModuleList(
                [nn.ModuleList([nn.ModuleList([MultiLayerPerceptron(inputs_dim_level0 if level_num == 0
                                                                    else inputs_dim_not_level0, hidden_units, dropout,
                                                                    output_layer=False) for _ in range(expert_num)])
                                for _ in range(num_tasks)]) for level_num in range(num_level)])

        # 1. experts
        # task-specific experts, ex: [embed_output_dim, 128, 64, 32] -- [32, 128, 64, 32], no output layer here
        self.specific_experts = multi_module_list(self.num_levels, self.num_tasks, self.specific_expert_num,
                                                  self.embed_output_dim, expert_mlp_dims[-1], expert_mlp_dims)

        # shared experts, ex: [embed_output_dim, 128, 64, 32] -- [32, 128, 64, 32], no output layer here
        # both specific and shared use same expert_mlp_dims[-1] output to make sure later can vertical stack
        self.shared_experts = multi_module_list(self.num_levels, 1, self.specific_expert_num,
                                                self.embed_output_dim, expert_mlp_dims[-1], expert_mlp_dims)

        # 2. gates
        # gates for task-specific experts
        specific_gate_output_dim = self.specific_expert_num + self.shared_expert_num
        self.specific_gate_dnn = multi_module_list(self.num_levels, self.num_tasks, 1,
                                                   self.embed_output_dim, expert_mlp_dims[-1], gate_mlp_dims)

        # here input dim is always gate_mlp_dims last dim, since final layer always add after specific gate dnn
        self.specific_gate_dnn_final_layer = nn.ModuleList(
            [nn.ModuleList([nn.Linear(gate_mlp_dims[-1], specific_gate_output_dim, bias=False)
                            for _ in range(self.num_tasks)]) for _ in range(self.num_levels)])

        # gates for shared experts, ex: 2 * 1 + 1 = 3, for all experts output
        shared_gate_output_dim = self.num_tasks * self.specific_expert_num + self.shared_expert_num

        self.shared_gate_dnn = nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim if level_num == 0
                                                                   else expert_mlp_dims[-1], gate_mlp_dims, dropout,
                                                                   output_layer=False)
                                              for level_num in range(self.num_levels)])

        # here input dim is always gate_mlp_dims last dim, since final layer always add after shared gate dnn
        self.shared_gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_mlp_dims[-1], shared_gate_output_dim, bias=False) for _ in range(self.num_levels)])

        # 3. tower dnn (task-specific)
        self.tower_dnn = nn.ModuleList(
            [MultiLayerPerceptron(expert_mlp_dims[-1], tower_mlp_dims, dropout, output_layer=False)
             for _ in range(self.num_tasks)])

        self.tower_dnn_final_layer = nn.ModuleList(
            [nn.Linear(tower_mlp_dims[-1], 1, bias=False) for _ in range(self.num_tasks)])

    def cgc_net(self, inputs, level_num):
        # a single cgc Layer, ple net is several cgc net horizontal stacked and tower output net
        # output will be a list contains all gate output, including number of task gate + shared task gate
        # inputs: [task1, task2, ... taskn, shared task] each one is embed_x, ex: [4096, 130]

        # 1. experts
        # task-specific experts
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                # extract specific level, task and which expert, and pass corresponding task input embed_x
                # get each level and each task and each specific expert -> ex: [embed_output_dim, expert_mlp_dims]
                # ex: [batch_size, embed_output_dim] * [embed_output_dim, expert_mlp_dims[-1] ex: 32]
                # -> [batch_size, expert_mlp_dims[-1] ex: 32]
                specific_expert_output = self.specific_experts[level_num][i][j](inputs[i])
                # [batch_size, expert_mlp_dims[-1]] * num_experts * num_tasks
                specific_expert_outputs.append(specific_expert_output)

        # shared experts
        shared_expert_outputs = []
        for k in range(self.shared_expert_num):
            # extract specific level, only one task and which expert, and pass corresponding shared task input embed_x
            # get each level and only one task and each specific expert -> ex: [embed_output_dim, expert_mlp_dims]
            # ex: [batch_size, embed_output_dim] * [embed_output_dim, expert_mlp_dims[-1] ex: 32]
            # -> [batch_size, expert_mlp_dims[-1] ex: 32]
            shared_expert_output = self.shared_experts[level_num][0][k](inputs[-1])
            # [batch_size, expert_mlp_dims[-1]] * num_experts * 1
            shared_expert_outputs.append(shared_expert_output)

        # 2. gates
        # gates for task-specific experts
        cgc_outs = []
        for i in range(self.num_tasks):
            # extract specific expert outputs, each task have number of specific expert
            # concat task-specific expert and task-shared expert
            # ex: list concat [batch_size, expert_mlp_dims[-1], batch_size, expert_mlp_dims[-1]]
            cur_experts_outputs = specific_expert_outputs[
                                  i * self.specific_expert_num: (i + 1) * self.specific_expert_num
                                  ] + shared_expert_outputs
            # stack both output vertical -> [batch_size, expert_mlp_dims[-1], batch_size, expert_mlp_dims[-1]]
            # -> [batch_size, 2, expert_mlp_dims[-1]] == [batch_size, specific_gate_output_dim, expert_mlp_dims[-1]]
            # both specific and shared output same output dims, but meaning is different
            cur_experts_outputs = torch.stack(cur_experts_outputs, 1)

            # gate dnn
            # extract specific level, task and only one expert, and pass corresponding task input embed_x
            # get each level and each task and only one expert -> ex: [embed_output_dim, gate_mlp_dims]
            # ex: [batch_size, embed_output_dim] * [embed_output_dim, gate_mlp_dims[-1] ex: 16]
            # -> [batch_size, gate_mlp_dims[-1] ex: 16]
            gate_dnn_out = self.specific_gate_dnn[level_num][i][0](inputs[i])
            # get each level and each task -> ex: [gate_mlp_dims[-1], specific_gate_output_dim ex: 2]
            # ex: [batch_size, gate_mlp_dims[-1]] * [gate_mlp_dims[-1], specific_gate_output_dim ex: 2]
            # -> [batch_size, specific_gate_output_dim ex: 2]
            # here calculate each sample to each task all experts (specific + shared) weight
            gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](gate_dnn_out)
            # softmax get probability -> [batch_size, specific_gate_output_dim ex: 2]
            # -> unsqueeze -> [batch_size, 1, specific_gate_output_dim ex: 2]
            # [batch_size, 1, specific_gate_output_dim ex: 2] * [batch_size, 2, expert_mlp_dims[-1] ex: 32]
            # -> [batch_size, 1, expert_mlp_dims[-1] ex: 32]
            # each sample on each expert gate sum weight each task all expert (specific + shared) output on same dims
            # then we got sum weighted expert_mlp_dims[-1] on each vector dim
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)
            # squeeze -> [batch_size, expert_mlp_dims[-1]] -> append -> * num_tasks
            cgc_outs.append(gate_mul_expert.squeeze())

        # gates for shared experts
        # [batch_size, expert_mlp_dims[-1]] * num_experts * num_tasks + [batch_size, expert_mlp_dims[-1]] * num_experts
        cur_experts_outputs = specific_expert_outputs + shared_expert_outputs
        # stack ->[batch_size, num_specific_experts * num_tasks + num_shared_experts ex: 3, expert_mlp_dims[-1]]
        cur_experts_outputs = torch.stack(cur_experts_outputs, 1)

        # shared gate dnn
        # extract specific level, and pass corresponding shared task input embed_x
        # get each level and only one task and only one expert -> ex: [embed_output_dim, gate_mlp_dims]
        # ex: [batch_size, embed_output_dim] * [embed_output_dim, gate_mlp_dims[-1] ex: 16]
        # -> [batch_size, gate_mlp_dims[-1] ex: 16]
        gate_dnn_out = self.shared_gate_dnn[level_num](inputs[-1])
        # get each level and shared task -> ex: [gate_mlp_dims[-1], shared_gate_output_dim ex: 3]
        # ex: [batch_size, gate_mlp_dims[-1]] * [gate_mlp_dims[-1], shared_gate_output_dim ex: 3]
        # -> [batch_size, shared_gate_output_dim ex: 3]
        # here calculate each sample to all tasks expert (specific for all tasks + shared) weight
        gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](gate_dnn_out)
        # softmax get probability -> [batch_size, shared_gate_output_dim ex: 3]
        # -> unsqueeze -> [batch_size, 1, shared_gate_output_dim ex: 3]
        # [batch_size, 1, shared_gate_output_dim ex: 3] * [batch_size, 3, expert_mlp_dims[-1] ex: 32]
        # -> [batch_size, 1, expert_mlp_dims[-1] ex: 32]
        # each sample on each expert gate sum weight all tasks expert (specific for all tasks + shared)
        # output on same dims, then we got sum weighted expert_mlp_dims[-1] on each vector dim
        gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)
        # squeeze -> [batch_size, expert_mlp_dims[-1]] -> append shared task which is only one
        cgc_outs.append(gate_mul_expert.squeeze())

        # final cgc outs -> [batch_size, expert_mlp_dims[-1] ex: 32] * (num_task ex: 2 + shared task ex: 1 --> 3)
        return cgc_outs

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, num_tasks]
        """
        # x -> [batch_size, num_fields]
        # embed(x) -> [batch_size, num_fields, embed_dim]
        # view -> [batch_size, num_fields * embed_dim] -> [batch_size, embed_output_dim]
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)

        # repeat embed_x for (num_task + shared task) times to generate cgc input
        # [task1, task2, ... taskn, shared task] each one is embed_x
        ple_inputs = [embed_x] * (self.num_tasks + 1)
        ple_outputs = []
        for i in range(self.num_levels):
            # get each level cgc output, ex: [batch_size, expert_mlp_dims[-1] ex: 32] * (num_task + shared task = 3)
            ple_outputs = self.cgc_net(inputs=ple_inputs, level_num=i)
            ple_inputs = ple_outputs

        # tower dnn (task-specific)
        task_outs = []
        # each gate weighted output will run one task tower
        # ple outputs contains each task gate output and one shared task gate output, but last one we don't use
        for i in range(self.num_tasks):
            # get each task tower dnn -> ex: [expert_mlp_dims[-1], tower_mlp_dims]
            # ex: [batch_size, expert_mlp_dims[-1]] * [expert_mlp_dims[-1], tower_mlp_dims[-1] ex: 32]
            # -> [batch_size, tower_mlp_dims[-1] ex: 32]
            tower_dnn_out = self.tower_dnn[i](ple_outputs[i])
            # get each task tower dnn last layer -> ex: [tower_mlp_dims[-1] ex: 32, 1]
            # ex: [batch_size, tower_mlp_dims[-1] ex: 32] * [tower_mlp_dims[-1] ex: 32, 1] -> [batch_size, 1]
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
