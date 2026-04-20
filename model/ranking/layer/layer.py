import torch
import torch.nn as nn
import numpy as np


class FeaturesEmbedding(nn.Module):
    """
    Embedding each features each value to embed dimensions
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # field_dims: list record each feature number of unique value
        # ex: field_dims = [2, 3, 4, 5], embedding lookup vocab size will be sum(field_dims) -> 14, like [14, 10]
        # which means transfer each features each unique value into embed dimes
        # same as one-hot each feature and then apply embedding, but here all unique feature will be used in same vocab
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # ex: field_dims = [2, 3, 4, 5], cum_sum -> [2, 5, 9, 14], [:-1]-> [2, 5, 9] -> add 0 then offsets [0, 2, 5, 9]
        # first feature not need add shift since first no shift from left side
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        # init parameter data
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, num_fields, embedding_dim]
        """
        # x -> [batch_size, num_fields] + [num_fields] -> [batch_size, num_fields]
        x = x + x.new_tensor(self.offsets)
        #  x * embedding -> [batch_size, num_fields] ->
        #  [batch_size, num_fields, sum(num_fields)] * [sum(num_fields), embed_dim]
        #  -> [batch_size, num_fields, embedding_dim]
        # same as nlp embedding layer, x input num_fields is equal length in nlp, and each is one is index in vocab,
        # sum(num_fields) is vocab size, then we need list each feature in vocab where is 1, and then do look up.
        return self.embedding(x)


class FeaturesLinear(nn.Module):
    """
    Feature linear layer, same as fully connected linear layer. output = w_i * x + w_0
    """

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        # same as simple linear layer but output dim is 1 -> [sum(num_fields), 1], ex: [241895, 1],
        self.linear = torch.nn.Embedding(sum(field_dims), output_dim)
        # bias -> [1, ], will be broadcasting on each feature
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # same as features embedding layer, ex: [39, ]
        # use to shift each feature index value by previous all unique value size, transfer index to vocab index
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields] -> [4096, 39], each feature will be an index like in nlp token index in vocab.
        # but in here each feature index not cumulate by feature, ex: feature1: [0, 1, 2], features2: [0, 1]
        # but before pass into embedding, each unique value should be transfer to index in all vocab size, so
        # after shift each by each feature sum of unique value, we will have ex: feature1: [0, 1, 2], features2: [3, 4]
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # x * linear -> [batch_size, num_fields] -> [batch_size, num_fields, sum(num_fields)] * [sum(num_fields), 1]
        # -> [batch_size, num_fields, 1] -> sum -> [batch_size, 1] + bias -> [batch_size, 1] + [1, ] -> [batch_size, 1]
        # here sum across all features in each batch, same as linear layer, for each feature times weight then sum
        return torch.sum(self.linear(x), dim=1) + self.bias


class FeaturesCross(nn.Module):
    """
    Feature cross layer, output = 0.5 * sum_f((sum_i(vi_f * x_i))^2 - sum_i(vi_f^2 * x_i^2))
    Calculate each feature cross interaction weight.
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embed_dim]
        :return: [batch_size, 1] or [batch_size, embed_dim]
        """
        # input x already apply embedding layer then pass into cross layer
        # formula (vi_f * x_i)^2 -> x * cross -> ([n_samples, n_features] * [n_features, k])^2 -> [n_samples, k]^2
        # x -> [batch_size, num_fields, embedding_dim] -> sum -> [batch_size, k]^2 -> [4096, 10]
        square_of_sum = torch.pow(torch.sum(x, dim=1), 2)
        # (vi_f^2 * x_i^2) -> x^2 * cross^2 -> ([n_samples, n_features])^2 * ([n_features, k])^2 -> [n_samples, k]
        # x -> [batch_size, num_fields, embedding_dim]^2 -> sum -> [batch_size, k] -> [4096, 10]
        sum_of_square = torch.sum(torch.pow(x, 2), dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            # each sample sum along k dims, sum([batch_size, k] - [batch_size, k]) -> [batch_size, 1] -> [4096, 1]
            output = torch.sum(output, dim=1, keepdim=True)

        return 0.5 * output


class MultiLayerPerceptron(nn.Module):
    """
    Feed forward neural network layer, same name as mlp layer. l1 = w_1 * x + w_1_0, l2 = w_2 * l1 + w_2_0,
    """

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        # stack the layer
        layers = list()
        # loop through all layer output embedding dims
        for embed_dim in embed_dims:
            # add linear layer -> [input_dime, embed_dim]
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # add batch norm layer on embed dim
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            # add relu activation function
            layers.append(torch.nn.ReLU())
            # add drop out layer
            layers.append(torch.nn.Dropout(p=dropout))
            # last layer output dime is next layer input dime
            input_dim = embed_dim
        # if we have output final layer, then add one linear layer for output
        if output_layer:
            # add output linear layer -> [input_dime, 1]
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: [batch_size, embed_dim]
        :return: if have output layer [batch_size, 1], else [batch_size, embed_dim]
        """
        # x -> [batch_size, embed_dim]
        # [batch_size, embed_dim] * [embed_dim, embed_dim_1] * [embed_dim, embed_dim_2] ... -> [batch_size, embed_dim]
        return self.mlp(x)


class EmbeddingsInteraction(nn.Module):
    """
    Embedding interaction layer.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_dim]
        :return: [batch_size, num_fields * (num_fields - 1) // 2, embedding_dim]
        """
        num_fields = x.shape[1]
        row, col = [], []
        # loop through all feature combination, each feature need calculate inner product with all other feature embed
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                row.append(i)
                col.append(j)
        # x1: [batch_size, 1, embedding_dim] * x2: [batch_size, 1, embedding_dim] -> [batch_size, 1, embedding_dim]
        # we have num_fields * (num_fields - 1) // 2 combination, mul will keep same output shape as input
        # then we have [batch_size, num_fields * (num_fields)//2, embedding_dim] output
        interaction = torch.mul(x[:, row], x[:, col])

        return interaction


class Dice(nn.Module):
    """
    Dice activation function. y_i = alpha_i * (1 - p_i) * y_i + p_i * y_i
    p_i = 1 / (1 + e^(-E(y_i)/sqrt(Var(y_i) + theta))
    p is same like take BN of x first then sigmoid get probability.
    """

    def __init__(self):
        super().__init__()
        # hyper-parameter
        self.alpha = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedd_dim]
        :return: [batch_size, num_fields, embedd_dim]
        """
        # calculate mean across mini-batch, avg -> [num_fields, embedd_dim]
        avg = x.mean(dim=0)
        # calculate std across mini-batch, std -> [num_fields, embedd_dim]
        std = x.std(dim=0)
        # same as BN, without parameters, norm_x -> [batch_size, num_fields, embedd_dim]
        norm_x = (x - avg) / std
        # sigmoid -> [batch_size, num_fields, embedd_dim]
        p = torch.sigmoid(norm_x)
        # [batch_size, num_fields] * constant -> [batch_size, num_fields, embedd_dim]
        output = x.mul(p) + self.alpha * x.mul(1 - p)

        return output


class BridgeModule(nn.Module):
    """Bridge Module used in EDCN

      Input shape
        - A list of two 2D tensor with shape: ``(batch_size, input_dim)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, output_dim)``.

    Arguments - **bridge_type**: The type of bridge interaction, one of 'pointwise_addition', 'hadamard_product',
    'concatenation', 'attention_pooling'

        - **activation**: Activation function to use.

      References - [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep
      CTR Models.](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)

    """

    def __init__(self, input_dim, bridge_type='hadamard_product'):
        super().__init__()

        self.bridge_type = bridge_type

        # build dense layer according to bridge type, for add and product no parameter layer need
        # currently only support input x and input h has same input dim
        if self.bridge_type == "concatenation":
            # concat will double input dim
            self.w = nn.Linear(input_dim * 2, input_dim, bias=True)
        elif self.bridge_type == "attention_pooling":
            self.w_x = nn.Linear(input_dim, input_dim, bias=True)
            self.p_x = nn.Linear(input_dim, input_dim)
            self.w_h = nn.Linear(input_dim, input_dim, bias=True)
            self.p_h = nn.Linear(input_dim, input_dim)

    def forward(self, x, h):
        """
        :param x: [batch_size, input_dim]
        :param h: [batch_size, input_dim]
        :return: [batch_size, input_dim]
        """
        if x.shape[-1] != h.shape[-1]:
            raise ValueError(
                f"Only support two layer input same dim, Got `x` input shape: {x.shape}, `h` input shape: {h.shape}"
            )

        # init dense fusion function output
        f = None
        if self.bridge_type == "pointwise_addition":
            # no parameter need
            # [batch_size, input_dim] + [batch_size, input_dim] -> [batch_size, input_dim]
            f = x + h
        elif self.bridge_type == "hadamard_product":
            # no parameter need
            # [batch_size, input_dim] * [batch_size, input_dim] -> [batch_size, input_dim]
            f = x * h
        elif self.bridge_type == "concatenation":
            # concat -> [batch_size, input_dim * 2] * [input_dim * 2, input_dim] -> [batch_size, input_dim]
            stacked = torch.cat([x, h], dim=1)
            f = self.w(stacked)
            # relu -> [batch_size, input_dim]
            f = torch.relu(f)
        elif self.bridge_type == "attention_pooling":
            # calculate attention score for x
            # [batch_size, input_dim] * [input_dim, input_dim] -> [batch_size, input_dim]
            a_x = self.w_x(x)
            a_x = torch.relu(a_x)
            # [batch_size, input_dim] * [input_dim, input_dim] -> [batch_size, input_dim]
            a_x = self.p_x(a_x)
            # apply softmax along last dim
            a_x = torch.softmax(a_x, dim=-1)

            # calculate attention score for h
            # [batch_size, input_dim] * [input_dim, input_dim] -> [batch_size, input_dim]
            a_h = self.w_h(h)
            a_h = torch.relu(a_h)
            # [batch_size, input_dim] * [input_dim, input_dim] -> [batch_size, input_dim]
            a_h = self.p_h(a_h)
            # apply softmax along last dim
            a_h = torch.softmax(a_h, dim=-1)

            # tims attention score for each input and sum together
            f = a_x * x + a_h * h

        return f


class RegulationModule(nn.Module):
    """Regulation module used in EDCN.

      Input shape
        - 3D tensor with shape: ``(batch_size, num_fields, embedding_dim)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, num_fields * embedding_dim)``.

      Arguments
        - **tau** : Positive float, the temperature coefficient to control
        distribution of field-wise gating unit.

      References
        - [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models.](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)
    """

    def __init__(self, num_fields, embedding_dim, tau=1.0):
        super().__init__()

        if tau == 0:
            raise ValueError("RegulationModule tau can not be zero.")
        self.tau = 1.0 / tau

        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.output_dim = num_fields * embedding_dim

        # init matrix g for each field -> [1, num_fields, 1]
        self.g = nn.Parameter(torch.ones((1, self.num_fields, 1)))

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_dim]
        :return: [batch_size, num_fields * embedding_dim]
        """
        n_dims = len(x.shape)
        if n_dims != 3:
            raise ValueError(
                f"Unexpected inputs dimensions {n_dims}, expect to be 3 dimensions"
            )

        # [1, num_fields, 1] * [1] -> [1, num_fields, 1] -> softmax -> [1, num_fields, 1]
        # get each field gate score times temperature coefficient than softmax along with field dim
        field_gating_score = torch.softmax(self.g * self.tau, 1)
        # [batch_size, num_fields, embedding_dim] * [1, num_fields, 1] -> [batch_size, num_fields, embedding_dim]
        e = x * field_gating_score
        # [batch_size, num_fields, embedding_dim] -> [batch_size, num_fields * embedding_dim]
        e = e.view(-1, self.output_dim)

        return e


class CrossNetworkMix(nn.Module):
    """
    This is pytorch implementation version of cross net, original from DCN_V2 Mix.
    Here we only apply low rank projection implementation. No MoE logic in here.
    """
    def __init__(self, input_dim, num_layers, low_rank):
        super().__init__()
        self.num_layers = num_layers
        # each cross layer have one low rank projection linear layer -> [input_dim, low_rank] without bias
        # input_dim will be embed output dims
        self.v = nn.ModuleList([
            nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_layers)
        ])
        # each cross layer have one projection back linear layer -> [low_rank, input_dim] without bias
        self.u = nn.ModuleList([
            nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_layers)
        ])
        # each cross layer have one bias -> [input_dim, ]
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: [batch_size, num_fields * embed_dim]
        :return: [batch_size, num_fields * embed_dim]
        """
        # x -> [batch_size, num_fields * embed_dim]
        # save init input x value to make sure after cross many layer not far away from init input x
        x_0 = x
        # create cross layer
        for i in range(self.num_layers):
            # num_fields * embed_dim == embed_output_dim = input_dim, projection into low rank dims
            # [batch_size, num_fields * embed_dim] * [input_dim, low_rank] -> [batch_size, low_rank]
            v_x = self.v[i](x)
            # nonlinear activation in low rank space, tanh -> [batch_size, low_rank]
            v_x = torch.tanh(v_x)
            # projection from low rank dims back to input dims
            # [batch_size, low_rank] * [low_rank, input_dim] -> [batch_size, input_dim = num_fields * embed_dim]
            uv_x = self.u[i](v_x)
            # [batch_size, num_fields * embed_dim] + [input_dim, ] -> [batch_size, num_fields * embed_dim]
            # x_0 * x_w: [batch_size, num_fields * embed_dim] * [batch_size, num_fields * embed_dim]
            # -> [batch_size, num_fields * embed_dim] Hadamard-product keep same dims
            # [batch_size, num_fields * embed_dim] + [batch_size, num_fields * embed_dim]
            # -> [batch_size, num_fields * embed_dim]
            x = x_0 * (uv_x + self.b[i]) + x

        return x