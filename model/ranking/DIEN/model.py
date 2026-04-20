import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from notes.ranking.layer.layer import Dice, FeaturesEmbedding


class Attention(nn.Module):
    """
    Attention score, a_t = exp(h_t * W * e_a) / sum_j_t(exp(h_j * W * e_a))
    Use first layer GRU hidden state output, weight matrix and behavior_ad_embedding to get attention score
    """
    def __init__(self, embed_dims, attention_dim):
        super().__init__()
        # get two embed dims, one from GRU hidden state output, another from behavior_ad_embedding
        embed_dim1, embed_dim2 = embed_dims
        self.mlp = nn.Sequential(
            # mlp layer -> [embed_dim1 + embed_dim2 + embed_dim1 * embed_dim2, attention_dim]
            nn.Linear(embed_dim1 + embed_dim2 + embed_dim1 * embed_dim2, attention_dim),
            # activation layer not change shape, [embed_dim1 + embed_dim2 + embed_dim1 * embed_dim2, attention_dim]
            Dice(),
            # mlp layer -> [attention_dim, 1]
            nn.Linear(attention_dim, 1),
        )

    def forward(self, packed: PackedSequence, query):
        """
        :param packed: PackedSequence
        :param query: [batch_size, embed_dim]
        :return: [total_timestep, 1]
        """
        # unpacked the PackedSequence
        # query shape -> [batch_size, embed_dim] ex: [64, 10]
        # x shape -> [total_timestep, embed_dim] ex: [42160, 10]
        x, batch_sizes, sorted_indices, unsorted_indices = packed
        # sorted query according to packed sorted original batch index
        query = query[sorted_indices]
        # get each timestep all index, ex: batch_size: 64, which means first timestep all 64 sample in batch exist
        # and this index is where located in x matrix
        idx_list = []
        for batch_size in batch_sizes:
            idx_list.extend(range(batch_size))
        # query -> [batch_size, embed_dim] ex: [64, 10] -> [total_timestep, embed_dim] ex: [42160, 10]
        # for each sample each timestep build corresponding query embed
        query = query[idx_list]

        # outer product
        i1, i2 = [], []
        for i in range(x.shape[-1]):
            for j in range(query.shape[-1]):
                i1.append(i)
                i2.append(j)
        # x: [total_timestep, 1] * query: [total_timestep, 1] -> [total_timestep, 1]
        # we have embed_dim1 * embed_dim2 combination
        # mul will keep same output shape as input then we have [total_timestep, embed_dim1 * embed_dim2] output
        # -> reshape -> [total_timestep, embed_dim1 * embed_dim2]
        # here use outer product, each gru output will multipy query (ad) in all embedd dim combination
        p = x[:, i1].mul(query[:, i2]).reshape(x.shape[0], -1)

        # horizontal stack -> [total_timestep, embed_dim1] + [total_timestep, embed_dim1 * embed_dim2]
        # + [total_timestep, embed_dim2] -> [total_timestep, embed_dim1 * embed_dim2 + embed_dim1 + embed_dim2]
        # mlp -> [total_timestep, embed_dim1 * embed_dim2 + embed_dim1 + embed_dim2]
        # * [embed_dim1 * embed_dim2 + embed_dim1 + embed_dim2, attention_dim] * [attention_dim, 1]
        # -> [total_timestep, 1]
        # here use mlp layer to calculate gru, product, query (ad) cross interaction raw attention score
        att = self.mlp(torch.hstack([x, p, query]))

        return att


class AUGRUCell(nn.Module):
    """
    AUGRU one cell implementation,
    update gate u_t = w_u * i_t + u_u * h_t-1 + b_u
    reset gate r_t = w_r * i_t + u_r * h_t-1 + b_r
    candidate gate ~h_t = w_h * i_t + r_t mul u_h * h_t-1 + b_h
    defined update gate ~u_t = a_t * u_t
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # GRU update gate
        self.update_gate = nn.Sequential(
            # linear layer [input_size + hidden_size, 1], ex: [10 + 10, 1]
            # u_t = w_u * i_t + u_u * h_t-1 + b_u, here we use one weight matrix to train fit both i_t and h_t-1 stacked
            nn.Linear(input_size + hidden_size, 1),
            nn.Sigmoid()
        )
        # GRU reset gate
        self.reset_gate = nn.Sequential(
            # linear layer [input_size + hidden_size, 1], ex: [10 + 10, 1]
            # r_t = w_r * i_t + u_r * h_t-1 + b_r
            nn.Linear(input_size + hidden_size, 1),
            nn.Sigmoid()
        )

        # GRU candidate gate
        self.candidate = nn.Sequential(
            # linear layer [input_size + hidden_size, hidden_size],  ex: [10 + 10, 10]
            # ~h_t = w_h * i_t + r_t mul u_h * h_t-1 + b_h
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, x, h, att):
        """
        :param x: [timestep_length, embed_dim]
        :param h: [timestep_length, hidden]
        :param att: [timestep_length, 1]
        :return: [timestep_length, hidden]
        """
        # x -> [timestep_length, embed_dim] ex: [64, 10] h -> [timestep_length, hidden_size] ex: [64, 10]
        # stack -> [timestep_length, embed_dim + hidden_size] ex: [64, 10 + 10]
        # update_gate -> [64, 10 + 10] * [10 + 10, 1] -> [64, 1]
        u = self.update_gate(torch.hstack([x, h]))
        # att -> [timestep_length, 1] ex: [64, 1] * u -> [64, 1] -> [timestep_length, 1]
        # get new defined update gate ~u_t = a_t * u_t in paper
        u = att * u
        # x -> [timestep_length, embed_dim] ex: [64, 10] h -> [timestep_length, hidden_size] ex: [64, 10]
        # stack -> [timestep_length, embed_dim + hidden_size] ex: [64, 10 + 10]
        # reset_gate -> [64, 10 + 10] * [10 + 10, 1] -> [64, 1]
        r = self.reset_gate(torch.hstack([x, h]))
        # get ~h_t = w_h * i_t + r_t mul u_h * h_t-1 + b_h
        # x -> [timestep_length, embed_dim] ex: [64, 10]
        # h * r -> [timestep_length, hidden_size] * [timestep_length, 1] ex: [64, 10] * [64, 1] -> [64, 10]
        # stack -> [timestep_length, embed_dim + hidden_size] ex: [64, 10 + 10]
        # candidate_gate -> [64, 10 + 10] * [10 + 10, 10] -> [64, 10]
        tilde_h = self.candidate(torch.hstack([x, h * r]))
        # (1 - [timestep_length, 1]) * [timestep_length, hidden] + [timestep_length, 1] * [timestep_length, hidden]
        # [timestep_length, hidden] + [timestep_length, hidden] - > [timestep_length, hidden]
        # update next timestep input hidden state, h_t = (1- ~u_t) * h_t-1 + ~u_t * tilde_h
        h = (1 - u) * h + u * tilde_h

        return h


class AUGRU(nn.Module):
    """
    AUGRU implementation called Interest Evolution Layer in paper
    """
    def __init__(self, input_size, hidden_size, embed_dim, attention_dim):
        super().__init__()
        # hidden size
        self.hidden_size = hidden_size
        # attention layer, use first layer gru output and behaviors embedding get attention scores
        self.attention = Attention([hidden_size, embed_dim], attention_dim)
        # self defined augru cell, here input_size is i_t dim, hidden_size is h_t dim
        self.augru_cell = AUGRUCell(input_size, hidden_size)

    def forward(self, packed: PackedSequence, query, h=None):
        """
        :param packed: PackedSequence
        :param query: [batch_size, embed_dim]
        :param h: [first_timestep_size, hidden_size] or None
        :return: PackedSequence and output_h: [first_timestep_size, hidden_size]
        """
        # unpacked the PackedSequence
        # x -> [42160, 10], batch_size -> [40, ]
        x, batch_sizes, sorted_indices, unsorted_indices = packed
        # get raw attention score
        # query -> [64, 10], last behaviors_ad_embeddings which is ads embedding output for each sample
        # attention -> [42160, 1]
        att = self.attention(packed, query)
        device = x.device
        if h is None:
            # init h_t0 will be all 0 since first timestep -> [first_timestep_size, hidden_size] ex: [64, 10]
            h = torch.zeros(batch_sizes[0], self.hidden_size, device=device)

        # init placeholder output to store each timestep all sample hidden state output
        # -> [total_timestep, hidden_size] ex: [42160, 10]
        output = torch.zeros(x.shape[0], self.hidden_size)
        # init output_h placeholder -> [first_timestep_size, hidden_size] ex: [64, 10]
        output_h = torch.zeros(batch_sizes[0], self.hidden_size, device=device)

        start = 0
        # loop through each batch size to unpacked x
        for batch_size in batch_sizes:
            # extract each batch same timestep, ex: x[0: 0 + 64] this is first timestep in all samples packed in x
            _x = x[start: start + batch_size]
            # extract corresponding attention score ex: att[0: 0 + 64]
            _att = att[start: start + batch_size]
            # extract hidden state, since h use first timestep longest sample as init, always extract from 0 index
            _h = h[:batch_size]
            # get hidden state output from current timestep and update h -> [timestep_length, hidden] ex: [64, 10]
            h = self.augru_cell(_x, _h, _att)
            # store current timestep all sample hidden state output into corresponding index
            output[start: start + batch_size] = h
            # store current hidden state output, this will always be updated by current timestep
            output_h[:batch_size] = h
            # update next start index on packed x
            start += batch_size

        return PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices), output_h[unsorted_indices]


class DeepInterestEvolutionNetworkModel(nn.Module):
    """
    A pytorch implementation of DIEN.
    """
    def __init__(self, field_dims, embed_dim, attention_dim, mlp_dims):
        super().__init__()
        # keep hidden size same as embedding dim
        hidden_size = embed_dim
        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embed = FeaturesEmbedding(field_dims, embed_dim)
        # gru cell, [embed_dim, hidden_size]
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        # self defined augru cell, here input_size (embed_dim), hidden_size is same
        self.augru = AUGRU(hidden_size, hidden_size, embed_dim, attention_dim)
        # mlp layer
        self.mlp = nn.Sequential(
            # input will be last timestep augru output hidden state and ad embed horizontal concat result
            # mlp layer -> [embed_dim + hidden_size, mlp_dims[0] ex: 200]
            nn.Linear(embed_dim + hidden_size, mlp_dims[0]),
            Dice(),
            # mlp layer -> [mlp_dims[0] ex: 200, mlp_dims[1] ex: 100]
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            Dice(),
            # mlp layer -> [mlp_dims[1] ex: 100, 1]
            nn.Linear(mlp_dims[1], 1)
        )

    def forward(self, x, neg_sample=None):
        """
        :param x: [batch_size, num_fields]
        :param neg_sample: [batch_size, num_fields - 2]
        :return: output: [batch_size, 1], auxiliary_output: [2, all_timestep]
        """
        # neg_sample -> [batch_size, num_fields - 2] ex: [64, 41 - 2]
        # x -> [batch_size, num_fields] ex: [64, 41]
        # embedding(x) -> [batch_size, num_behaviors + 1, embed_dim]
        behaviors_ad_embeddings = self.embed(x)
        # x[:, :-1] > 0 -> [64, 40] -> sum -> [64, ], get each batch no zero length
        lengths = (x[:, :-1] > 0).sum(dim=1).cpu()
        # pack padded sequence will flatten each sample into one dims, this will help accelerate rnn model run
        # data: [42160, 10], each row is same timestep across whole batch
        # batch_size: [40, ], each element represent each timestep length  across whole batch
        packed_behaviors = pack_padded_sequence(behaviors_ad_embeddings, lengths, batch_first=True,
                                                enforce_sorted=False)
        # get gru output, here gru accept pack_padded_sequence object as input
        packed_gru_output, _ = self.gru(packed_behaviors)
        # same packed output object, data: [42160, 10], batch_size: [40, ], same meaning as input packed
        # contain the output features (h_t) from the last layer of the GRU, for each timestep and packed across dim=0
        augru_output, h = self.augru(packed_gru_output, behaviors_ad_embeddings[:, -1])
        # concat last timestep augru hidden state and ad embedding as input mlp layer
        # h -> [last_timestep_size, hidden_size] ex: [64, 10], ad_embedd -> [batch_size, embed_dim]
        # stack -> [batch_size, hidden_size + embed_dim] ex: [64, 10 + 10]
        concated = torch.hstack([h, behaviors_ad_embeddings[:, -1]])
        # [batch_size, hidden_size + embed_dim] * ... * [mlp_dims, 1] -> [batch_size, 1]
        output = self.mlp(concated)
        # apply sigmoid -> [batch_size, 1]
        output = torch.sigmoid(output)

        # if no negative sample means only use last layer output to calculate loss
        if neg_sample is None:
            # return output directly
            return output
        else:
            # otherwise add auxiliary loss part
            # unpack the augru all timestep and all sample each hidden state output
            # gru_output -> [batch_size, num_behaviors (timestep length), embed_dim]
            gru_output, _ = pad_packed_sequence(packed_gru_output, batch_first=True)
            # get label index in gru output gru_output[:, 1:] -> [batch_size, num_behaviors - 1, embed_dim]
            # get all no padding index gru output embedding, here flatten batch_size and length into 1 dim
            # [batch_size, num_behaviors - 1, embed_dim][batch_size, num_behaviors - 1] -> [all_timestep, embed_dim]
            gru_embedding = gru_output[:, 1:][neg_sample > 0]

            # get positive label index embedding
            # behaviors_ad_embeddings[:, 1:-1] -> [batch_size, num_behaviors - 1, embed_dim]
            # same as above get corresponding no padding embedding and flatten
            # [batch_size, num_behaviors - 1, embed_dim][batch_size, num_behaviors - 1] -> [all_timestep, embed_dim]
            pos_embedding = behaviors_ad_embeddings[:, 1:-1][neg_sample > 0]
            # get negative label index embedding by lookup embedding table
            # embed(x) -> [batch_size, num_behaviors - 1, embed_dim]
            # same as above get corresponding no padding embedding and flatten
            # [batch_size, num_behaviors - 1, embed_dim][batch_size, num_behaviors - 1] -> [all_timestep, embed_dim]
            neg_embedding = self.embed(neg_sample)[neg_sample > 0]

            # add log loss inner product part for both positive and negative
            # from paper is sigmoid(h_t * e_positive) and sigmoid(h_t * e_negative)
            # [all_timestep, embed_dim] * [all_timestep, embed_dim] -> sum -> [all_timestep, ]
            pred_pos = (gru_embedding * pos_embedding).sum(dim=1)
            pred_neg = (gru_embedding * neg_embedding).sum(dim=1)
            # cat -> [all_timestep, ] + [all_timestep, ] -> [all_timestep * 2, ]
            # sigmoid -> [all_timestep * 2, ] -> reshape -> [2, all_timestep]
            auxiliary_output = torch.sigmoid(torch.cat([pred_pos, pred_neg], dim=0)).reshape(2, -1)

            return output, auxiliary_output
