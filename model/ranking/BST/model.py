import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import numpy as np
from notes.ranking.layer.layer import FeaturesEmbedding, MultiLayerPerceptron


class TransformerBlock(nn.Module):
    """
    A pytorch implementation of TransformerBlock from BST paper.
    S′ = LayerNorm(S + Dropout(MH(S))
    F = LayerNorm(S′ + Dropout(LeakyReLU(S′W_(1) + b_(1))W_(2) + b_(2)))
    """

    def __init__(self, attn_embed_dim=64, ffn_dim=64, num_heads=8, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super().__init__()
        # build self-attention layer, with dropout
        self.attention = MultiheadAttention(attn_embed_dim,
                                            num_heads=num_heads,
                                            dropout=attn_dropout,
                                            batch_first=True)
        # linear layer
        # mlp -> [attn_embed_dim ex: 64, ffn_dim ex: 64] -> LeakyRelu -> mlp [ffn_dim, attn_embed_dim]
        self.ffn = nn.Sequential(nn.Linear(attn_embed_dim, ffn_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(ffn_dim, attn_embed_dim))
        # use residual nor not
        self.use_residual = use_residual
        # drop out layer after self-attention
        self.dropout1 = nn.Dropout(net_dropout)
        # drop out layer after ffn
        self.dropout2 = nn.Dropout(net_dropout)
        # ln layer after self-attention
        self.layer_norm1 = nn.LayerNorm(attn_embed_dim) if layer_norm else None
        # ln layer after ffn
        self.layer_norm2 = nn.LayerNorm(attn_embed_dim) if layer_norm else None

    def forward(self, x, attn_mask=None):
        """
        :param x: [batch_size, seq_len, embed_dim]
        :return: [batch_size, seq_len, attn_embed_dim]
        """
        # S′ = LayerNorm(S + Dropout(MH(S))
        # self-attention -> [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, attn_embed_dim]
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        # attn -> dropout -> [batch_size, seq_len, attn_embed_dim]
        s = self.dropout1(attn)
        # s -> [batch_size, seq_len, attn_embed_dim] + [batch_size, seq_len, attn_embed_dim]
        # -> [batch_size, seq_len, attn_embed_dim]
        if self.use_residual:
            s += x
        # ln -> [batch_size, seq_len, attn_embed_dim]
        if self.layer_norm1 is not None:
            s = self.layer_norm1(s)

        # F = LayerNorm(S′ + Dropout(LeakyReLU(S′W_(1) + b_(1))W_(2) + b_(2))) ffn -> [batch_size, seq_len,
        # attn_embed_dim] -> [batch_size, seq_len, ffn_dim] -> [batch_size, seq_len, attn_embed_dim] dropout
        # -> [batch_size, len, attn_embed_dim]
        out = self.dropout2(self.ffn(s))
        # out -> [batch_size, seq_len, attn_embed_dim] + [batch_size, seq_len, attn_embed_dim]
        # -> [batch_size, seq_len, attn_embed_dim]
        if self.use_residual:
            out += s
        # ln -> [batch_size, seq_len, attn_embed_dim]
        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out


class BehaviorTransformer(nn.Module):
    def __init__(self,
                 seq_len=1,
                 attn_embed_dim=64,
                 num_heads=8,
                 stacked_transformer_layers=1,
                 attn_dropout=0.0,
                 net_dropout=0.0,
                 use_position_emb=True,
                 position_dim=4,
                 layer_norm=True,
                 use_residual=True):
        super().__init__()
        # use position embedding dim size
        # the formular in paper pos(vi) = t(vt) − t(vi)
        # but in source code use sin/cos position embedding
        self.position_dim = position_dim
        # use position embedding or not
        self.use_position_emb = use_position_emb
        # build transformer blocks, here self-attention output dims and ffn input dims are same
        self.transformer_blocks = nn.ModuleList(TransformerBlock(attn_embed_dim=attn_embed_dim,
                                                                 ffn_dim=attn_embed_dim,
                                                                 num_heads=num_heads,
                                                                 attn_dropout=attn_dropout,
                                                                 net_dropout=net_dropout,
                                                                 layer_norm=layer_norm,
                                                                 use_residual=use_residual)
                                                for _ in range(stacked_transformer_layers))
        # add position embedding layer
        if self.use_position_emb:
            # position embedding -> [sequence_len, position_dim]
            # each use behavior like token in sequence, each will with a position_dim ex: 4 size embedding
            self.position_emb = nn.Parameter(torch.Tensor(seq_len, position_dim))
            # init position embedding with sin/cos pe
            # here it's different compare with original paper
            self.reset_parameters()

    def reset_parameters(self):
        # get user history sequence length
        seq_len = self.position_emb.size(0)
        # init position embedding all zeros -> [seq_len, position_dim]
        pe = torch.zeros(seq_len, self.position_dim)
        # init sequence [0, 1, 2, ..., seq_len - 1] -> [[0, 1, 2, ..., seq_len - 1]] -> size: [1, len(seq_len)]
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        # here is exp(1 / (log(10000)^(2 * i / d_model))), here i is index of position embedding
        div_term = torch.exp(torch.arange(0, self.position_dim, 2).float() * (-np.log(10000.0) / self.position_dim))
        # sin and cos position embedding
        # pe[:, 0::2] pick all even index from last dim, then pe[:, 1::2] pick all even index
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # assign position embedding with sin and cos embedding result pe, this is not trainable
        self.position_emb.data = pe

    def forward(self, x, attn_mask=None):
        """
        :param x: [batch_size, seq_len, embed_dim]
        :return: [batch_size, seq_len, attn_embed_dim]
        """
        if self.use_position_emb:
            # pe -> [seq_len, position_dim] -> unsqueeze -> [1, seq_len, position_dim] -> repeat
            # -> [batch_size, seq_len, position_dim]
            # cat -> [batch_size, seq_len, embed_dim] + [batch_size, seq_len, position_dim]
            # -> [batch_size, seq_len, embed_dim + position_dim]
            # same as paper, concat input x embedding with position embedding
            x = torch.cat([x, self.position_emb.unsqueeze(0).repeat(x.size(0), 1, 1)], dim=-1)

        # run transformer blocks loop
        for i in range(len(self.transformer_blocks)):
            # x -> [batch_size, seq_len, embed_dim + position_dim] -> transformer
            # -> [batch_size, seq_len, attn_embed_dim]
            x = self.transformer_blocks[i](x, attn_mask=attn_mask)

        return x


class BehaviorSequenceTransformerModel(nn.Module):
    """
    A pytorch implementation of BST model.
    Here only support one sequence type and no other user feature to use
    """

    def __init__(self,
                 field_dims,
                 sequence_length,
                 mlp_dims,
                 num_heads=2,
                 stacked_transformer_layers=1,
                 attention_dropout=0,
                 embed_dim=10,
                 net_dropout=0,
                 layer_norm=True,
                 use_residual=True,
                 seq_pooling_type="mean",
                 use_position_emb=True,
                 ):
        super().__init__()

        self.seq_pooling_type = seq_pooling_type
        self.field_dims = field_dims
        self.embedding_dim = embed_dim
        self.num_heads = num_heads

        if seq_pooling_type not in ["mean", "sum", "target", "concat"]:
            raise ValueError(
                f"Sequence pooling type {seq_pooling_type} is not supported."
            )

        # add target item, update sequence which will be passed into self-attention
        sequence_length += 1

        # build embedding layer for input x to transfer sparse tensor to dense vector, ex: [sum(fields_dims), embed_dim]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # calculate after embedding and concat what's input dim of self-attention which is same as attn embedding
        # here we only have one sequence type feature
        # ex: 10 * (1 + 1) when concat position embedding and sequence embedding
        attn_embed_dim = embed_dim * (1 + int(use_position_emb))
        # get sequence out dim for mlp input
        # if concat then sequence output from self-attention will be flattened
        seq_out_dim = attn_embed_dim * sequence_length if self.seq_pooling_type == "concat" else attn_embed_dim

        # build transformer encoder
        self.transformer_encoder = BehaviorTransformer(
            seq_len=sequence_length,
            attn_embed_dim=attn_embed_dim,
            num_heads=num_heads,
            stacked_transformer_layers=stacked_transformer_layers,
            attn_dropout=attention_dropout,
            net_dropout=net_dropout,
            position_dim=embed_dim,
            use_position_emb=use_position_emb,
            layer_norm=layer_norm,
            use_residual=use_residual
        )

        # mlp layer dims, ex: [embed_output_dim, 128, 64, 32, 16, 1]
        # here input will be only sequence self-attention output, we don't have other user features to concat
        self.mlp = MultiLayerPerceptron(seq_out_dim, mlp_dims, net_dropout)

    def sequence_pooling(self, transformer_out, mask):
        # sequence pooling depends on different pooling type
        # here need add back mask, since after transformer original mask behaviors may have some value but need set to 0
        if self.seq_pooling_type == "mean":
            # [batch_size, num_behaviors + 1, attn_embed_dim] * [batch_size, num_behaviors + 1, 1] -> sum
            # -> [batch_size, attn_embed_dim] / [batch_size, 1] -> [batch_size, attn_embed_dim]
            return (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1.e-12)
        elif self.seq_pooling_type == "sum":
            # [batch_size, num_behaviors + 1, attn_embed_dim] * [batch_size, num_behaviors + 1, 1] -> sum
            # -> [batch_size, attn_embed_dim]
            return (transformer_out * mask).sum(dim=1)
        elif self.seq_pooling_type == "target":
            # [batch_size, attn_embed_dim]
            # here just select last one target transformer as output
            return transformer_out[:, -1, :]
        elif self.seq_pooling_type == "concat":
            # [batch_size, num_behaviors + 1, attn_embed_dim] -> flatten
            # -> [batch_size, (num_behaviors + 1) * attn_embed_dim]
            return transformer_out.flatten(start_dim=1)
        else:
            return None

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # create mask for padding behaviors and add one dim for embedding multipy -> [batch_size, num_behaviors + 1, 1]
        mask = (x > 0).float().unsqueeze(-1)
        # embedding(x) -> [batch_size, num_behaviors + 1, embed_dim] ex: [64, 40 + 1, 8]
        # mul [batch_size, num_behaviors + 1, 1] -> [batch_size, num_behaviors + 1, embed_dim]
        # convert padding behaviors embedding into all zeros
        behaviors_ad_embeddings = self.embedding(x).mul(mask)
        # [batch_size, num_behaviors + 1, embed_dim] -> transformer -> [batch_size, num_behaviors + 1, attn_embed_dim]
        transformer_out = self.transformer_encoder(behaviors_ad_embeddings)
        # pooling transformer out for pass into mlp
        # [batch_size, num_behaviors + 1, attn_embed_dim]
        # -> [batch_size, attn_embed_dim] or [batch_size, (num_behaviors + 1) * attn_embed_dim]
        pooling_emb = self.sequence_pooling(transformer_out, mask)
        # [batch_size, attn_embed_dim] * ... * [mlp_dims, 1] -> [batch_size, 1]
        output = self.mlp(pooling_emb)

        return torch.sigmoid(output)
