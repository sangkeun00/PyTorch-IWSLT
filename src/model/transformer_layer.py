import torch
import torch.nn as nn
import torch.nn.functional as F

from multihead_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        num_heads,
        attn_dropout=0.,
        act_dropout=0.,
        dropout=0.,
        layernorm_before=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.layernorm_before = layernorm_before
        self.act_dropout = act_dropout
        self.dropout = dropout

        # self-attention part
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            mode='fc',
        )
        self.attn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        # point-wise ffn
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, mask):
        # self-attn part
        identity = x
        if self.layernorm_before:
            x = self.attn_layernorm(x)
        x = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.attn_layernorm(x)

        # point-wise ffn part
        identity = x
        if self.layernorm_before:
            x = self.ffn_layernorm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.ffn_layernorm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
