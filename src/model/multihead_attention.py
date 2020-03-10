import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=4,
        attn_dropout=0.,
        bias=False,
        kdim=None,
        vdim=None,
        mode=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim

        self.q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self, g=1.0):
        nn.init.xavier_uniform_(self.q.weight, gain=g)
        nn.init.xavier_uniform_(self.k.weight, gain=g)
        nn.init.xavier_uniform_(self.v.weight, gain=g)
        nn.init.xavier_uniform_(self.out.weight)

        if self.bias:
            nn.init.constant_(self.q.bias, 0)
            nn.init.constant_(self.k.bias, 0)
            nn.init.constant_(self.v.bias, 0)
            nn.init.constant_(self.out.bias, 0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        mask=None,
        cache=None
    ):
        tgt_len, bsz, qdim = query.size()
        assert qdim == self.embed_dim
        assert (key is None) == (value is None)
        if key is None:
            assert cache is not None and 'enc_key' in cache

        query = self.q(query)
        if key is None:
            key, value = cache['enc_key'], cache['enc_value']
        else:
            key = self.k(key).view(-1, bsz, self.num_heads, self.kdim)
            value = self.v(value).view(-1, bsz, self.num_heads, self.vdim)

            if cache is not None:
                cache['enc_key'], cache['enc_value'] = key, value


