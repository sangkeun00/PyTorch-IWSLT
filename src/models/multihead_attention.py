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
        mode=None
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
        key,
        value,
        mask=None
    ):
        tgt_len, bsz, qdim = query.size()
        assert qdim == self.embed_dim
        assert key.size(2) == self.kdim and value.size(2) == self.vdim

        q = self.q(query).contiguous()
        k = self.k(key).contiguous()
        v = self.v(value).contiguous()

        # qkv: (T, B, C) -> (T, B, H, C//H)
        q = q.view(q.size(0), bsz, self.num_heads, self.head_dim)
        k = k.view(k.size(0), bsz, self.num_heads, self.head_dim)
        v = v.view(v.size(0), bsz, self.num_heads, self.head_dim)

        attn = torch.einsum('ibhd,jbhd->bhij', q, k)
        attn *= self.scale

        attn = attn.masked_fill_(mask[:, None, :, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum('bhij,jbhd->ibhd', attn, v)

        out = out.reshape(out.size(0), out.size(1), self.embed_dim).contiguous()

        out = self.out(out)

        return out


if __name__ == '__main__':
    edim = 16
    mha = MultiHeadAttention(embed_dim=edim, attn_dropout=0, num_heads=4)

    src_len = 7
    tgt_len = 4
    bsz = 11
    q = torch.randn(tgt_len, bsz, edim)
    k = torch.randn(src_len, bsz, edim)
    v = torch.randn(src_len, bsz, edim)

    mask = (torch.randn(bsz, tgt_len, src_len) > 0)

    out = mha(q, k, v, mask)

    print(out)
