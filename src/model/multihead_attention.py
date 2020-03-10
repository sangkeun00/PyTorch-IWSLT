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
        else:
            assert key.size(2) == self.kdim and value.size(2) == self.vdim

        """
        # qkv: (T, B, C) -> (B, H, T, C//H)
        q = self.q(query) * self.scale
        q = q.contiguous().view(-1, bsz * self.num_heads, self.head_dim).permute(1, 0, 2)
        if key is None:
            k, v = cache['enc_key'], cache['enc_value']
        else:
            k, v = self.k(key), self.v(value)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).permute(1, 0, 2)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).permute(1, 0, 2)
            if cache is not None:
                cache['enc_key'], cache['enc_value'] = k, v

        # qkv: (B * H, T, C//H), attn: (B * H, T, T) -> out: (B * H, T, C//H)
        attn = torch.bmm(q, k.transpose(1, 2)).view(bsz, self.num_heads, length, -1)
        attn = attn.masked_fill_(mask.unsqueeze(1), -1e9)
        attn = attn.view(bsz * self.num_heads, length, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.bmm(attn, v)


        # out: (B, H, T, C//H) -> (T, B, C)
        out = out.transpose(0, 1).contiguous()
        out = out.view(length, bsz, self.embed_dim)

        out = self.out(out)

        return out
        """
        # qkv: (T, B, C) -> (B, H, T, C//H)
        q = self.q(query).contiguous()
        q = q.view(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        if key is None:
            k, v = cache['enc_key'], cache['enc_value']
        else:
            k = self.k(key).contiguous()
            v = self.v(value).contiguous()
            k = k.view(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.view(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            if cache is not None:
                cache['enc_key'], cache['enc_value'] = k, v
        q *= self.scale

        # qkv: (B, H, T, C//H) -> attn: (B, H, T, T), out: (B, H, T, C//H)
        attn = torch.matmul(q, k.transpose(2, 3))
        attn = attn.masked_fill_(mask.unsqueeze(1), -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)

        # out: (B, H, T, C//H) -> (T, B, C)
        out = out.permute(2, 0, 1, 3).contiguous()
        out = out.view(-1, bsz, self.embed_dim)

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
