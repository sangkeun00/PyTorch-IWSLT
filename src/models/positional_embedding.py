import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=1026):
        super().__init__()

        assert embed_dim % 2 == 0

        self.embed_dim = embed_dim
        half_dim = embed_dim // 2

        # scale = 1/(10000^(i/(hd-1)))
        scale = torch.exp(-torch.arange(half_dim).float() / (half_dim - 1) *
                          math.log(10000.))
        # pe = pos/(10000^(i/(hd-1)))
        pe = torch.arange(max_len).float()[:, None] * scale[None, :]
        # pe = sin(pos/(10000^(i/(hd-1)))) & cos(pos/(10000^(i/(hd-1))))
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1)

        self.register_buffer('pe', pe)

    def forward(self, tokens, shift=0):
        out = self.pe[shift:shift + tokens.shape[1], :]
        out = out.unsqueeze(0)

        return out.detach()
