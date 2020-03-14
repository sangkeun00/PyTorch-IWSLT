import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()

        assert embed_dim % 2 == 0

        self.embed_dim = embed_dim
        half_dim = embed_dim // 2

        scale = math.log(10000.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim).float() * -scale).unsqueeze(0)
        pos = torch.arange(max_len).float().unsqueeze(1)
        pe = emb * pos
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1)
        self.register_buffer('pe', pe)

    def forward(self, tokens):
        out = self.pe[:tokens.shape[1], :]
        out = out.unsqueeze(0)

        return out.detach()
