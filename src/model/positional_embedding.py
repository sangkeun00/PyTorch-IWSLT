import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        assert embed_dim % 2 == 0

        self.embed_dim = embed_dim
        half_dim = embed_dim

        scale = math.log(10000.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim).float() * -scale)
        self.register_buffer('emb', emb)

    def forward(self, tokens):
        pos = torch.arange(len(tokens), device=tokens.device).float()
        out = self.emb.unsqueeze(0) * pos.unsqueeze(1)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        out = out.unsqueeze(0)

        assert out.size() == (1, len(tokens), self.embed_dim)

        return out
