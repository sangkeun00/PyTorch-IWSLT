import math

import torch

"""
def positional_embedding(src_tokens, embed_dim):
    # TODO: implement actual positional encoding
    # size = src_tokens.size()
    # size = list(size + (embed_dim, ))
    # return torch.zeros(size, device=src_tokens.device)
    assert embed_dim % 2 == 0

    max_len = src_tokens.shape[0]
    half_dim = embed_dim // 2
    scale = math.log(10000.) / (half_dim - 1)
    coef = torch.arange(half_dim).float().to(src_tokens)
    coef = torch.exp(coef * -scale) # (embed_dim//2, )

    pos = torch.arange(max_len).float().to(src_tokens) # (max_len, )
    emb = coef.unsqueeze(0) * pos.unsqueeze(1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    assert emb.size() == (max_len, embed_dim)

    return emb.unsqueeze(0)
"""

def create_mask(lengths, max_length=None, causal=False):
    """create_mask

    :param lengths: [B]
    :param max_length: int

    returns [B, T or 1, T]
    """
    if max_length is None:
        max_length = lengths.max()
    index = torch.arange(max_length, device=lengths.device)
    padding_mask = index[None, :] >= lengths[:, None]
    if causal:
        causal_mask = index[:, None] < index[None, :]
        return (causal_mask[None, :, :] | padding_mask[:, None, :])

    return padding_mask[:, None, :]
