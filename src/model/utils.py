import torch


def positional_embedding(src_tokens, embed_dim):
    # TODO: implement actual positional encoding
    size = src_tokens.size()
    size = list(size + (embed_dim, ))
    return torch.zeros(size, device=src_tokens.device)


def create_mask(lengths, max_length=None):
    """create_mask

    :param lengths: [B]
    :param max_length: int

    returns [B, T]
    """
    if max_length is None:
        max_length = lengths.max()
    index = torch.arange(max_length, device=lengths.device)
    return index[None, :] >= lengths[:, None]
