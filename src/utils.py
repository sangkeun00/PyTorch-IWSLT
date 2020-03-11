import torch


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
