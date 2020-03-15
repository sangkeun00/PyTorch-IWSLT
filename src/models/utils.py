import torch


def create_causual_mask(size, dtype=torch.float32):
    index = torch.arange(size)
    causal_mask = index[:, None] < index[None, :]
    shape = causal_mask.size()
    causal_mask = torch.where(causal_mask,
                              torch.ones(shape, dtype=dtype) * float('-inf'),
                              torch.zeros(shape, dtype=dtype))
    return causal_mask


def cache_states(cache, key, tensor, dim=0):
    if cache is not None:
        if key not in cache:
            cache[key] = tensor
        else:
            cache[key] = torch.cat([cache[key], tensor], dim=dim)


def get_states(cache, key):
    if cache is not None and key in cache:
        return cache[key]
