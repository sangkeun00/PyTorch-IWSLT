import math

import torch
import torch.nn.functional as F


def masked_nll(logits, lengths, targets, label_smoothing=0.0, pad_id=0):
    """masked_nll

    :param logits: [B, T, C]
    :param lengths: [B]
    :param targets: [B, T]
    :param label_smoothing: [0., 1.]
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    n_token = log_probs.size()[-1]

    tgt_one_hots = F.one_hot(targets, n_token).to(logits.dtype)
    mask = (targets == pad_id).to(logits.dtype)
    inp_q = 1. - mask
    nll = -(log_probs * tgt_one_hots).sum(dim=-1)

    if label_smoothing > 0.:
        loss = (nll * (1 - label_smoothing) -
                log_probs.mean(dim=-1) * label_smoothing)
        return (inp_q * loss).sum() / inp_q.sum()
    else:
        return (inp_q * nll).sum() / inp_q.sum()


def create_causual_mask(size, dtype=torch.float32):
    index = torch.arange(size)
    causal_mask = index[:, None] < index[None, :]
    shape = causal_mask.size()
    causal_mask = torch.where(causal_mask,
                              torch.ones(shape, dtype=dtype) * float('-inf'),
                              torch.zeros(shape, dtype=dtype))
    return causal_mask
