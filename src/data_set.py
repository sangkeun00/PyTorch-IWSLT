import numpy as np
import torch
from torch.utils.data import DataLoader


def to_device(iterator, device):
    for seq in iterator:
        if isinstance(seq, list):
            yield [x.to(device, non_blocking=True) for x in seq]
        elif isinstance(seq, torch.Tensor):
            yield seq.to(device, non_blocking=True)
        else:
            raise NotImplementedError()


def get_dataloader(dset, batch_size, shuffle=True, num_workers=2):
    vocab = dset.vocab
    pad_id = vocab.PAD_ID

    def pad_seqs(batch):
        length = np.array([len(item) for item in batch])
        max_length = length.max()

        data = np.array(
            [(item + [pad_id] * (max_length - len(item))) for item in batch])
        data = torch.tensor(data, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        return data, length

    def my_collate(batch):
        src_seqs, src_lengths = pad_seqs([item[0] for item in batch])
        tgt_seqs, tgt_lengths = pad_seqs([item[1] for item in batch])
        return [src_seqs, src_lengths, tgt_seqs, tgt_lengths]

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate,
        pin_memory=True)
    return dataloader
