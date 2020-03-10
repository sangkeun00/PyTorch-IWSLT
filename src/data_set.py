import os
from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SplittedDataset(object):
    def __init__(self,
                 input_dir,
                 lowercase=False,
                 lang_src='en',
                 lang_tgt='de',
                 path_format='{split}.bpe.{lang}',
                 attach_ends=True):
        self.attach_ends = attach_ends

        trn_src_path = os.path.join(
            input_dir, path_format.format(split='train', lang=lang_src))
        trn_tgt_path = os.path.join(
            input_dir, path_format.format(split='train', lang=lang_tgt))

        val_src_path = os.path.join(
            input_dir, path_format.format(split='dev', lang=lang_src))
        val_tgt_path = os.path.join(
            input_dir, path_format.format(split='dev', lang=lang_tgt))

        tst_src_path = os.path.join(
            input_dir, path_format.format(split='test', lang=lang_src))
        tst_tgt_path = os.path.join(
            input_dir, path_format.format(split='test', lang=lang_tgt))

        # load vocabularies
        vocab_src_path = os.path.join(
            input_dir, path_format.format(split='vocab', lang=lang_src))
        vocab_tgt_path = os.path.join(
            input_dir, path_format.format(split='vocab', lang=lang_tgt))

        self.vocab_src = Vocab.load(vocab_src_path,
                                    train_path=trn_src_path,
                                    lowercase=lowercase)
        self.vocab_tgt = Vocab.load(vocab_tgt_path,
                                    train_path=trn_tgt_path,
                                    lowercase=lowercase)

        # load datasets
        self.data_sets = {
            'trn':
            SingleDataset(trn_src_path,
                          trn_tgt_path,
                          vocab_src=self.vocab_src,
                          vocab_tgt=self.vocab_tgt,
                          attach_ends=attach_ends),
            'val':
            SingleDataset(val_src_path,
                          val_tgt_path,
                          vocab_src=self.vocab_src,
                          vocab_tgt=self.vocab_tgt,
                          attach_ends=attach_ends),
            'tst':
            SingleDataset(tst_src_path,
                          tst_tgt_path,
                          vocab_src=self.vocab_src,
                          vocab_tgt=self.vocab_tgt,
                          attach_ends=attach_ends),
        }

    def __getitem__(self, key):
        return self.data_sets[key]


class SingleDataset(Dataset):
    def __init__(self,
                 path_src,
                 path_tgt,
                 vocab_src,
                 vocab_tgt,
                 attach_ends=True):
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.attach_ends = attach_ends
        sents_src = []
        for sent in load_sents(path_src):
            sent = vocab_src.encode_sent(sent, attach_ends=attach_ends)

            sents_src.append(sent)
        sents_tgt = []
        for sent in load_sents(path_tgt):
            sent = vocab_tgt.encode_sent(sent, attach_ends=attach_ends)

            sents_tgt.append(sent)

        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        assert len(self.sents_src) == len(self.sents_tgt)

    def __getitem__(self, index):
        return self.sents_src[index], self.sents_tgt[index]

    def __len__(self):
        return len(self.sents_src)


class Vocab(object):
    PAD_ID = 0
    START_ID = 1
    END_ID = 2
    UNK_ID = 3

    @staticmethod
    def load(path, train_path=None, lowercase=False, min_freq=2):
        if not os.path.exists(path):
            assert train_path is not None
            print('build vocab from %s...' % train_path)
            vocab = Vocab.build_vocab(train_path,
                                      lowercase=lowercase,
                                      min_freq=min_freq)
            torch.save(vocab, path)

        vocab = torch.load(path)
        assert isinstance(vocab, Vocab)
        assert vocab.lowercase == lowercase
        return vocab

    @staticmethod
    def build_vocab(path, lowercase=False, min_freq=2):
        cnts = Counter()
        for sent in load_sents(path):
            if lowercase:
                sent = sent.lower()
            tks = sent.strip().split()
            cnts.update(tks)

        vocab = {
            '<pad>': Vocab.PAD_ID,
            '<s>': Vocab.START_ID,
            '</s>': Vocab.END_ID,
            '<unk>': Vocab.UNK_ID
        }

        for key, c in sorted(cnts.items(), reverse=True):
            if c >= min_freq and key not in vocab:
                vocab[key] = len(vocab)

        return Vocab(vocab, lowercase=lowercase)

    def __init__(self, vocab, lowercase):

        self.lowercase = lowercase
        self.vocab = vocab
        self.tks = {tid: key for key, tid in vocab.items()}

    def dump(self, path):
        torch.save(self, path)

    def get_token(self, tid):
        return self.tks[tid]

    def get_token_id(self, token):
        return self.vocab.get(token, Vocab.UNK_ID)

    def encode_sent(self, sent, attach_ends=True):
        if isinstance(sent, list):
            sent = ' '.join(sent)
        if self.lowercase:
            sent = sent.lower()
        tks = sent.strip().split()
        tids = [self.get_token_id(tk) for tk in tks]
        if attach_ends:
            tids = [Vocab.START_ID] + tids + [Vocab.END_ID]
        return tids

    def decode_ids(self, tids, dettach_ends=True):
        if dettach_ends:
            if len(tids) >= 1 and tids[0] == Vocab.START_ID:
                tids = tids[1:]
            if len(tids) >= 1 and tids[-1] == Vocab.END_ID:
                tids = tids[:-1]
        return [self.get_token[tid] for tid in tids]


def load_sents(path):
    with open(path) as infile:
        for line in infile:
            sent = ' '.join(line.strip().split())
            yield sent


def to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=True)
    if isinstance(tensor, Mapping):
        return {key: to_device(value, device) for key, value in tensor.items()}
    if isinstance(tensor, Iterable):
        return [to_device(x, device) for x in tensor]
    raise NotImplementedError()


def yield_to_device(generator, device):
    for x in generator:
        yield to_device(x, device)


def get_dataloader(dset, batch_size, shuffle=True, num_workers=2):
    vocab_src = dset.vocab_src
    vocab_tgt = dset.vocab_tgt
    assert vocab_src.PAD_ID == vocab_tgt.PAD_ID
    pad_id = vocab_src.PAD_ID

    def pad_seqs(batch):
        length = np.array([len(item) for item in batch])
        max_length = length.max()

        data = np.array([(item + [pad_id] * (max_length - len(item)))
                         for item in batch])
        data = torch.tensor(data, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        return data, length

    def my_collate(batch):
        src_seqs, src_lengths = pad_seqs([item[0] for item in batch])
        tgt_seqs, tgt_lengths = pad_seqs([item[1] for item in batch])
        return [src_seqs, src_lengths, tgt_seqs, tgt_lengths]

    dataloader = DataLoader(dset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=my_collate,
                            pin_memory=True)
    return dataloader
