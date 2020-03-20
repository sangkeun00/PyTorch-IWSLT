import os
from collections import Counter

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
        src_max_len = -1
        for sent in load_sents(path_src):
            sent = vocab_src.encode_sent(sent, attach_ends=False)
            src_max_len = max(src_max_len, len(sent))
            sents_src.append(sent)
        sents_tgt = []
        for sent in load_sents(path_tgt):
            sent = vocab_tgt.encode_sent(sent, attach_ends=attach_ends)

            sents_tgt.append(sent)

        for i in range(len(sents_src)):
            sents_src[i] = sents_src[i] + [0] * (src_max_len -
                                                 len(sents_src[i]))

        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        assert len(self.sents_src) == len(self.sents_tgt)

    def __getitem__(self, index):
        src = torch.tensor(self.sents_src[index], dtype=torch.long)
        tgt = torch.tensor(self.sents_tgt[index], dtype=torch.long)
        return src, tgt

    def __len__(self):
        return len(self.sents_src)


class Vocab(object):
    PAD_ID = 0
    START_ID = 1
    END_ID = 2
    UNK_ID = 3

    @staticmethod
    def load(path, train_path=None, lowercase=False, min_freq=1):
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
    def build_vocab(path, lowercase=False, min_freq=1):
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

        for key, c in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and key not in vocab:
                vocab[key] = len(vocab)

        return Vocab(vocab, lowercase=lowercase)

    def __init__(self, vocab, lowercase):

        self.lowercase = lowercase
        self.vocab = vocab
        self.tks = [None] * len(vocab)
        for key, tid in vocab.items():
            self.tks[tid] = key

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
        tids = [int(t) for t in tids]
        if dettach_ends:
            while len(tids) >= 1 and tids[0] == Vocab.START_ID:
                tids = tids[1:]
            while len(tids) >= 1 and tids[-1] in {Vocab.END_ID, Vocab.PAD_ID}:
                tids = tids[:-1]
        return [self.get_token(tid) for tid in tids]

    def __len__(self):
        return len(self.tks)


def load_sents(path):
    with open(path) as infile:
        for line in infile:
            sent = ' '.join(line.strip().split())
            yield sent


class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            num_pads = torch.sum(self.sampler.data_source[idx][0] == 0)
            num_pads = int(num_pads / 3)
            if len(buckets[num_pads]) == 0:
                buckets[num_pads] = []
            buckets[num_pads].append(idx)

            if len(buckets[num_pads]) == self.batch_size:
                batch = list(buckets[num_pads])
                yield batch
                yielded += 1
                buckets[num_pads] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an incorrect number of batches."


def get_dataloader(dset,
                   batch_size,
                   shuffle=True,
                   sample_by_length=True,
                   num_workers=2,
                   pin_memory=True):
    vocab_src = dset.vocab_src
    vocab_tgt = dset.vocab_tgt
    assert vocab_src.PAD_ID == vocab_tgt.PAD_ID
    pad_id = vocab_src.PAD_ID
    # assume pad_id = 0 in this implementation
    assert pad_id == 0

    def pad_seqs(batch):
        length = np.array([len(item) - 1 for item in batch])
        max_length = length.max()

        data = torch.zeros((len(batch), max_length), dtype=torch.long)
        data2 = torch.zeros((len(batch), max_length), dtype=torch.long)
        for i in range(len(batch)):
            data[i, :len(batch[i]) - 1] = batch[i][:-1]
            data2[i, :len(batch[i]) - 1] = batch[i][1:]
        length = torch.tensor(length, dtype=torch.long)
        return data, data2, length

    def pack_seqs(batch):
        lengths = np.array([torch.sum((b != pad_id)).item() for b in batch])
        max_len = lengths.max()
        data = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i in range(len(batch)):
            data[i] = batch[i][:max_len]
        lengths = torch.tensor(lengths, dtype=torch.long)
        return data, lengths

    def my_collate(batch):
        src_seqs, src_lengths = pack_seqs([item[0] for item in batch])
        tgt_seqs, tgt_seqs2, tgt_lengths = pad_seqs(
            [item[1] for item in batch])
        return [src_seqs, src_lengths, tgt_seqs, tgt_seqs2, tgt_lengths]

    if sample_by_length and shuffle:
        ran_sampler = torch.utils.data.RandomSampler(dset)
        len_sampler = LenMatchBatchSampler(ran_sampler,
                                           batch_size=batch_size,
                                           drop_last=False)
        dataloader = DataLoader(dset,
                                num_workers=num_workers,
                                collate_fn=my_collate,
                                batch_sampler=len_sampler,
                                pin_memory=pin_memory)
    else:
        dataloader = DataLoader(dset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                collate_fn=my_collate,
                                pin_memory=pin_memory)
    return dataloader
