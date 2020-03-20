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
        for sent in load_sents(path_src):
            sent = vocab_src.encode_sent(sent, attach_ends=False)
            sents_src.append(sent)
        sents_tgt = []
        for sent in load_sents(path_tgt):
            sent = vocab_tgt.encode_sent(sent, attach_ends=attach_ends)

            sents_tgt.append(sent)

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
    def __init__(self, sampler, batch_size, max_tokens=None, group_ratio=3):
        super().__init__(sampler, batch_size=batch_size, drop_last=False)
        self.max_tokens = max_tokens
        self.group_ratio = group_ratio

        max_length = min_length = None
        for idx in self.sampler:
            length = self.sampler.data_source[idx][0].size(0)
            if max_length is None or length > max_length:
                max_length = length
            if min_length is None or length < min_length:
                min_length = length
        max_num_pads = (max_length - min_length)
        self.num_buckets = int(max_num_pads // group_ratio) + 1
        self.max_length = max_length

        # compute number of batches
        if max_tokens is None or max_tokens <= 0:
            num_batches = len(self.sampler) // batch_size
            if len(self.sampler) % batch_size != 0:
                num_batches += 1
        else:
            num_batches = 0
            # count how many items are put into each bucket
            buckets = [0] * self.num_buckets
            for idx in self.sampler:
                length = self.sampler.data_source[idx][0].size(0)
                bk_idx = self._get_bk_idx(length)
                buckets[bk_idx] += 1

            for bk_idx in range(self.num_buckets):
                bk_len = self._get_bk_len(bk_idx)
                bk_batch_size = max_tokens // bk_len
                if bk_batch_size >= batch_size:
                    bk_batch_size = batch_size
                num_batches += buckets[bk_idx] // bk_batch_size
                buckets[bk_idx] %= bk_batch_size

            batch_len = 0
            batch = []
            for bk_idx, size in enumerate(buckets):
                bk_len = self._get_bk_len(bk_idx)
                while size > 0:
                    size -= 1
                    if self._exploded(batch_len, bk_len):
                        num_batches += 1
                        batch = []
                        batch_len = 0

                    batch.append(idx)
                    batch_len += bk_len

                    if len(batch) == self.batch_size:
                        num_batches += 1
                        batch = []
                        batch_len = 0
            if len(batch) > 0 and not self.drop_last:
                num_batches += 1

        self.num_batches = num_batches

    def _get_bk_len(self, bk_idx):
        return self.max_length - int(bk_idx * self.group_ratio)

    def _get_bk_idx(self, length):
        num_pads = self.max_length - length
        bk_idx = int(num_pads // self.group_ratio)
        return bk_idx

    def _exploded(self, batch_len, next_len):
        if batch_len == 0:
            return False
        if (self.max_tokens is not None and self.max_tokens > 0
                and batch_len + next_len > self.max_tokens):
            return True
        return False

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        buckets = [[] for _ in range(self.num_buckets)]
        bucket_lens = [0] * self.num_buckets

        # variables to check all indices are returned exactly once
        yielded = 0
        yielded_indices = []

        def reset_bucket(bk_idx):
            buckets[bk_idx] = []
            bucket_lens[bk_idx] = 0

        for idx in self.sampler:
            length = self.sampler.data_source[idx][0].size(0)
            bk_idx = self._get_bk_idx(length)
            bk_len = self._get_bk_len(bk_idx)
            if self._exploded(bucket_lens[bk_idx], bk_len):
                batch = buckets[bk_idx]
                yield batch
                yielded_indices.append(batch)
                yielded += 1
                reset_bucket(bk_idx)

            buckets[bk_idx].append(idx)
            bucket_lens[bk_idx] += bk_len
            if len(buckets[bk_idx]) == self.batch_size:
                batch = buckets[bk_idx]
                yield batch
                yielded_indices.append(batch)
                yielded += 1
                reset_bucket(bk_idx)

        batch = []
        batch_len = 0
        leftover = [(bk_idx, idx) for bk_idx, bucket in enumerate(buckets)
                    for idx in bucket]
        for bk_idx, idx in leftover:
            bk_len = self._get_bk_len(bk_idx)
            if self._exploded(batch_len, bk_len):
                yield batch
                yielded_indices.append(batch)
                yielded += 1
                batch = []
                batch_len = 0

            batch.append(idx)
            batch_len += bk_len

            if len(batch) == self.batch_size:
                yield batch
                yielded_indices.append(batch)
                yielded += 1
                batch = []
                batch_len = 0

        if len(batch) > 0 and not self.drop_last:
            yield batch
            yielded_indices.append(batch)
            yielded += 1

        assert len(
            self
        ) == yielded, 'produced an incorrect number of batches. %d != %d' % (
            len(self), yielded)
        yielded_idx = set()
        for indices in yielded_indices:
            yielded_idx.update(indices)
        assert len(self.sampler) == len(
            yielded_idx
        ), 'produced an incorrect number of instances. %d != %d' % (len(
            self.sampler), len(yielded_idx))


def get_dataloader(dset,
                   batch_size,
                   max_tokens=None,
                   shuffle=True,
                   sample_by_length=True,
                   num_workers=2,
                   pin_memory=True):
    vocab_src = dset.vocab_src
    vocab_tgt = dset.vocab_tgt
    assert vocab_src.PAD_ID == vocab_tgt.PAD_ID
    pad_id = vocab_src.PAD_ID

    def pad_tgt_seqs(batch):
        lengths = np.array([len(item) - 1 for item in batch])
        max_length = lengths.max()

        data = torch.full((len(batch), max_length),
                          fill_value=pad_id, dtype=torch.long)
        data2 = torch.full((len(batch), max_length),
                           fill_value=pad_id, dtype=torch.long)
        for i in range(len(batch)):
            data[i, :lengths[i]] = batch[i][:-1]
            data2[i, :lengths[i]] = batch[i][1:]
        lengths = torch.tensor(lengths, dtype=torch.long)
        return data, data2, lengths

    def pad_src_seqs(batch):
        lengths = np.array([len(item) for item in batch])
        max_length = lengths.max()
        data = torch.full((len(batch), max_length),
                          fill_value=pad_id, dtype=torch.long)
        for i in range(len(batch)):
            data[i, :lengths[i]] = batch[i]
        lengths = torch.tensor(lengths, dtype=torch.long)
        return data, lengths

    def my_collate(batch):
        src_seqs, src_lengths = pad_src_seqs([item[0] for item in batch])
        tgt_seqs, tgt_seqs2, tgt_lengths = pad_tgt_seqs(
            [item[1] for item in batch])
        return [src_seqs, src_lengths, tgt_seqs, tgt_seqs2, tgt_lengths]

    if sample_by_length and shuffle:
        ran_sampler = torch.utils.data.RandomSampler(dset)
        len_sampler = LenMatchBatchSampler(ran_sampler,
                                           batch_size=batch_size,
                                           max_tokens=max_tokens)
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
