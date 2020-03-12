import unittest
from argparse import Namespace

import torch

from ..model import transformer
from .. import data_set


class TransformerTest(unittest.TestCase):
    def setUp(self):
        self.data_splits = data_set.SplittedDataset('test_data/single',
                                                    lowercase=False)

        self.args = Namespace(dec_embed_dim=32,
                              dec_ffn_dim=32,
                              dec_num_heads=2,
                              dec_num_layers=2,
                              dec_layernorm_before=False,
                              enc_embed_dim=32,
                              enc_ffn_dim=32,
                              enc_num_heads=2,
                              enc_num_layers=2,
                              enc_layernorm_before=False,
                              dropout=0.,
                              act_dropout=0.,
                              attn_dropout=0.,
                              embed_dropout=0.)
        self.model = transformer.Transformer(
            self.args,
            src_dict=self.data_splits.vocab_src,
            tgt_dict=self.data_splits.vocab_tgt)

    def test_runnable(self):
        self.model.reset_parameters()
        dl = data_set.get_dataloader(self.data_splits['trn'], batch_size=10)
        for src_tokens, src_lengths, tgt_tokens, tgt_lengths in dl:
            self.model.forward(src_tokens, src_lengths, tgt_tokens,
                               tgt_lengths)


if __name__ == '__main__':
    unittest.main()
