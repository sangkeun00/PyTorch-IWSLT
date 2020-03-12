import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

from ..model import transformer
from ..model import utils
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

    def test_can_converge(self):
        self.model.reset_parameters()
        dl = data_set.get_dataloader(self.data_splits['trn'], batch_size=10)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(15):
            for src_tokens, src_lengths, tgt_tokens, tgt_lengths in dl:
                opt.zero_grad()
                outputs = self.model.forward(src_tokens, src_lengths,
                                             tgt_tokens, tgt_lengths)
                nll = utils.masked_nll(outputs[:, :-1, :], tgt_lengths - 1,
                                       tgt_tokens[:, 1:])
                nll.backward()
                opt.step()
        final_nll = nll.cpu().item()
        self.assertLessEqual(final_nll, 0.1,
                             'loss must converge for simple data')


if __name__ == '__main__':
    unittest.main()
