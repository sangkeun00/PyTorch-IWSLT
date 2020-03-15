import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

from ..optim.lr_scheduler import InverseSqrtScheduler
from ..model import transformer
from ..model import easy_transformer
from ..model import utils
from .. import data_set


class TransformerTest(unittest.TestCase):
    def setUp(self):
        self.data_splits = data_set.SplittedDataset('test_data/single',
                                                    lowercase=False)

        self.args = Namespace(dec_embed_dim=64,
                              dec_ffn_dim=64,
                              dec_num_heads=4,
                              dec_num_layers=2,
                              dec_layernorm_before=False,
                              enc_embed_dim=64,
                              enc_ffn_dim=64,
                              enc_num_heads=4,
                              enc_num_layers=2,
                              enc_layernorm_before=False,
                              dropout=0.,
                              act_dropout=0.,
                              attn_dropout=0.,
                              embed_dropout=0.,
                              dec_tied_weight=False)
        self.model = transformer.Transformer(
            self.args,
            src_dict=self.data_splits.vocab_src,
            tgt_dict=self.data_splits.vocab_tgt)
        self.easy_model = easy_transformer.EasyTransformer(
            self.args,
            src_dict=self.data_splits.vocab_src,
            tgt_dict=self.data_splits.vocab_tgt)

    def runnable_check(self, model):
        model.reset_parameters()
        model.train()
        dl = data_set.get_dataloader(self.data_splits['trn'], batch_size=10)
        for src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths in dl:
            model.forward(src_tokens, src_lengths, tgt_inputs, tgt_lengths)

    def converge_check(self, model, epochs=15):
        model.reset_parameters()
        model.train()
        dl = data_set.get_dataloader(self.data_splits['trn'], batch_size=10)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(epochs):
            for src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths in dl:
                opt.zero_grad()
                outputs = model.forward(src_tokens, src_lengths, tgt_inputs,
                                        tgt_lengths)
                nll = utils.masked_nll(outputs, tgt_lengths, tgt_outputs)
                nll.backward()
                opt.step()
        final_nll = nll.cpu().item()
        self.assertLessEqual(final_nll, 0.15,
                             'loss must converge for simple data')

    def test_runnable(self):
        self.runnable_check(self.model)

    def test_can_converge(self):
        self.converge_check(self.model)

    def test_decoded(self):
        # do not shuffle, alway use the same 4 instances
        # TODO: It seems the 5th instance is quite difficult
        # for the model (?), unsure if there is problem
        dl = data_set.get_dataloader(self.data_splits['trn'],
                                     batch_size=4,
                                     shuffle=False)
        for src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths in dl:
            # get one batch
            break

        bsz, length = tgt_outputs.size()

        model = self.model
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        sch = InverseSqrtScheduler(opt, warmup_steps=500, min_lr=1e-9)
        # we try at most 100 epochs before failure
        for _ in range(100):
            model.train()
            for _ in range(50):
                opt.zero_grad()
                outputs = model.forward(src_tokens, src_lengths, tgt_inputs,
                                        tgt_lengths)
                nll = utils.masked_nll(outputs, tgt_lengths, tgt_outputs)
                nll.backward()
                opt.step()
                sch.step()

            model.eval()
            outputs = model.greedy_decode(src_tokens, max_length=length + 10)

            if outputs.size(1) != tgt_outputs.size(1):
                continue

            if not (outputs == tgt_outputs).all():
                continue

            # check success!
            break

        self.assertEqual(outputs.size(1), tgt_outputs.size(1),
                         'output_length must equal')
        for i in range(bsz):
            for j in range(length):
                self.assertEqual(
                    outputs[i][j].item(), tgt_outputs[i][j].item(),
                    '(%d, %d) equal %d = %d' %
                    (i, j, outputs[i][j].item(), tgt_outputs[i][j].item()))

    def test_easy_runnable(self):
        self.runnable_check(self.easy_model)

    def test_easy_can_converge(self):
        self.converge_check(self.easy_model)


if __name__ == '__main__':
    unittest.main()
