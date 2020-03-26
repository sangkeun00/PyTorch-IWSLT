import unittest
from argparse import Namespace

import torch

from ..optim.lr_scheduler import InverseSqrtScheduler
from ..models import transformer
from .. import losses
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

    def runnable_check(self, model):
        model.reset_parameters()
        model.train()
        dl = data_set.get_dataloader(self.data_splits['trn'],
                                     batch_size=10,
                                     pin_memory=False)
        for src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths in dl:
            model.forward(src_tokens, src_lengths, tgt_inputs, tgt_lengths)

    def converge_check(self, model, epochs=15):
        model.reset_parameters()
        model.train()
        dl = data_set.get_dataloader(self.data_splits['trn'],
                                     batch_size=10,
                                     pin_memory=False)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(epochs):
            for src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths in dl:
                opt.zero_grad()
                outputs = model.forward(src_tokens, src_lengths, tgt_inputs,
                                        tgt_lengths)
                nll, _ = losses.masked_nll(outputs, tgt_lengths, tgt_outputs)
                nll.backward()
                opt.step()
        final_nll = nll.cpu().item()
        self.assertLessEqual(final_nll, 0.15,
                             'loss must converge for simple data')

    def test_runnable(self):
        self.runnable_check(self.model)

    def test_can_converge(self):
        self.converge_check(self.model)

    def test_greedy_decoded(self):
        # do not shuffle, alway use the same 5 instances
        dl = data_set.get_dataloader(self.data_splits['trn'],
                                     pin_memory=False,
                                     batch_size=5,
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
                nll, _ = losses.masked_nll(outputs, tgt_lengths, tgt_outputs)
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

    def test_beam_decoded(self):
        # do not shuffle, alway use the same 5 instances
        dl = data_set.get_dataloader(self.data_splits['trn'],
                                     batch_size=5,
                                     shuffle=False,
                                     pin_memory=False)
        for src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths in dl:
            # get one batch
            break

        bsz, length = tgt_outputs.size()

        for _ in range(1):
            model = self.model
            model.reset_parameters()
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
            sch = InverseSqrtScheduler(opt, warmup_steps=500, min_lr=1e-9)
            model.train()
            for _ in range(500):
                opt.zero_grad()
                outputs = model.forward(src_tokens, src_lengths, tgt_inputs,
                                        tgt_lengths)
                nll, _ = losses.masked_nll(outputs, tgt_lengths, tgt_outputs)
                nll.backward()
                opt.step()
                sch.step()

            model.eval()
            outputs_1 = model.beam_decode(src_tokens,
                                        max_length=length + 10,
                                        length_normalize=False,
                                        beam_size=3)
            outputs_2 = model.beam_decode(src_tokens,
                                        max_length=length + 10,
                                        length_normalize=True,
                                        beam_size=3)

            if outputs_1.size(1) != tgt_outputs.size(1):
                continue
            if outputs_2.size(1) != tgt_outputs.size(1):
                continue

            if not (outputs_1 == tgt_outputs).all():
                continue
            if not (outputs_2 == tgt_outputs).all():
                continue

            # check success!
            break

        self.assertEqual(outputs_1.size(1), tgt_outputs.size(1),
                         'output_length must equal')
        self.assertEqual(outputs_2.size(1), tgt_outputs.size(1),
                         'output_length must equal')
        for outputs in (outputs_1, outputs_2):
            for i in range(bsz):
                for j in range(length):
                    self.assertEqual(
                        outputs[i][j].item(), tgt_outputs[i][j].item(),
                        '(%d, %d) equal %d = %d' %
                        (i, j, outputs[i][j].item(), tgt_outputs[i][j].item()))


if __name__ == '__main__':
    unittest.main()
