import argparse

import torch
import pytorch_lightning as pl
from . import data_set
from . import models
from . import lr_scheduler
from . import losses


class Seq2SegModel(pl.LightningModule):
    def __init__(self, args, data_splits):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.data_splits = data_splits
        if args.transformer_impl == 'pytorch':
            self.model = models.easy_transformer.EasyTransformer(
                args,
                src_dict=data_splits.vocab_src,
                tgt_dict=data_splits.vocab_tgt)
        elif args.transformer_impl == 'custom':
            self.model = models.transformer.Transformer(
                args,
                src_dict=data_splits.vocab_src,
                tgt_dict=data_splits.vocab_tgt)

    def forward(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        return self.model(src_tokens, src_lengths, tgt_tokens, tgt_lengths)

    def training_step(self, batch, batch_nb):
        src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths = batch
        logits = self.forward(src_tokens, src_lengths, tgt_inputs, tgt_lengths)
        loss = losses.masked_nll(logits, tgt_lengths, tgt_outputs)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        src_tokens, src_lengths, tgt_inputs, tgt_outputs, tgt_lengths = batch
        logits = self.forward(src_tokens, src_lengths, tgt_inputs, tgt_lengths)
        loss = losses.masked_nll(logits, tgt_lengths, tgt_outputs)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_losses = [item['val_loss'] for item in outputs]
        return {'val_loss': torch.stack(val_losses).mean()}

    def configure_optimizers(self):
        if self.args.optim == 'adam':
            opt = torch.optim.Adam(self.model.parameters(),
                                   lr=self.learning_rate,
                                   betas=(0.9, 0.997),
                                   weight_decay=self.weight_decay)
        elif self.args.optim == 'adamw':
            opt = torch.optim.AdamW(self.model.parameters(),
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.997),
                                    weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()
        if self.args.decay_method == 'inverse_sqrt':
            scheduler = lr_scheduler.InverseSqrtScheduler(
                opt,
                warmup_steps=self.args.warmup_steps,
                min_lr=self.args.min_lr)
            sch = {'scheduler': scheduler, 'interval': 'step'}
        else:
            raise NotImplementedError()
        return [opt], [sch]

    def train_dataloader(self):
        return data_set.get_dataloader(self.data_splits['trn'],
                                       batch_size=self.args.batch_size)

    def val_dataloader(self):
        return data_set.get_dataloader(self.data_splits['val'],
                                       batch_size=self.args.batch_size,
                                       shuffle=False)

    def test_dataloader(self):
        return data_set.get_dataloader(self.data_splits['tst'],
                                       batch_size=self.args.batch_size,
                                       shuffle=False)


def main():
    args = parse_args()

    # initialize dataset
    data_splits = data_set.SplittedDataset(args.data_dir,
                                           lang_src=args.lang_src,
                                           lang_tgt=args.lang_tgt,
                                           lowercase=args.lowercase)

    # initialize model
    model = Seq2SegModel(args, data_splits=data_splits)

    trainer = pl.Trainer(precision=16 if args.fp16 else 32,
                         max_epochs=args.max_epochs,
                         gpus=args.gpus,
                         amp_level='O1',
                         accumulate_grad_batches=2)
    trainer.fit(model)


def parse_args():
    parser = argparse.ArgumentParser()
    # environment parameters
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--gpus', type=int, nargs='+')

    # data parameters
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--data-dir', default='data/iwslt-2014')
    parser.add_argument('--lang-src', default='en')
    parser.add_argument('--lang-tgt', default='de')

    # training parameters
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--learning-rate',
                        type=float,
                        default=5e-4,
                        help='learning rate')
    parser.add_argument('--optim', choices=('adam', 'adamw'), default='adamw')
    parser.add_argument('--decay-method',
                        choices=('inverse_sqrt', 'cos'),
                        default='inverse_sqrt')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--min-lr', type=float, default=1e-9)
    parser.add_argument('--warmup-steps', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=80)
    parser.add_argument('--label-smoothing', type=float, default=0.)

    # model parameters
    parser.add_argument('--transformer-impl',
                        choices=('custom', 'pytorch'),
                        default='custom')
    parser.add_argument('--dec-embed-dim', default=512)
    parser.add_argument('--dec-ffn-dim', default=1024)
    parser.add_argument('--dec-num-heads', default=4)
    parser.add_argument('--dec-num-layers', default=6)
    parser.add_argument('--dec-layernorm-before', action='store_true')
    parser.add_argument('--enc-embed-dim', default=512)
    parser.add_argument('--enc-ffn-dim', default=1024)
    parser.add_argument('--enc-num-heads', default=4)
    parser.add_argument('--enc-num-layers', default=6)
    parser.add_argument('--enc-layernorm-before', action='store_true')

    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--act-dropout', default=0.0)
    parser.add_argument('--attn-dropout', default=0.0)
    parser.add_argument('--embed-dropout', default=0.3)

    return parser.parse_args()


if __name__ == '__main__':
    main()
