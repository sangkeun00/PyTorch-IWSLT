import argparse

import torch
import pytorch_lightning as pl
import data_set
import model as models
import lr_scheduler


class Seq2SegModel(pl.LightningModule):
    def __init__(self, args, data_splits):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate
        self.data_splits = data_splits
        self.model = models.transformer.Transformer(
            args,
            src_dict=data_splits.vocab_src,
            tgt_dict=data_splits.vocab_tgt)

    def forward(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        return self.model(src_tokens, src_lengths, tgt_tokens, tgt_lengths)

    def training_step(self, batch, batch_nb):
        src_tokens, src_lengths, tgt_tokens, tgt_lengths = batch
        logits = self.forward(src_tokens, src_lengths, tgt_tokens, tgt_lengths)
        loss = models.utils.masked_nll(logits[:, :-1, :], tgt_lengths - 1,
                                       tgt_tokens[:, 1:])
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        src_tokens, src_lengths, tgt_tokens, tgt_lengths = batch
        logits = self.forward(src_tokens, src_lengths, tgt_tokens, tgt_lengths)
        loss = models.utils.masked_nll(logits[:, :-1, :], tgt_lengths - 1,
                                       tgt_tokens[:, 1:])
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_losses = [item['val_loss'] for item in outputs]
        return {'val_loss': torch.stack(val_losses).mean()}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.args.decay_method == 'inverse_sqrt':
            sch = lr_scheduler.InverseSqrtScheduler(
                opt,
                warmup_steps=self.args.warmup_steps,
                min_lr=self.args.min_lr)
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
                                           lowercase=args.lowercase)

    # initialize model
    model = Seq2SegModel(args, data_splits=data_splits)

    trainer = pl.Trainer(
        precision=16 if args.fp16 else 32,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
    )
    trainer.fit(model)


def parse_args():
    parser = argparse.ArgumentParser()
    # environment parameters
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--gpus', type=int, nargs='+')

    # data parameters
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--data-dir', default='data/iwslt-2014')

    # training parameters
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--decay-method',
                        choices=('inverse_sqrt', 'cos'),
                        default='inverse_sqrt')
    parser.add_argument('--min-lr', type=float, default=0.004)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=55)
    parser.add_argument('--label-smoothing', type=float, default=0.0)

    # model parameters
    parser.add_argument('--dec-embed-dim', default=6)
    parser.add_argument('--dec-ffn-dim', default=6)
    parser.add_argument('--dec-num-heads', default=6)
    parser.add_argument('--dec-num-layers', default=6)
    parser.add_argument('--dec-layernorm-before', action='store_true')
    parser.add_argument('--enc-embed-dim', default=6)
    parser.add_argument('--enc-ffn-dim', default=6)
    parser.add_argument('--enc-num-heads', default=6)
    parser.add_argument('--enc-num-layers', default=6)
    parser.add_argument('--enc-layernorm-before', action='store_true')

    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--act-dropout', default=0.1)
    parser.add_argument('--attn-dropout', default=0.1)
    parser.add_argument('--embed-dropout', default=0.1)

    return parser.parse_args()


if __name__ == '__main__':
    main()
