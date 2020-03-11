import argparse

import torch
import pytorch_lightning as pl
import data_set
import model as models


class Seq2SegModel(pl.LightningModule):
    def __init__(self, args, data_splits):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate
        self.data_splits = data_splits
        self.encoder = models.transformer.TransformerEncoder(
            enc_embed_dim=args.enc_embed_dim,
            enc_ffn_dim=args.enc_ffn_dim,
            enc_num_heads=args.enc_num_heads,
            enc_num_layers=args.enc_num_layers,
            src_dict=data_splits.vocab_src,
            enc_layernorm_before=args.enc_layernorm_before,
            attn_dropout=args.attn_dropout,
            act_dropout=args.act_dropout,
            embed_dropout=args.embed_dropout,
            dropout=args.dropout)
        self.decoder = models.transformer.TransformerDecoder(
            dec_embed_dim=args.dec_embed_dim,
            dec_ffn_dim=args.dec_ffn_dim,
            dec_num_heads=args.dec_num_heads,
            dec_num_layers=args.dec_num_layers,
            tgt_dict=data_splits.vocab_tgt,
            dec_layernorm_before=args.enc_layernorm_before,
            attn_dropout=args.attn_dropout,
            act_dropout=args.act_dropout,
            embed_dropout=args.embed_dropout,
            dropout=args.dropout)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        pass

    def configure_optimizers(self):
        # TODO: just test! should move enc & dec to the same net!
        opt = torch.optim.Adam(self.encoder.parameters(),
                               lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return data_set.get_dataloader(self.data_splits['trn'],
                                       batch_size=self.args.batch_size)

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
                        choices=('poly', 'cos'),
                        default='poly')
    parser.add_argument('--min-lr', type=float, default=0.004)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=60)

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
