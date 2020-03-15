import argparse
import time

import torch
import pytorch_lightning as pl
import data_set
import model as models
import optim as optim


class Trainer(object):
    def __init__(self, args, data_splits, device):

        self.args = args
        self.data_splits = data_splits
        self.device = device

        if args.transformer_impl == 'pytorch':
            self.model = models.easy_transformer.EasyTransformer(
                args=args,
                src_dict=data_splits.vocab_src,
                tgt_dict=data_splits.vocab_tgt,
            ).to(device)
        elif args.transformer_impl == 'custom':
            self.model = models.transformer.Transformer(
                args=args,
                src_dict=data_splits.vocab_src,
                tgt_dict=data_splits.vocab_tgt,
            ).to(device)

        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.997),
                weight_decay=args.weight_decay
            )
        elif args.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.997),
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError

        self.scheduler = None
        if args.decay_method == 'inverse_sqrt':
            self.scheduler = optim.lr_scheduler.InverseSqrtScheduler(
                self.optimizer,
                warmup_steps=args.warmup_steps,
                min_lr=args.min_lr
            )
        else:
            raise ValueError

        self.train_loader = data_set.get_dataloader(
            dset=data_splits['trn'],
            batch_size=args.batch_size
        )
        self.val_loader = data_set.get_dataloader(
            dset=data_splits['val'],
            batch_size=args.batch_size
        )
        self.test_loader = data_set.get_dataloader(
            dset=data_splits['tst'],
            batch_size=args.batch_size
        )

    def train(self):
        for epoch in range(self.args.max_epochs):
            cum_loss = 0
            cum_tokens = 0
            self.optimizer.zero_grad()
            print("[Epoch {} (Train)]".format(epoch))
            for idx, batch in enumerate(self.train_loader):
                # Data loading
                src = batch[0].to(self.device)
                src_lens = batch[1].to(self.device)
                tgt_in = batch[2].to(self.device)
                tgt_out = batch[3].to(self.device)
                tgt_lens = batch[4].to(self.device)

                # Loss calculation
                logits = self.model(src, src_lens, tgt_in, tgt_lens)
                loss = models.utils.masked_nll(
                    logits=logits,
                    lengths=tgt_lens,
                    targets=tgt_out,
                    label_smoothing=self.args.label_smoothing,
                )

                # Optimizer update
                loss.backward()
                if (idx + 1) % self.args.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                # Logging
                cur_loss = loss.item()
                cur_tokens = torch.sum(tgt_lens).cpu().item()
                cum_loss += cur_loss * cur_tokens
                cum_tokens += cur_tokens
                avg_loss = cum_loss / cum_tokens
                avg_ppl = 2 ** avg_loss
                print(("\r[Step {}/{}] Batch Loss: {:.4f}, "
                       "Avg Loss: {:.4f}, Avg Perplexity: {:.4f}").format(
                          idx,
                          len(self.train_loader),
                          cur_loss,
                          avg_loss,
                          avg_ppl),
                      end="")

            print("\n[Epoch {} (Validation)]".format(epoch))
            val_loss = self.validation()
            print("Validation Loss: {:.4f}".format(val_loss))

    def validation(self):
        cum_loss = 0
        cum_tokens = 0
        for batch in self.val_loader:
            # Data loading
            src = batch[0].to(self.device)
            src_lens = batch[1].to(self.device)
            tgt_in = batch[2].to(self.device)
            tgt_out = batch[3].to(self.device)
            tgt_lens = batch[4].to(self.device)

            # Loss calculation
            with torch.no_grad():
                logits = self.model(src, src_lens, tgt_in, tgt_lens)
                loss = models.utils.masked_nll(logits, tgt_lens, tgt_out)

            # Logging
            cur_loss = loss.item()
            cur_tokens = torch.sum(tgt_lens).cpu().item()
            cum_loss += cur_loss * cur_tokens
            cum_tokens += cur_tokens

        return cum_loss / cum_tokens

    def test(self):
        pass


def main():
    args = parse_args()
    print(str(args))

    # initialize dataset
    data_splits = data_set.SplittedDataset(args.data_dir,
                                           lang_src=args.lang_src,
                                           lang_tgt=args.lang_tgt,
                                           lowercase=args.lowercase)

    cuda_device = "cuda:{}".format(args.gpu)
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    # initialize trainer
    trainer = Trainer(args, data_splits, device)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    # environment parameters
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--gpu', type=int, default=0)

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
    parser.add_argument('--gradient-accumulation', type=int, default=2)

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
    parser.add_argument('--dec-tied-weight', type=bool, default=True)

    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--act-dropout', default=0.0)
    parser.add_argument('--attn-dropout', default=0.0)
    parser.add_argument('--embed-dropout', default=0.3)

    return parser.parse_args()


if __name__ == '__main__':
    main()
