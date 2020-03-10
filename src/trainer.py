import argparse

import pytorch_lightning as pl
import data_set
import model as models


class Seq2SegModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass


def main():
    args = parse_args()
    model = Seq2SegModel()

    trainer = pl.Trainer(
        #
    )
    trainer.fit(model)


def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


if __name__ == '__main__':
    main()
