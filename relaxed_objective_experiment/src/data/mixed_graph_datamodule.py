from typing import Tuple
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning import LightningDataModule
from torch.utils.data import random_split
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader


class MixedGraphDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset: DGLDataset,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["dataset"])

        self.dataset = dataset

    def setup(self, stage: str) -> None:
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.dataset,
            lengths=self.hparams.train_val_test_split,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return GraphDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return GraphDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return GraphDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
