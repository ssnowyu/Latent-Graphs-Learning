import os.path
from typing import Tuple, Optional

import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split

from src.data.components.graph_dataset import GraphDataset


class GraphDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        name_for_observed_data: str,
        name_for_graph_data: str,
        raw_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        processed_data_name: Optional[str] = None,
        force_reload: bool = False,
        verbose: bool = False,
        num_time_step: Optional[int] = None,
        without_node_feature: bool = False,
        without_edge_feature: bool = False,
    ):
        # LightningDataModule.__init__(self)

        super().__init__()
        # self.data_dir = data_dir
        # self.train_val_test_split = train_val_test_split
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.pin_memory = pin_memory
        self.save_hyperparameters()

        self.data_train: Optional[DGLDataset] = None
        self.data_val: Optional[DGLDataset] = None
        self.data_test: Optional[DGLDataset] = None

    def setup(self, stage: str) -> None:
        dataset = GraphDataset(
            name_for_observed_data=self.hparams.name_for_observed_data,
            name_for_graph_data=self.hparams.name_for_graph_data,
            raw_dir=os.path.join(self.hparams.data_dir, self.hparams.raw_dir),
            save_dir=os.path.join(self.hparams.data_dir, self.hparams.save_dir),
            force_reload=self.hparams.force_reload,
            verbose=self.hparams.verbose,
            num_time_step=self.hparams.num_time_step,
            without_node_feature=self.hparams.without_node_feature,
            without_edge_feature=self.hparams.without_edge_feature,
            processed_data_name=self.hparams.processed_data_name
        )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.manual_seed(12345),
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

if __name__ == '__main__':
    data_module = GraphDataModule(
        data_dir='../../data/q1',
        train_val_test_split=(7, 2, 1),
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        name_for_observed_data='2023-06-22-14-13-18-snapshots.csv',
        name_for_graph_data='2023-06-22-14-13-18-graph.csv',
        raw_dir='raw',
        save_dir='processed',
        num_time_step=5,
        without_node_feature=False,
        without_edge_feature=False,
    )
    print('complete')