import random
from typing import Optional

import dgl
import torch
from dgl.nn.pytorch import GraphConv
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import Linear
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score

from src.models.components.nri_mp import NRIMessagePassing
from src.utils.graph_utils import complete_graph, init_node_feature


class NRIEncoder4Graph(LightningModule):
    def __init__(
        self,
        # net: nn.Module,
        feat_dim: int,
        message_passing: NRIMessagePassing,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.message_passing = message_passing

        self.initial_node_conv = GraphConv(feat_dim, feat_dim)
        self.score_trans = torch.nn.Sequential(
            Linear(feat_dim, int(feat_dim / 2)),
            torch.nn.ReLU(),
            Linear(int(feat_dim / 2), 1)
        )
        self.sigmoid = torch.nn.Sigmoid()

        self.criterion = nn.BCEWithLogitsLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # metrics for early stopping
        self.F1 = BinaryF1Score()

    def step(self, g: dgl.DGLGraph):
        init_node_feat: torch.Tensor = init_node_feature(g, self.hparams.feat_dim, self.initial_node_conv, self.device)
        init_graphs = [complete_graph(n).to(self.device) for n in g.batch_num_nodes()]
        init_g = dgl.batch(init_graphs)

        edge_feat = self.message_passing(init_g, init_node_feat)

        return init_g, edge_feat

    def training_step(self, batch, idx):
        g, label, mask = batch

        init_g, edge_feat = self.step(g)

        loss = torch.zeros(1).to(self.device)
        for each_init_g, each_label, each_mask, each_edge_feat in zip(
            dgl.unbatch(init_g),
            label,
            mask,
            torch.split(edge_feat, init_g.batch_num_edges().tolist(), dim=0),
        ):
            num_nodes = each_init_g.num_nodes()

            # selected negative samples
            positive_indices = [
                (i * num_nodes + j).item()
                for i, j in torch.masked_select(each_label, each_mask.bool())
                .view(2, -1)
                .unbind(-1)
            ]
            negative_indices = list(
                set(range(num_nodes * num_nodes)) - set(positive_indices)
            )
            negative_indices = random.choices(negative_indices, k=len(positive_indices))
            selected_sample_feat = each_edge_feat[positive_indices + negative_indices]

            pred = self.score_trans(selected_sample_feat).view(-1)
            target = (
                torch.tensor(
                    [1 for _ in range(len(positive_indices))]
                    + [0 for _ in range(len(negative_indices))],
                )
                .to(self.device)
                .float()
            )

            loss += self.criterion(pred, target)

        loss /= init_g.batch_size

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False)
        self.log("train/epoch_loss", self.train_loss, on_step=False, on_epoch=True)

        return loss

    def val_test_step(self, batch):
        g, label, mask = batch
        init_g, edge_feat = self.step(g)

        preds = self.score_trans(edge_feat).view(-1)
        preds = self.sigmoid(preds)

        targets = []
        for each_init_g, each_label, each_mask in zip(
            dgl.unbatch(init_g),
            label,
            mask,
        ):
            num_nodes = each_init_g.num_nodes()
            positive_indices = [
                (i * num_nodes + j).item()
                for i, j in torch.masked_select(each_label, each_mask.bool())
                .view(2, -1)
                .unbind(-1)
            ]
            target = torch.zeros(each_init_g.num_edges()).to(self.device).float()
            target[positive_indices] = 1
            targets.append(target)
        targets = torch.cat(targets, dim=-1)

        return preds, targets

    def validation_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        preds, targets = self.val_test_step(batch)

        self.F1(preds, targets)
        self.log("val/F1", self.F1, on_step=False, on_epoch=True)

        return {"preds": preds, "targets": targets}

    def test_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        preds, targets = self.val_test_step(batch)

        return {"preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}
