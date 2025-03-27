import random
from typing import Optional

import dgl
import torch
from dgl.nn.pytorch import GraphConv
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score

from src.utils.graph_utils import init_node_feature


class GCN4Graph(LightningModule):
    def __init__(
        self,
        gcn_layer: torch.nn.Module,
        score_layer: torch.nn.Module,
        feat_dim: int,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["encoder", "decoder"])

        self.gcn_layer = gcn_layer
        self.score_layer = score_layer

        self.initial_node_conv = GraphConv(feat_dim, feat_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_loss = MeanMetric()
        self.F1 = BinaryF1Score()

    def training_step(self, batch, idx) -> STEP_OUTPUT:
        g, label, mask = batch
        init_node_feat = init_node_feature(
            g, self.hparams.feat_dim, self.initial_node_conv, self.device
        )

        node_feat = self.gcn_layer(g, init_node_feat)

        loss = torch.zeros(1).to(self.device)
        for each_init_g, each_label, each_mask, node_feat_for_each_graph in zip(
            dgl.unbatch(g),
            label,
            mask,
            torch.split(node_feat, g.batch_num_nodes().tolist(), dim=0),
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
            # score = self.score_layer(node_feat_for_each_graph)
            score = self.score_layer(
                torch.cat(
                    (
                        torch.repeat_interleave(
                            node_feat_for_each_graph,
                            repeats=node_feat_for_each_graph.size(0),
                            dim=0,
                        ),
                        node_feat_for_each_graph.repeat(
                            node_feat_for_each_graph.size(0), 1
                        ),
                    ),
                    dim=-1
                )
            ).view(-1)

            pred = score[positive_indices + negative_indices]
            target = (
                torch.tensor(
                    [1 for _ in range(len(positive_indices))]
                    + [0 for _ in range(len(negative_indices))],
                )
                .to(self.device)
                .float()
            )
            loss += self.criterion(pred, target)

        loss /= g.batch_size

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False)
        self.log("train/epoch_loss", self.train_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        preds, targets = self.val_test_step(batch)

        self.F1(preds, targets)
        self.log("val/F1", self.F1, on_step=False, on_epoch=True)

        return {"preds": preds, "targets": targets}

    def test_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        preds, targets = self.val_test_step(batch)

        return {"preds": preds, "targets": targets}

    def val_test_step(self, batch):
        g, label, mask = batch
        init_node_feat = init_node_feature(
            g, self.hparams.feat_dim, self.initial_node_conv, self.device
        )

        node_feat = self.gcn_layer(g, init_node_feat)

        targets = []
        preds = []
        for each_init_g, each_label, each_mask, node_feat_for_each_graph in zip(
            dgl.unbatch(g),
            label,
            mask,
            torch.split(node_feat, g.batch_num_nodes().tolist(), dim=0),
        ):
            num_nodes = each_init_g.num_nodes()

            positive_indices = [
                (i * num_nodes + j).item()
                for i, j in torch.masked_select(each_label, each_mask.bool())
                .view(2, -1)
                .unbind(-1)
            ]
            target = torch.zeros(num_nodes ** 2).to(self.device).float()
            target[positive_indices] = 1
            targets.append(target)

            preds.append(self.score_layer(
                torch.cat(
                    (
                        torch.repeat_interleave(
                            node_feat_for_each_graph,
                            repeats=node_feat_for_each_graph.size(0),
                            dim=0,
                        ),
                        node_feat_for_each_graph.repeat(
                            node_feat_for_each_graph.size(0), 1
                        ),
                    ),
                    dim=-1
                )
            ).view(-1))

        targets = torch.cat(targets, dim=-1)
        preds = torch.cat(preds, dim=-1)
        preds = self.sigmoid(preds)

        return preds, targets

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}


# class GCNLayer(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super().__init__()
#
#         self.conv1 = GraphConv(input_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, hidden_dim)
#         self.conv3 = GraphConv(hidden_dim, latent_dim)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
#         g = dgl.add_self_loop(g)
#         feat = self.relu(self.conv1(g, feat))
#         feat = self.relu(self.conv2(g, feat))
#         feat = self.conv3(g, feat)
#         return feat


# class DotProductLayer(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.src_trans = torch.nn.Linear(input_dim, hidden_dim)
#         self.dst_trans = torch.nn.Linear(input_dim, hidden_dim)
#
#     def forward(self, feat: torch.Tensor):
#         src_feat = self.src_trans(feat)
#         dst_feat = self.dst_trans(feat)
#         adj_matrix = torch.mm(src_feat, dst_feat.T)
#         return adj_matrix.view(-1)
