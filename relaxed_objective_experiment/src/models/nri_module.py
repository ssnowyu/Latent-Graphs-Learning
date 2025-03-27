from typing import Any, Dict, Optional
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from src.utils.graph_utils import complete_graph
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
import dgl
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F


class NRIModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        mapper: torch.nn.Module,
        num_mixture: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        learnable_feat: bool,
        num_features: Optional[int] = None,
        dim_feature: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.mapper = mapper

        self.save_hyperparameters(logger=False, ignore=["encoder", "decoder", "mapper"])

        # metric objects for calculating and averaging accuracy across batches
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()

        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_f1_best = MaxMetric()

        self.criterion = BCEWithLogitsLoss()

        self.num_mixture = num_mixture

        self.sigmoid = torch.nn.Sigmoid()

        self.bn = torch.nn.BatchNorm1d(1)

        if learnable_feat:
            if num_features is not None and dim_feature is not None:
                self.feat = torch.nn.Embedding(num_features, dim_feature)
            else:
                raise ValueError(
                    "num_features and dim_feature are expected if learnable_feat is True"
                )

    def train_step(self, batch_graph: dgl.DGLGraph):
        g = dgl.edge_type_subgraph(batch_graph, ["origin"])
        if self.hparams.learnable_feat:
            g.ndata["feat"] = self.feat(g.ndata[dgl.NID])
        g = dgl.to_homogeneous(g, ndata=["feat"])

        feat = g.ndata["feat"]
        init_g = dgl.batch(
            [complete_graph(n).to(self.device) for n in batch_graph.batch_num_nodes()]
        )

        h = self.encoder(init_g, feat)  # (num_edges, dim_edge_feat)
        h = self.mapper(h)
        init_g.edata["feat"] = h  # (num_edges, num_mixture, dim_edge_feat)
        graphs = dgl.unbatch(batch_graph)
        init_graphs = dgl.unbatch(init_g)

        edge_feat = []
        label = []
        for g, i_g in zip(graphs, init_graphs):
            for i in range(self.hparams.num_mixture):
                sub_g = dgl.edge_type_subgraph(g, [f"{i}"])
                sub_g = dgl.to_homogeneous(sub_g)
                neg_srcs, neg_dsts = dgl.sampling.global_uniform_negative_sampling(
                    sub_g, sub_g.num_edges()
                )

                pos_srcs, pos_dsts = sub_g.edges()

                neg_edges = i_g.edge_ids(neg_srcs, neg_dsts)
                pos_edges = i_g.edge_ids(pos_srcs, pos_dsts)

                sub_edge_feat = torch.cat(
                    (
                        i_g.edata["feat"][pos_edges, i, :],
                        i_g.edata["feat"][neg_edges, i, :],
                    ),
                    dim=0,
                )  # (num_pos + num_neg, dim_edge_feat)
                edge_feat.append(sub_edge_feat)

                label.append(
                    torch.tensor(
                        [1 for _ in range(pos_srcs.size(0))]
                        + [0 for _ in range(neg_srcs.size(0))],
                        device=self.device,
                        dtype=torch.float32,
                    )
                )  # (num_pos + num_neg, )

        edge_feat = torch.cat(
            edge_feat, dim=0
        )  # (num_nodes * num_mixture, dim_edge_feat)
        label = torch.cat(label, dim=0)  # (num_nodes * num_mixture,)

        logits = self.decoder(edge_feat)  # (num_nodes * num_mixture, 1)
        logits = self.bn(logits)
        # print(logits.squeeze(-1))
        # print(label)
        loss = self.criterion(logits.squeeze(-1), label)
        return loss

    def training_step(self, batch: dgl.DGLHeteroGraph, batch_idx: int) -> STEP_OUTPUT:
        loss = self.train_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def val_test_step(self, batch_graph):
        g = dgl.edge_type_subgraph(batch_graph, ["origin"])
        if self.hparams.learnable_feat:
            g.ndata["feat"] = self.feat(g.ndata[dgl.NID])
        g = dgl.to_homogeneous(g, ndata=["feat"])

        feat = g.ndata["feat"]
        init_g = dgl.batch(
            [complete_graph(n).to(self.device) for n in batch_graph.batch_num_nodes()]
        )

        h = self.encoder(init_g, feat)  # (num_edges, dim_edge_feat)
        h = self.mapper(h)
        init_g.edata["feat"] = h  # (num_edges, dim_edge_feat)

        graphs = dgl.unbatch(batch_graph)
        init_graphs = dgl.unbatch(init_g)

        edge_feat = []
        label = []
        for g, i_g in zip(graphs, init_graphs):
            sub_edge_feat = i_g.edata["feat"]
            edge_feat.append(sub_edge_feat)

            sub_label = []
            for i in range(self.hparams.num_mixture):
                sub_g = dgl.edge_type_subgraph(g, [f"{i}"])
                sub_g = dgl.to_homogeneous(sub_g)
                sub_label.append(sub_g.adjacency_matrix().to_dense().flatten())
            sub_label = torch.stack(sub_label, dim=0)  # (num_mixture, num_nodes ** 2)
            sub_label = sub_label.T  # (num_nodes ** 2, num_mixture)
            label.append(sub_label)

        edge_feat = torch.cat(
            edge_feat, dim=0
        )  # (batch_num_nodes ** 2, num_mixture, dim_edge_feat)
        label = torch.cat(label, dim=0)  # (batch_num_nodes ** 2, num_mixture)

        logits = self.decoder(edge_feat)

        return logits.squeeze(-1), label

    def validation_step(self, batch: dgl.DGLHeteroGraph, batch_idx: int) -> STEP_OUTPUT:
        preds, targets = self.val_test_step(batch)

        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)

        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: dgl.DGLHeteroGraph, batch_idx: int) -> STEP_OUTPUT:
        preds, targets = self.val_test_step(batch)

        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)

        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/recall",
            self.test_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
