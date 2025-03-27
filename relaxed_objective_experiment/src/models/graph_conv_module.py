from typing import Any, Dict, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
import dgl
from torch.nn import BCEWithLogitsLoss


class GraphConvModule(LightningModule):
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
        self.save_hyperparameters(logger=False, ignore=["encoder", "decoder", "mapper"])

        self.encoder = encoder
        self.decoder = decoder
        self.mapper = mapper

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

        if learnable_feat:
            if num_features is not None and dim_feature is not None:
                self.feat = torch.nn.Embedding(num_features, dim_feature)
            else:
                raise ValueError(
                    "num_features and dim_feature are expected if learnable_feat is True"
                )

    def train_step(self, batch_graph):
        g = dgl.edge_type_subgraph(batch_graph, ["origin"])
        if self.hparams.learnable_feat:
            g.ndata["feat"] = self.feat(g.ndata[dgl.NID])
        g = dgl.to_homogeneous(g, ndata=["feat"])

        feat = g.ndata["feat"]
        h = self.encoder(g, feat)  # [num_nodes, dim_feat]
        h = self.mapper(h)  # [num_nodes, num_mixture, dim_feat]

        graphs = dgl.unbatch(batch_graph)

        offset = 0
        src_feat = []
        dst_feat = []
        label = []
        for g in graphs:
            for i in range(self.hparams.num_mixture):
                sub_g = dgl.edge_type_subgraph(g, [f"{i}"])
                sub_g = dgl.to_homogeneous(sub_g)
                neg_srcs, neg_dsts = dgl.sampling.global_uniform_negative_sampling(
                    sub_g, sub_g.num_edges()
                )
                neg_srcs = neg_srcs + offset
                neg_dsts = neg_dsts + offset

                pos_srcs, pos_dsts = sub_g.edges()
                pos_srcs = pos_srcs + offset
                pos_dsts = pos_dsts + offset

                src_feat.append(
                    torch.cat((h[pos_srcs, i, :], h[neg_srcs, i, :]), dim=0)
                )  # (num_pos + num_neg, dim_feat)
                dst_feat.append(
                    torch.cat((h[pos_dsts, i, :], h[neg_dsts, i, :]), dim=0)
                )  # (num_pos + num_neg, dim_feat)
                label.append(
                    torch.tensor(
                        [1 for _ in range(pos_srcs.size(0))]
                        + [0 for _ in range(neg_srcs.size(0))],
                        device=self.device,
                        dtype=torch.float32,
                    )
                )  # ((num_pos + num_neg) * num_mixture, )
            offset += g.num_nodes()
        src_feat = torch.cat(src_feat, dim=0)  # (num_nodes * num_mixture, dim_feat)
        dst_feat = torch.cat(dst_feat, dim=0)  # (num_nodes * num_mixture, dim_feat)
        label = torch.cat(label, dim=0)  # (num_nodes * num_mixture,)

        logits = self.decoder(
            torch.cat((src_feat, dst_feat), dim=-1)
        )  # (num_nodes * num_mixture, 1)
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
        h = self.encoder(g, feat)
        h = self.mapper(h)

        graphs = dgl.unbatch(batch_graph)

        offset = 0
        src_feat = []
        dst_feat = []
        label = []
        for g in graphs:
            num_nodes = g.num_nodes()
            sub_h = h[
                offset : offset + num_nodes, :, :
            ]  # (num_nodes, num_mixture, dim_feat)
            src_feat.append(
                sub_h.repeat_interleave(num_nodes, dim=0)
            )  # (num_nodes ** 2, num_mixture, dim_feat)
            dst_feat.append(
                sub_h.repeat(num_nodes, 1, 1)
            )  # (num_nodes ** 2, num_mixture, dim_feat)

            sub_label = []
            for i in range(self.hparams.num_mixture):
                sub_g = dgl.edge_type_subgraph(g, [f"{i}"])
                sub_g = dgl.to_homogeneous(sub_g)
                sub_label.append(sub_g.adjacency_matrix().to_dense().flatten())
            sub_label = torch.stack(sub_label, dim=0)  # (num_mixture, num_nodes ** 2)
            sub_label = sub_label.T  # (num_nodes ** 2, num_mixture)
            label.append(sub_label)

            offset += num_nodes

        src_feat = torch.cat(src_feat, dim=0)
        dst_feat = torch.cat(dst_feat, dim=0)
        label = torch.cat(label, dim=0)

        logits = self.decoder(torch.cat((src_feat, dst_feat), dim=-1))

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
