from typing import Any, Dict, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
import dgl
from torch.nn import CrossEntropyLoss


class SublimeModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        num_mixture: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["encoder"])

        self.encoder = encoder

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

        self.criterion = CrossEntropyLoss()

        self.num_mixture = num_mixture

        self.sigmoid = torch.nn.Sigmoid()

    def training_step(self, batch: dgl.DGLHeteroGraph, batch_idx: int) -> STEP_OUTPUT:
        graphs = dgl.unbatch(batch)

        all_logits = []
        for g in graphs:
            origin_g = dgl.edge_type_subgraph(g, ["origin"])
            origin_g = dgl.to_homogeneous(origin_g, ndata=["feat"])
            feat = origin_g.ndata["feat"]
            origin_g.edata["w"] = torch.ones(origin_g.num_edges(), device=self.device)

            logits, _ = self.encoder(feat, origin_g)
            all_logits.append(logits)

        logits = torch.cat(all_logits, dim=0)
        labels = batch.ndata["label"]

        loss = self.criterion(self.sigmoid(logits), labels.float())
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: dgl.DGLHeteroGraph, batch_idx: int) -> STEP_OUTPUT:
        graphs = dgl.unbatch(batch)

        for g in graphs:
            origin_g = dgl.edge_type_subgraph(g, ["origin"])
            origin_g = dgl.to_homogeneous(origin_g, ndata=["feat"])
            feat = origin_g.ndata["feat"]
            origin_g.edata["w"] = torch.ones(origin_g.num_edges(), device=self.device)
            _, new_g = self.encoder(feat, origin_g)
            latent_g = dgl.edge_type_subgraph(
                g, [f"{i}" for i in range(self.hparams.num_mixture)]
            )
            latent_g = dgl.to_homogeneous(latent_g)
            label_adj = latent_g.adjacency_matrix().to_dense()
            optimized_adj = new_g.adjacency_matrix().to_dense()
            self.val_precision(optimized_adj, label_adj)
            self.val_recall(optimized_adj, label_adj)
            self.val_f1(optimized_adj, label_adj)

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
        graphs = dgl.unbatch(batch)

        for g in graphs:
            origin_g = dgl.edge_type_subgraph(g, ["origin"])
            origin_g = dgl.to_homogeneous(origin_g, ndata=["feat"])
            feat = origin_g.ndata["feat"]
            origin_g.edata["w"] = torch.ones(origin_g.num_edges(), device=self.device)
            _, new_g = self.encoder(feat, origin_g)
            latent_g = dgl.edge_type_subgraph(
                g, [f"{i}" for i in range(self.hparams.num_mixture)]
            )
            latent_g = dgl.to_homogeneous(latent_g)
            label_adj = latent_g.adjacency_matrix().to_dense()
            optimized_adj = new_g.adjacency_matrix().to_dense()
            self.test_precision(optimized_adj, label_adj)
            self.test_recall(optimized_adj, label_adj)
            self.test_f1(optimized_adj, label_adj)

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
