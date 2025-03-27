from typing import Optional

import dgl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn


class Frequency4Graph(LightningModule):
    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__()

        self.save_hyperparameters()

        self.dummy_param = nn.Parameter(torch.zeros(1))

    def training_step(self, batch, idx) -> STEP_OUTPUT:
        return torch.zeros(1, requires_grad=True).to(self.device)

    def validation_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        pass

    def test_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        g, label, mask = batch

        preds = []
        targets = []
        for each_g, each_label, each_mask in zip(dgl.unbatch(g), label, mask):
            num_nodes = each_g.num_nodes()

            positive_indices = [
                (i * num_nodes + j).item()
                for i, j in torch.masked_select(each_label, each_mask.bool())
                .view(2, -1)
                .unbind(-1)
            ]
            target = torch.zeros(num_nodes ** 2).to(self.device).float()
            target[positive_indices] = 1
            targets.append(target)

            # preds.append(g.adjacency_matrix().view(-1))
            adj_matrix = torch.zeros(num_nodes ** 2).to(self.device).float()
            srcs, dsts = each_g.edges()
            edge_indices = [(src * num_nodes + dst) for src, dst in zip(srcs, dsts)]
            adj_matrix[edge_indices] = 1
            preds.append(adj_matrix)

        targets = torch.cat(targets, dim=-1)
        preds = torch.cat(preds, dim=-1)

        return {"preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}