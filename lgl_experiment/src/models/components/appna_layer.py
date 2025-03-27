import dgl
import torch
from dgl.nn.pytorch import APPNPConv
from torch.nn import Module, ReLU, Sequential, Linear


class APPNALayer(Module):
    def __init__(self, k, alpha, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv = APPNPConv(k, alpha)
        self.trans = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        g = dgl.add_self_loop(g)
        feat = self.conv(g, feat)
        feat = self.trans(feat)
        return feat