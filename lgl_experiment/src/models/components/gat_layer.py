import dgl
import torch
from dgl.nn.pytorch import GATConv
from torch.nn import Module, ReLU


class GATLayer(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super().__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, num_heads)
        self.relu = ReLU()

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        g = dgl.add_self_loop(g)
        feat = self.conv1(g, feat).flatten(-2)
        feat = self.relu(feat)
        feat = self.conv2(g, feat)
        feat = feat.mean(-2)

        return feat