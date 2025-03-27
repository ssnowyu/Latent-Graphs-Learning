import dgl
import torch
from dgl.nn.pytorch import GraphConv
from torch.nn import Module, ReLU


class GCNLayer(Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, latent_dim)
        self.relu = ReLU()

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        g = dgl.add_self_loop(g)
        feat = self.relu(self.conv1(g, feat))
        feat = self.relu(self.conv2(g, feat))
        feat = self.conv3(g, feat)
        return feat