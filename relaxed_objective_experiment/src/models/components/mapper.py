from math import floor
from torch.nn import Module, ReLU, Sequential, Linear, Flatten
import torch


class MLPMapper(Module):
    def __init__(
        self,
        dim_in: int,
        num_mixture: int,
    ) -> None:
        super().__init__()
        dim_hidden = dim_in * floor(num_mixture / 2)
        self.fc1 = Linear(dim_in, dim_hidden)
        self.fc2 = Linear(dim_hidden, dim_in * num_mixture)
        self.relu = ReLU()
        self.num_mixture = num_mixture
        self.dim_in = dim_in

    def forward(self, feat: torch.Tensor):
        # input shape: (*, dim_feat)
        x = self.relu(self.fc1(feat))
        x = self.fc2(x)
        return x.view(-1, self.num_mixture, self.dim_in)
