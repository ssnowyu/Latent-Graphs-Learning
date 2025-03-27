from torch.nn import Module, Linear, ReLU, Sequential
import torch

class ScoreLayer(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
        )

    def forward(self, feat: torch.Tensor):
        return self.layer(feat)
