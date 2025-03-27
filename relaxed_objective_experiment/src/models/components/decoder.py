from torch.nn import Module, ReLU, Linear


class NodeMLPDecoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ) -> None:
        super().__init__()
        self.linear1 = Linear(dim_in * 2, dim_hidden)
        self.linear2 = Linear(dim_hidden, 1)
        self.relu = ReLU()

    def forward(self, feat):
        x = self.linear1(feat)
        x = self.relu(x)
        return self.linear2(x)


class EdgeMLPDecoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ) -> None:
        super().__init__()
        self.linear1 = Linear(dim_in * 2, dim_hidden)
        self.linear2 = Linear(dim_hidden, 1)
        self.relu = ReLU()

    def forward(self, feat):
        x = self.linear1(feat)
        x = self.relu(x)
        return self.linear2(x)
