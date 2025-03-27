import os
import dgl
import torch
from dgl.data import DGLDataset


class LabeledSTACDataset(DGLDataset):
    def __init__(
        self,
        raw_dir=None,
        num_mixture=5,
        save_dir=None,
        force_reload=False,
        verbose=False,
    ):
        self.num_mixture = num_mixture
        super().__init__(
            name="STACDataset",
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        path = os.path.join(self.raw_dir, f"data_{self.num_mixture}.bin")
        self.graphs, _ = dgl.load_graphs(path)
        for g in self.graphs:
            l = torch.zeros((g.num_nodes(), self.num_mixture), dtype=torch.int64)
            for i in range(self.num_mixture):
                srcs, dsts = g.edges(etype=("n", f"{i}", "n"))
                nodes = torch.cat((srcs, dsts), dim=0).unique()
                l[nodes, i] = 1
            g.ndata["label"] = l

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


if __name__ == "__main__":
    dataset = LabeledSTACDataset(raw_dir="../../../data/STAC/processed")
    g = dataset.__getitem__(2)
    print(g.ndata["label"])
