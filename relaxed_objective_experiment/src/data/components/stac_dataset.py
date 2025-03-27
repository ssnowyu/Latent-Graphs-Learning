import os
import dgl
from dgl.data import DGLDataset


class STACDataset(DGLDataset):
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

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
