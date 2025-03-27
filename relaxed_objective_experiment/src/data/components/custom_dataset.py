import os
import dgl
from dgl.data import DGLDataset


class CustomDataset(DGLDataset):
    def __init__(
        self,
        raw_dir=None,
        num_mixture=5,
        overlap_rate=0.1,
        save_dir=None,
        force_reload=False,
        verbose=False,
    ):
        self.num_mixture = num_mixture
        self.overlap_rate = overlap_rate
        super().__init__(
            name="CustomDataset",
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        path = os.path.join(
            self.raw_dir, f"data_mix_{self.num_mixture}_overlap_{self.overlap_rate}.bin"
        )
        self.graphs, _ = dgl.load_graphs(path)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
