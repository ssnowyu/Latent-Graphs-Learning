import pickle
from typing import Union, List, Tuple

import torch
from torch import nn
from torch_geometric.data import Data, InMemoryDataset


class SequentialGraphDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["nodes.pkl", "true_edges.pkl", "observed_edges.pkl"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        nodes = pickle.load(open(self.raw_paths[0], "rb"))
        true_edges = pickle.load(open(self.raw_paths[1], "rb"))
        observed_edges = pickle.load(open(self.raw_paths[2], "rb"))

        num_sample = len(nodes)

        observed_features = []
        merged_edges = []
        for node_index, edges in zip(nodes, observed_edges):
            # merge observed edges
            tmp_edges = []
            for es in edges:
                tmp_edges.extend(es)
            tmp_edges = list(set(tmp_edges))
            tmp_edges = [
                (node_index.index(s), node_index.index(t)) for t, s in tmp_edges
            ]

            merged_edges.append(torch.LongTensor(tmp_edges).permute(1, 0))

            adj_matrices = []
            for es in edges:
                sources, targets = zip(*es)
                sources = [node_index.index(s) for s in sources]
                targets = [node_index.index(t) for t in targets]
                adj = torch.sparse_coo_tensor(
                    indices=torch.LongTensor([sources, targets]),
                    values=torch.ones(size=(len(sources),)),
                    dtype=torch.float32,
                    size=(len(node_index), len(node_index)),
                ).to_dense()
                adj_matrices.append(adj)
            observed_features.append(torch.cat(adj_matrices, dim=-1))

        ground_truth = []
        for node_index, edges in zip(nodes, true_edges):
            sources, targets = zip(*edges)
            sources = [node_index.index(s) for s in sources]
            targets = [node_index.index(t) for t in targets]
            adj = torch.sparse_coo_tensor(
                indices=torch.LongTensor([sources, targets]),
                values=torch.ones(size=(len(sources),)),
                dtype=torch.float32,
                size=(len(node_index), len(node_index)),
            ).to_dense()
            ground_truth.append(adj)

        # expand feature by 0
        max_length = max([feature.size(-1) for feature in observed_features])
        observed_features = [
            nn.ZeroPad2d((0, max_length - feature.size(-1), 0, 0))(feature)
            for feature in observed_features
        ]
        max_length = max([feature.size(-1) for feature in observed_features])
        ground_truth = [
            nn.ZeroPad2d((0, max_length - feature.size(-1), 0, 0))(feature)
            for feature in ground_truth
        ]

        # # graph without edge
        # data_list = [Data(x=observed_features[i], y=ground_truth[i]) for i in range(num_sample)]

        data_list = [
            Data(x=observed_features[i], edge_index=merged_edges[i], y=ground_truth[i])
            for i in range(num_sample)
        ]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = SequentialGraphDataset(root="../../../data")
