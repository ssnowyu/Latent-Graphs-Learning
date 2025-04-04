import os.path
from typing import Optional, List, Dict, Tuple

import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence


class GraphDataset(DGLDataset):
    def __init__(
        self,
        name_for_observed_data: str,
        name_for_graph_data: str,
        raw_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        force_reload: bool = False,
        verbose: bool = False,
        num_time_step: Optional[int] = None,
        without_node_feature: bool = False,
        without_edge_feature: bool = False,
        processed_data_name: Optional[str] = None,
    ):
        self.name_for_observed_data = name_for_observed_data
        self.name_for_chain_data = name_for_graph_data
        self.num_time_step = num_time_step
        self.without_node_feature = without_node_feature
        self.without_edge_feature = without_edge_feature
        self.graphs: List[dgl.DGLGraph] = []
        self.labels: Optional[torch.Tensor] = None
        self.label_masks: Optional[torch.Tensor] = None
        self.processed_data_name = processed_data_name

        super(GraphDataset, self).__init__(
            name="",
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        path_for_observed_data = os.path.join(self.raw_dir, self.name_for_observed_data)
        path_for_label_data = os.path.join(self.raw_path, self.name_for_chain_data)
        observed_data = pd.read_csv(path_for_observed_data)
        graph_data = pd.read_csv(path_for_label_data)

        grouped_observed_data = observed_data.groupby("sample_id")
        grouped_label_data = graph_data.groupby("sample_id")

        labels = []
        for (_, each_observed_data), (_, each_label_data) in zip(
            grouped_observed_data, grouped_label_data
        ):
            nodes = each_label_data[["source", "target"]].stack().unique().tolist()

            # graph data
            edges: Dict[Tuple[int, int], torch.Tensor] = {}
            for i in range(self.num_time_step):
                observed_data_for_each_step = each_observed_data[
                    each_observed_data["time_step"] == i
                ]
                observed_source_node_indices = [
                    nodes.index(n) for n in observed_data_for_each_step["source"]
                ]
                observed_target_node_indices = [
                    nodes.index(n) for n in observed_data_for_each_step["target"]
                ]
                added_edges = [
                    (src, dst)
                    for src, dst in zip(
                        observed_source_node_indices, observed_target_node_indices
                    )
                    if (src, dst) in edges
                ]
                unadded_edges = [
                    (src, dst)
                    for src, dst in zip(
                        observed_source_node_indices, observed_target_node_indices
                    )
                    if (src, dst) not in edges
                ]
                for e in added_edges:
                    edges[e][0][i] = 1
                edges.update(
                    {
                        (src, dst): one_hot(torch.LongTensor([i]), self.num_time_step)
                        for src, dst in unadded_edges
                    }
                )

            if not self.without_node_feature:
                pass
            if not self.without_edge_feature:
                pass

            g = dgl.graph(tuple(zip(*edges.keys())), num_nodes=len(nodes))
            g.edata["edge_step"] = torch.cat(list(edges.values()), dim=0)
            self.graphs.append(g)

            label_src_indices = [nodes.index(n) for n in each_label_data["source"]]
            label_dst_indices = [nodes.index(n) for n in each_label_data["target"]]
            labels.append(
                torch.tensor((label_src_indices, label_dst_indices), dtype=torch.int64)
            )

        self.labels = pad_sequence(
            [item.T for item in labels], batch_first=True, padding_value=-1
        ).permute(0, 2, 1)
        self.label_masks = self.labels != -1

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.label_masks[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        dgl.save_graphs(
            self.save_graph_path,
            self.graphs,
            {"labels": self.labels, "label_masks": self.label_masks},
        )

    def load(self):
        self.graphs, label_dict = dgl.load_graphs(self.save_graph_path)
        self.labels = label_dict["labels"]
        self.label_masks = label_dict["label_masks"]

    def has_cache(self):
        self.save_graph_path = os.path.join(self.save_path, f"{self.processed_data_name}.bin")
        return os.path.exists(self.save_graph_path)


if __name__ == "__main__":
    dataset = GraphDataset(
        name_for_observed_data="2023-06-22-14-13-18-snapshots.csv",
        name_for_graph_data="2023-06-22-14-13-18-graph.csv",
        raw_dir="../../../data/raw",
        save_dir="../../../data/processed",
        num_time_step=5,
    )
    dataset.process()
