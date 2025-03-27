import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import os
import argparse
import torch
import networkx as nx
from tqdm import tqdm
import numpy as np
import dgl

from src.utils.seed_utils import seed_everything

seed_everything(12345)

RAW_DIR = "data/custom/raw"
PROCESSED_DIR = "data/custom/processed"
DIM_FEATURE = 768
NUM_DIALOGUES_PER_SAMPLE = 10
NUM_SAMPLES = 2000


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_samples", type=int, default=2000, help="The number of samples"
)
parser.add_argument(
    "--num_mixture",
    type=int,
    default=5,
    help="The number of latent graphs in a mixed graph",
)
parser.add_argument(
    "--min_node_number",
    type=int,
    default=20,
    help="Minimum number of nodes in the a latent graph",
)
parser.add_argument(
    "--max_node_number",
    type=int,
    default=30,
    help="Maximum number of nodes in the a latent graph",
)
parser.add_argument("--overlap_rate", type=float, default=0.1)

args = parser.parse_args()


num_nodes_matrix = np.random.randint(
    args.min_node_number, args.max_node_number + 1, (args.num_samples, args.num_mixture)
)
num_attached_edges_matrix = np.random.randint(
    5, 10, (args.num_samples, args.num_mixture)
)

num_all_nodes = 5000
original_node_ids = list(range(num_all_nodes))
print(f"Number of nodes: {num_all_nodes}")

graphs = []
for i in tqdm(
    range(args.num_samples),
    desc="Processing",
    unit="item",
):
    nodes = []
    edges = []
    node_id_offset = 0
    graph_data = {}
    latent_graph_edges = []
    for j in range(args.num_mixture):
        num_nodes = num_nodes_matrix[i, j]
        num_attached_edges = num_attached_edges_matrix[i, j]

        nx_g = nx.barabasi_albert_graph(n=num_nodes, m=num_attached_edges)
        offset_nodes = np.array(nx_g.nodes) + node_id_offset
        offset_edges = np.array(nx_g.edges) + node_id_offset
        edges.append(offset_edges)

        node_id_offset += num_nodes

        # srcs = np.concatenate((offset_edges[:, 0], offset_edges[:, 1]), axis=0)
        # dsts = np.concatenate((offset_edges[:, 1], offset_edges[:, 0]), axis=0)

        # graph_data[("n", f"{j}", "n")] = (
        #     torch.from_numpy(srcs),
        #     torch.from_numpy(dsts),
        # )

    edges_matrix = np.concatenate(edges, axis=0)
    num_nodes = node_id_offset
    new_num_nodes = int(num_nodes * (1.0 - args.overlap_rate))
    new_node_ids = list(range(new_num_nodes))
    selected_ids = np.random.choice(new_node_ids, size=num_nodes, replace=True)
    mixed_edges = np.take(selected_ids, edges_matrix)
    mixed_srcs = mixed_edges[:, 0]
    mixed_dsts = mixed_edges[:, 1]
    graph_data[("n", "origin", "n")] = (
        torch.from_numpy(mixed_srcs),
        torch.from_numpy(mixed_dsts),
    )
    for idx, each_edges in enumerate(edges):
        reindex_edges = np.take(selected_ids, each_edges)
        graph_data[("n", f"{idx}", "n")] = (
            torch.from_numpy(reindex_edges[:, 0]),
            torch.from_numpy(reindex_edges[:, 1]),
        )

    g = dgl.heterograph(graph_data)

    # original NID
    original_nids = np.random.choice(
        original_node_ids, size=g.num_nodes(), replace=False
    )

    g.ndata[dgl.NID] = torch.tensor(original_nids)

    graphs.append(g)

dgl.save_graphs(
    os.path.join(
        PROCESSED_DIR, f"data_mix_{args.num_mixture}_overlap_{args.overlap_rate}.bin"
    ),
    graphs,
)
