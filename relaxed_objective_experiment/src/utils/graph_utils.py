from typing import Any, Optional

import torch
import dgl


def complete_graph(num_nodes: int):
    src, dst = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes))
    src = src.flatten()
    dst = dst.flatten()
    g = dgl.DGLGraph((src, dst))
    return g
