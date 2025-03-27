from typing import Any, Optional

import torch
import dgl

def complete_graph(num_nodes: int):
    src, dst = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes))
    src = src.flatten()
    dst = dst.flatten()
    g = dgl.DGLGraph((src, dst))
    return g

def init_node_feature(g: dgl.DGLGraph, feat_dim: int, conv: torch.nn.Module, device: Optional[Any] = None) -> torch.Tensor:
    if device is None:
        device = 'cpu'
    random_node_feat = torch.randn(g.num_nodes(), feat_dim).to(device)
    edge_step = g.edata["edge_step"]
    num_step = edge_step.size(1)
    output_node_feat = []
    for i in range(num_step):
        mask = edge_step[:, i].bool()
        sub_g = dgl.edge_subgraph(g, mask)
        node_feat_for_sub_g = random_node_feat[sub_g.ndata[dgl.NID]]

        sub_g = dgl.add_self_loop(sub_g)
        output_node_feat_for_sub_g = conv(
            sub_g, node_feat_for_sub_g
        )
        padding_feat = torch.zeros(g.num_nodes(), feat_dim).to(device)
        padding_feat[sub_g.ndata[dgl.NID]] = output_node_feat_for_sub_g

        output_node_feat.append(padding_feat)

    output_node_feat = torch.stack(output_node_feat, dim=0)
    return output_node_feat.sum(dim=0)