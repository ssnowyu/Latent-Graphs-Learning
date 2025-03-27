from torch.nn import Module, ModuleList, BatchNorm1d, ReLU, Linear
import torch
import dgl
from dgl.nn import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class GCNEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layer must be greater than 2."

        self.conv_in = GraphConv(dim_in, dim_hidden)
        self.bn_in = BatchNorm1d(dim_hidden)
        self.conv_out = GraphConv(dim_hidden, dim_out)
        self.bn_out = BatchNorm1d(dim_out)
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.relu = ReLU()

        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(dim_hidden, dim_hidden))
            self.bns.append(BatchNorm1d(dim_hidden))

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)

        h = self.conv_in(g, feat)
        h = self.bn_in(h)
        h = self.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(g, h)
            h = bn(h)
            h = self.relu(h)

        h = self.conv_out(g, h)
        h = self.bn_out(h)
        h = self.relu(h)

        return h


class GATEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layer must be greater than 2."

        self.conv_in = GATConv(dim_in, dim_hidden, num_heads=num_heads)

        self.bn_in = BatchNorm1d(dim_hidden * num_heads)
        self.conv_out = GATConv(dim_hidden * num_heads, dim_out, 1)
        self.bn_out = BatchNorm1d(dim_out)
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.relu = ReLU()

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(dim_hidden * num_heads, dim_hidden, num_heads=num_heads)
            )
            self.bns.append(BatchNorm1d(dim_hidden * num_heads))

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)

        h = self.conv_in(g, feat).flatten(-2)
        h = self.bn_in(h)
        h = self.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(g, h).flatten(-2)
            h = bn(h)
            h = self.relu(h)

        h = self.conv_out(g, h).flatten(-2)
        h = self.bn_out(h)
        h = self.relu(h)

        return h


class SAGEEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layer must be greater than 2."

        self.conv_in = SAGEConv(dim_in, dim_hidden, aggregator_type="pool")
        self.bn_in = BatchNorm1d(dim_hidden)
        self.conv_out = SAGEConv(dim_hidden, dim_out, aggregator_type="pool")
        self.bn_out = BatchNorm1d(dim_out)
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.relu = ReLU()

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(dim_hidden, dim_hidden, aggregator_type="pool"))
            self.bns.append(BatchNorm1d(dim_hidden))

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)

        h = self.conv_in(g, feat)
        h = self.bn_in(h)
        h = self.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(g, h)
            h = bn(h)
            h = self.relu(h)

        h = self.conv_out(g, h)
        h = self.bn_out(h)
        h = self.relu(h)

        return h


def u_cat_v(edges):
    return {
        "edge_feat": torch.cat((edges.src["src_feat"], edges.src["dst_feat"]), dim=-1)
    }


class MLPLayer(Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int) -> None:
        super().__init__()

        self.linear1 = Linear(dim_in, dim_hidden)
        self.linear2 = Linear(dim_hidden, dim_out)
        self.relu = ReLU()
        self.bn1 = BatchNorm1d(dim_hidden)
        # self.bn2 = BatchNorm1d(dim_out)

    def forward(self, x):
        h = self.linear1(x)
        h = self.bn1(h)
        h = self.relu(h)
        return self.linear2(h)


class NRIEncoder(Module):
    r"""
    NRI (Neural Network Inference) message passing

    Mathematically it is defined as follows:

    .. math::
      h^{k+1}_j=f_v(\sum_{i\in \mathcal N(j)}f_e([h^k_i;h^k_j]))

    Parameters
    ----------
    feat_dim : int
        Input feature size; i.e, the number of dimensions of :math:`h^k_i` and :math:`h^k_j`
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.src_trans_1 = MLPLayer(feat_dim, feat_dim, feat_dim)
        self.dst_trans_1 = MLPLayer(feat_dim, feat_dim, feat_dim)
        self.src_trans_2 = MLPLayer(feat_dim, feat_dim, feat_dim)
        self.dst_trans_2 = MLPLayer(feat_dim, feat_dim, feat_dim)
        self.res_trans = MLPLayer(feat_dim * 2, feat_dim, feat_dim)
        self.relu = ReLU()
        self.bn_1 = BatchNorm1d(feat_dim)
        self.bn_2 = BatchNorm1d(feat_dim)
        self.bn_3 = BatchNorm1d(feat_dim)
        self.bn_4 = BatchNorm1d(feat_dim)
        self.bn_5 = BatchNorm1d(feat_dim)
        self.bn_6 = BatchNorm1d(feat_dim)

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor):
        r"""
        Compute NRI message passing.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            It represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        src_feat = self.relu(self.bn_1(self.src_trans_1(feat)))
        dst_feat = self.relu(self.bn_2(self.dst_trans_1(feat)))
        graph.ndata.update({"src_feat": src_feat, "dst_feat": dst_feat})
        graph.update_all(
            message_func=u_cat_v,
            reduce_func=fn.sum("edge_feat", "agg_feat"),
        )
        node_feat = self.relu(self.bn_3(self.res_trans(graph.ndata["agg_feat"])))
        src_feat = self.relu(self.bn_4(self.src_trans_2(node_feat)))
        dst_feat = self.relu(self.bn_5(self.dst_trans_2(node_feat)))
        graph.ndata.update({"src_feat": src_feat, "dst_feat": dst_feat})
        graph.apply_edges(u_cat_v)
        return graph.edata["edge_feat"]
