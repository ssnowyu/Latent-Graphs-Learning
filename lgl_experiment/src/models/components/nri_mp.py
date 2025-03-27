import dgl
import torch
from torch import nn
import dgl.function as fn


class  NRIMessagePassing(nn.Module):
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
        self.src_trans = nn.Linear(feat_dim, feat_dim, bias=False)
        self.dst_trans = nn.Linear(feat_dim, feat_dim, bias=True)
        self.res_trans = nn.Linear(feat_dim, feat_dim)


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
        src_feat = self.src_trans(feat)
        dst_feat = self.dst_trans(feat)
        graph.ndata.update({'src_feat': src_feat, 'dst_feat': dst_feat})
        graph.update_all(
            message_func=fn.u_add_v('src_feat', 'dst_feat', 'message_feat'),
            reduce_func=fn.sum('message_feat', 'agg_feat'),
        )
        out_feat = self.res_trans(graph.ndata['agg_feat'])
        src_feat = self.src_trans(out_feat)
        dst_feat = self.dst_trans(out_feat)
        graph.ndata.update({'src_feat': src_feat, 'dst_feat': dst_feat})
        graph.apply_edges(fn.u_add_v('src_feat', 'dst_feat', 'out_feat'))
        return graph.edata['out_feat']

if __name__ == '__main__':
    g = dgl.graph(([0, 1, 2, 3, 2, 5, 4], [1, 2, 3, 4, 0, 3, 1]))
    feat = torch.ones(6, 10)
    mp = NRIMessagePassing(10)
    res = mp(g, feat)
    pass
