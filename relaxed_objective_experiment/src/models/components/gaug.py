import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from src.models.components.opengsl_encoder import GCN, APPNP, GIN
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro as pyro

# from opengsl.data.preprocess.normalize import normalize
from src.utils.opengsl_utils import (
    scipy_sparse_to_sparse_tensor,
    sparse_tensor_to_scipy_sparse,
)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl


def normalize(mx, style="symmetric", add_loop=True, sparse=False):
    """
    Normalize the feature matrix or adj matrix.

    Parameters
    ----------
    mx : torch.tensor
        Feature matrix or adj matrix to normalize.
    style: str
        If set as ``row``, `mx` will be row-wise normalized.
        If set as ``symmetric``, `mx` will be normalized as in GCN.

    add_loop : bool
        Whether to add self loop.
    sparse : bool
        Whether the matrix is stored in sparse form. The returned tensor will be the same form.

    Returns
    -------
    normalized_mx : torch.tensor
        The normalized matrix.

    """
    if style == "row":
        return row_nomalize(mx)
    elif style == "symmetric":
        if sparse:
            return normalize_sp_tensor(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)


def row_nomalize(mx):
    """Row-normalize sparse matrix."""
    device = mx.device
    mx = mx.cpu().numpy()
    r_sum = np.array(mx.sum(1))
    r_inv = np.power(r_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.tensor(mx).to(device)
    return mx


def normalize_tensor(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A


def normalize_sp_tensor(adj, add_loop=True):
    device = adj.device
    adj = sparse_tensor_to_scipy_sparse(adj)
    adj = normalize_sp_matrix(adj, add_loop)
    adj = scipy_sparse_to_sparse_tensor(adj).to(device)
    return adj


def normalize_sp_matrix(adj, add_loop=True):
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new


class VGAE(nn.Module):
    """GAE/VGAE as edge prediction model"""

    def __init__(
        self,
        dim_feats,
        n_hidden,
        n_embed,
    ):
        super(VGAE, self).__init__()
        self.gae = True
        # self.gcn_base = GraphConvolution(dim_feats, dim_h, with_bias=False)
        # self.gcn_mean = GraphConvolution(dim_h, dim_z, with_bias=False)
        # self.gcn_logstd = GraphConvolution(dim_h, dim_z, with_bias=False)
        self.conv_graph = GCN(
            dim_feats,
            n_hidden,
            n_embed,
            bias=False,
            weight_initializer="glorot",
        )

    def forward(self, feats, adj):
        # GCN encoder
        # hidden = self.gcn_base(feats, adj)
        # self.mean = F.relu(self.gcn_mean(hidden, adj))
        _, mean = self.conv_graph((feats, adj, False))
        mean = F.relu(mean)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = mean
        else:
            # VGAE
            # self.logstd = F.relu(self.gcn_logstd(hidden, adj))
            # gaussian_noise = torch.randn_like(self.mean)
            # sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
            # Z = sampled_Z
            pass
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


class GAug(nn.Module):
    def __init__(self, dim_feats, n_hidden, n_classes, encoder_type="gcn"):
        super(GAug, self).__init__()
        self.temperature = 0.3
        self.alpha = 1
        # edge prediction network
        self.ep_net = VGAE(dim_feats, n_hidden, n_hidden)
        # node classification network
        # self.nc_net = GCN(dim_feats, dim_h, n_classes, dropout=dropout)
        if encoder_type == "gcn":
            self.nc_net = GCN(
                dim_feats,
                n_hidden,
                n_classes,
                weight_initializer="glorot",
                bias_initializer="zeros",
            )
        elif encoder_type == "appnp":
            self.nc_net = APPNP(
                dim_feats,
                n_hidden,
                n_classes,
            )
        elif encoder_type == "gin":
            self.nc_net = GIN(
                dim_feats,
                n_hidden,
                n_classes,
            )

    def sample_adj(self, adj_logits):
        """sample an adj from the predicted edge probabilities of ep_net"""
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling

        # print(adj_logits)
        # print(edge_probs)
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(
            temperature=self.temperature, probs=edge_probs
        ).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        # print(adj_orig)
        edge_probs = alpha * edge_probs + (1 - alpha) * adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(
            temperature=self.temperature, probs=edge_probs
        ).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def forward(self, feats, adj, adj_orig=None):
        # print(feats)
        # print(adj)
        adj_logits = self.ep_net(feats, adj)
        # print(adj_logits)
        if self.alpha == 1:
            adj_new = self.sample_adj(adj_logits)
        else:
            adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)
        adj_new_normed = normalize(adj_new)
        hidden, output = self.nc_net((feats, adj_new_normed, False))
        return output, adj_logits, adj_new


def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges.T]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score


class MultipleOptimizer:
    """a class that wraps multiple optimizers"""

    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer"""
        for param_group in self.optimizers[op_index].param_groups:
            param_group["lr"] = new_lr


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    """schedule the learning rate with the sigmoid function.
    The learning rate will start with near zero and end with near lr"""
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    # range the factors to [0, 1]
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule


if __name__ == "__main__":
    model = GAug(256, 128, 5)
    g = dgl.rand_graph(30, 100)
    # print(model)
    embeddings = torch.rand((30, 256))
    src, dst = g.edges()
    adj = torch.sparse_coo_tensor(
        indices=torch.stack([src, dst]),
        values=torch.ones_like(src).float(),
        size=(30, 30),
    )
    x, adj_logits, adj_new = model(embeddings, g.adjacency_matrix().to_dense())
    print(x)
    print(adj_logits)
    print(adj_new)
