import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import torch
from src.opengsl.module.encoder import (
    GCNDiagEncoder,
    GCNEncoder,
    APPNPEncoder,
    GINEncoder,
)
from src.opengsl.module.fuse import Interpolate
from src.opengsl.module.transform import Normalize, KNN, Symmetry
from src.opengsl.module.metric import InnerProduct


class GRCN(torch.nn.Module):
    def __init__(
        self,
        num_nodes,
        num_features,
        num_hidden,
        num_classes,
        device,
        encoder_type="gcn",
    ):
        super(GRCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        if encoder_type == "gcn":
            self.conv_task = GCNEncoder(
                num_features,
                num_hidden,
                num_classes,
            )
        elif encoder_type == "appnp":
            self.conv_task = APPNPEncoder(
                num_features,
                num_hidden,
                num_classes,
                spmm_type=1,
            )
        elif encoder_type == "gin":
            self.conv_task = GINEncoder(
                num_features,
                num_hidden,
                num_classes,
                spmm_type=1,
            )
        self.model_type = "diag"
        if self.model_type == "diag":
            self.conv_graph = GCNDiagEncoder(2, num_features)
        else:
            self.conv_graph = GCNEncoder(num_features, num_hidden, num_classes)

        self.K = 50
        self._normalize = True  # 用来决定是否对node embedding进行normalize

        self.metric = InnerProduct()
        self.normalize_a = Normalize(add_loop=False)
        self.normalize_e = Normalize("row-norm", p=2)
        self.knn = KNN(self.K, sparse_out=True)
        self.sym = Symmetry(1)
        self.fuse = Interpolate(1, 1)

    def graph_parameters(self):
        return list(self.conv_graph.parameters())

    def base_parameters(self):
        return list(self.conv_task.parameters())

    def cal_similarity_graph(self, node_embeddings):
        # 一个2head的相似度计算
        # 完全等价于普通cosine
        similarity_graph = self.metric(node_embeddings[:, : int(self.num_features / 2)])
        similarity_graph += self.metric(
            node_embeddings[:, int(self.num_features / 2) :]
        )
        return similarity_graph

    def _sparse_graph(self, raw_graph):
        new_adj = self.knn(adj=raw_graph)
        new_adj = self.sym(new_adj)
        return new_adj

    def _node_embeddings(self, input, Adj):
        norm_Adj = self.normalize_a(Adj)
        node_embeddings = self.conv_graph(input, norm_Adj)
        if self._normalize:
            node_embeddings = self.normalize_e(node_embeddings)
        return node_embeddings

    def forward(self, input, Adj):
        adjs = {}
        Adj.requires_grad = False
        node_embeddings = self._node_embeddings(input, Adj)
        Adj_new = self.cal_similarity_graph(node_embeddings)
        Adj_new = self._sparse_graph(Adj_new)
        Adj_final = self.fuse(Adj_new, Adj)
        Adj_final_norm = self.normalize_a(Adj_final.coalesce())
        x = self.conv_task(input, Adj_final_norm)

        adjs["new"] = Adj_new
        adjs["final"] = Adj_final

        return x, adjs


if __name__ == "__main__":
    import dgl

    model = GRCN(30, 256, 128, 10, device="cpu")
    g = dgl.rand_graph(30, 100)
    # print(model)
    embeddings = torch.rand((30, 256))
    src, dst = g.edges()
    adj = torch.sparse_coo_tensor(
        indices=torch.stack([src, dst]),
        values=torch.ones_like(src).float(),
        size=(30, 30),
    )
    x, adj = model(embeddings, adj)
    print(x)
    print(adj.size())
