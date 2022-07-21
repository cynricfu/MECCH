"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(
            tuple(tuple(canonical_etype) for canonical_etype in meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, target_ntype, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.target_ntype = target_ntype

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, x_dict):
        h = x_dict[self.target_ntype]

        for gnn in self.layers:
            h = gnn(g, h)

        return {self.target_ntype: self.predict(h)}

    def get_embs(self, g, x_dict):
        h = x_dict[self.target_ntype]

        for gnn in self.layers:
            h = gnn(g, h)

        return {self.target_ntype: self.predict(h)}, {self.target_ntype: h}


class HAN_lp(nn.Module):
    def __init__(self, g, metapaths_u, target_ntype_u, in_size_u, metapaths_v, target_ntype_v, in_size_v, hidden_size,
                 out_size, num_heads, dropout):
        super(HAN_lp, self).__init__()
        self.target_ntype_u = target_ntype_u
        self.target_ntype_v = target_ntype_v
        self.r = nn.Parameter(torch.Tensor(out_size))
        nn.init.ones_(self.r)
        # initial node embeddings
        if in_size_u < 0:
            in_size_u = hidden_size * num_heads[0]
            self.feats_u = nn.Parameter(torch.Tensor(g.num_nodes(target_ntype_u), in_size_u))
            nn.init.xavier_normal(self.feats_u)
        if in_size_v < 0:
            in_size_v = hidden_size * num_heads[0]
            self.feats_v = nn.Parameter(torch.Tensor(g.num_nodes(target_ntype_v), in_size_v))
            nn.init.xavier_normal(self.feats_v)

        self.model_u = HAN(metapaths_u, target_ntype_u, in_size_u, hidden_size, out_size, num_heads, dropout)
        self.model_v = HAN(metapaths_v, target_ntype_v, in_size_v, hidden_size, out_size, num_heads, dropout)

    def forward(self, pos_edges, neg_edges, g, x_dict):
        # set initial node embeddings
        if hasattr(self, 'feats_u'):
            x_dict_u = {self.target_ntype_u: self.feats_u}
        else:
            x_dict_u = {self.target_ntype_u: x_dict[self.target_ntype_u]}
        if hasattr(self, 'feats_v'):
            x_dict_v = {self.target_ntype_v: self.feats_v}
        else:
            x_dict_v = {self.target_ntype_v: x_dict[self.target_ntype_v]}

        h_u = self.model_u(g, x_dict_u)[self.target_ntype_u]
        h_v = self.model_v(g, x_dict_v)[self.target_ntype_v]

        pos_score = torch.sum(h_u[pos_edges[:, 0]] * self.r * h_v[pos_edges[:, 1]], dim=-1)
        neg_score = torch.sum(h_u[neg_edges[:, 0]] * self.r * h_v[neg_edges[:, 1]], dim=-1)

        return pos_score, neg_score
