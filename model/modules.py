import torch as th
import torch.nn as nn
import dgl.function as fn


class HeteroEmbedLayer(nn.Module):
    def __init__(self, n_nodes_dict, embed_size):
        super(HeteroEmbedLayer, self).__init__()

        self.embed_size = embed_size

        # embed nodes for each node type
        self.embeds = nn.ModuleDict()
        for ntype, num in n_nodes_dict.items():
            self.embeds[ntype] = nn.Embedding(num, embed_size)

    def forward(self, nids_dict):
        return {ntype: self.embeds[ntype](nids) for ntype, nids in nids_dict.items()}


class HeteroLinearLayer(nn.Module):
    def __init__(self, in_dim_dict, out_dim, bias=True):
        super(HeteroLinearLayer, self).__init__()

        self.out_dim = out_dim

        # linear projection for each node type
        self.linears = nn.ModuleDict()
        for ntype, in_dim in in_dim_dict.items():
            if in_dim > 0:
                self.linears[ntype] = nn.Linear(in_dim, out_dim, bias)

    def forward(self, h_dict):
        return {ntype: self.linears[ntype](h) for ntype, h in h_dict.items()}


class ScorePredictor(nn.Module):
    def __init__(self, dim, target_etype):
        super(ScorePredictor, self).__init__()
        self.target_etype = target_etype
        self.r = nn.Parameter(th.Tensor(dim))
        nn.init.ones_(self.r)

    def forward(self, edge_subgraph, h_dict):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h_dict
            edge_subgraph.apply_edges(fn.u_mul_v('h', 'h', 'score'), etype=self.target_etype)
            edge_subgraph.edges[self.target_etype].data['score'] = th.sum(
                edge_subgraph.edges[self.target_etype].data['score'] * self.r, dim=-1)
            return edge_subgraph.edges[self.target_etype].data['score']


class LinkPrediction_minibatch(nn.Module):
    def __init__(self, emb_model, emb_dim, target_etype):
        super(LinkPrediction_minibatch, self).__init__()
        self.emb_model = emb_model
        self.pred = ScorePredictor(emb_dim, target_etype)

    def forward(self, positive_graph, negative_graph, blocks, x_dict):
        h_dict = self.emb_model(blocks, x_dict)
        pos_score = self.pred(positive_graph, h_dict)
        neg_score = self.pred(negative_graph, h_dict)
        return pos_score, neg_score


class LinkPrediction_fullbatch(nn.Module):
    def __init__(self, emb_model, emb_dim, target_ntype_u, target_ntype_v):
        super(LinkPrediction_fullbatch, self).__init__()
        self.emb_model = emb_model
        self.r = nn.Parameter(th.Tensor(emb_dim))
        nn.init.ones_(self.r)
        self.target_ntype_u = target_ntype_u
        self.target_ntype_v = target_ntype_v

    def forward(self, positive_edges, negative_edges, g, x_dict):
        h_dict = self.emb_model(g, x_dict)
        h_u = h_dict[self.target_ntype_u]
        h_v = h_dict[self.target_ntype_v]

        pos_score = th.sum(h_u[positive_edges[:, 0]] * self.r * h_v[positive_edges[:, 1]], dim=-1)
        neg_score = th.sum(h_u[negative_edges[:, 0]] * self.r * h_v[negative_edges[:, 1]], dim=-1)

        return pos_score, neg_score
