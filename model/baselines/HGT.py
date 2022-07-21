import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

from model.modules import HeteroEmbedLayer, HeteroLinearLayer


def softmax_among_all(hetgraph, weight_name, norm_by="dst"):
    """Normalize edge weights by softmax across all neighbors.

    Parameters
    -------------
    hetgraph : DGLGraph
        The input heterogeneous graph.
    weight_name : str
        The name of the unnormalized edge weights.
    """
    # Convert to homogeneous graph; DGL will copy the specified data to the new graph.
    g = dgl.to_homogeneous(hetgraph, edata=[weight_name])
    # Call DGL's edge softmax
    g.edata[weight_name] = edge_softmax(g, g.edata[weight_name], norm_by=norm_by)
    # Convert it back; DGL again copies the data back to a heterogeneous storage.
    hetg2 = dgl.to_heterogeneous(g, hetgraph.ntypes, hetgraph.etypes)
    # Assign the normalized weights to the original graph
    for etype in hetg2.canonical_etypes:
        hetgraph.edges[etype].data[weight_name] = hetg2.edges[etype].data[weight_name]


class HGTLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            node_dict,
            edge_dict,
            n_heads,
            dropout=0.2,
            use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype][:sub_graph.num_dst_nodes(dsttype)]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                sub_graph.edata["t"] = sub_graph.edata["t"] * relation_pri / self.sqrt_dk

            softmax_among_all(G, "t", norm_by="dst")

            G.multi_update_all(
                {
                    etype: (fn.u_mul_e("v_%d" % e_id, "t", "m"), fn.sum("m", "t"))
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="sum",
            )

            new_h = {}
            for ntype in G.dsttypes:
                """
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                if G.num_dst_nodes(ntype) > 0:
                    n_id = node_dict[ntype]
                    alpha = torch.sigmoid(self.skip[n_id])
                    t = G.dstnodes[ntype].data["t"].view(-1, self.out_dim)
                    trans_out = self.drop(self.a_linears[n_id](t))
                    trans_out = trans_out * alpha + h[ntype][:G.num_dst_nodes(ntype)] * (1 - alpha)
                    if self.use_norm:
                        new_h[ntype] = self.norms[n_id](trans_out)
                    else:
                        new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, G, in_dim_dict, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        self.node_dict = {}
        self.edge_dict = {}
        for i, ntype in enumerate(G.ntypes):
            self.node_dict[ntype] = i
        for i, etype in enumerate(G.etypes):
            self.edge_dict[etype] = i
        self.gcs = nn.ModuleList()
        self.in_dim_dict = in_dim_dict
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers

        # input projection
        n_nodes_dict = {
            ntype: G.num_nodes(ntype) for ntype in G.ntypes if in_dim_dict[ntype] < 0
        }
        self.embed_layer = HeteroEmbedLayer(n_nodes_dict, n_hid)
        self.linear_layer = HeteroLinearLayer(in_dim_dict, n_hid)

        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(
                    n_hid,
                    n_hid,
                    self.node_dict,
                    self.edge_dict,
                    n_heads,
                    use_norm=use_norm,
                )
            )
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, x_dict):
        h_dict = {}
        if isinstance(G, list):
            # minibatch
            nids_dict = {
                ntype: nids
                for ntype, nids in G[0].srcdata[dgl.NID].items()
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict
            for ntype in h_dict:
                h_dict[ntype] = F.gelu(h_dict[ntype])

            for layer, block in zip(self.gcs, G):
                h_dict = layer(block, h_dict)
        else:
            # full batch
            nids_dict = {
                ntype: G.nodes(ntype)
                for ntype in G.ntypes
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict
            for ntype in h_dict:
                h_dict[ntype] = F.gelu(h_dict[ntype])

            for i in range(self.n_layers):
                h_dict = self.gcs[i](G, h_dict)
        return {ntype: self.out(h) for ntype, h in h_dict.items()}

    def get_embs(self, G, x_dict):
        h_dict = {}
        if isinstance(G, list):
            # minibatch
            nids_dict = {
                ntype: nids
                for ntype, nids in G[0].srcdata[dgl.NID].items()
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict
            for ntype in h_dict:
                h_dict[ntype] = F.gelu(h_dict[ntype])

            for layer, block in zip(self.gcs, G):
                h_dict = layer(block, h_dict)
        else:
            # full batch
            nids_dict = {
                ntype: G.nodes(ntype)
                for ntype in G.ntypes
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict
            for ntype in h_dict:
                h_dict[ntype] = F.gelu(h_dict[ntype])

            for i in range(self.n_layers):
                h_dict = self.gcs[i](G, h_dict)
        return {ntype: self.out(h) for ntype, h in h_dict.items()}, h_dict
