"""RGCN Example Implementation from DGL"""
import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model.modules import HeteroEmbedLayer, HeteroLinearLayer


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
            self,
            in_feat,
            out_feat,
            rel_names,
            num_bases,
            *,
            weight=True,
            bias=True,
            activation=None,
            self_loop=False,
            dropout=0.0
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    th.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(th.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGCN(nn.Module):
    def __init__(
            self,
            g,
            in_dim_dict,
            h_dim,
            out_dim,
            num_bases,
            num_hidden_layers=1,
            dropout=0,
            use_self_loop=False,
    ):
        super(RGCN, self).__init__()

        self.g = g
        self.in_dim_dict = in_dim_dict
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # input projection
        n_nodes_dict = {
            ntype: g.num_nodes(ntype) for ntype in g.ntypes if in_dim_dict[ntype] < 0
        }
        self.embed_layer = HeteroEmbedLayer(n_nodes_dict, h_dim)
        self.linear_layer = HeteroLinearLayer(in_dim_dict, h_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.h_dim,
                self.rel_names,
                self.num_bases,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout,
                weight=False,
            )
        )
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        # h2o
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.out_dim,
                self.rel_names,
                self.num_bases,
                activation=None,
                self_loop=self.use_self_loop,
            )
        )

    def forward(self, g=None, x_dict=None):
        if isinstance(g, list):
            # minibatch forward
            nids_dict = {
                ntype: nids
                for ntype, nids in g[0].srcdata[dgl.NID].items()
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict

            for layer, block in zip(self.layers, g):
                h_dict = layer(block, h_dict)
        else:
            # full graph forward
            nids_dict = {
                ntype: g.nodes(ntype)
                for ntype in g.ntypes
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict

            for layer in self.layers:
                h_dict = layer(g, h_dict)

        return h_dict

    def get_embs(self, g=None, x_dict=None):
        if isinstance(g, list):
            # minibatch forward
            nids_dict = {
                ntype: nids
                for ntype, nids in g[0].srcdata[dgl.NID].items()
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict

            for layer, block in zip(self.layers, g):
                embs_dict = h_dict
                h_dict = layer(block, h_dict)
        else:
            # full graph forward
            nids_dict = {
                ntype: g.nodes(ntype)
                for ntype in g.ntypes
                if self.in_dim_dict[ntype] < 0
            }
            h_embed_dict = self.embed_layer(nids_dict)
            h_linear_dict = self.linear_layer(x_dict)
            h_dict = h_embed_dict | h_linear_dict

            for layer in self.layers:
                embs_dict = h_dict
                h_dict = layer(g, h_dict)

        return h_dict, embs_dict
