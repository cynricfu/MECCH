import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from model.modules import HeteroEmbedLayer, HeteroLinearLayer
from model.utils import sub_metapaths, get_src_ntypes


class MetapathContextEncoder(nn.Module):
    def __init__(self, in_dim, encoder_type="mean", use_v=False, n_heads=8):
        assert in_dim % n_heads == 0
        super(MetapathContextEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.use_v = use_v
        self.n_heads = n_heads
        self.d_k = in_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        if encoder_type == "mean":
            pass
        elif encoder_type == "attention":
            self.k_linear = nn.Linear(in_dim, in_dim, False)
            if use_v:
                self.v_linear = nn.Linear(in_dim, in_dim, False)
            self.q_linear = nn.Linear(in_dim, in_dim, False)
        else:
            raise NotImplementedError

    def forward(self, block, h_dict, metapath_str):
        mp_list = sub_metapaths(metapath_str)
        src_ntypes = get_src_ntypes(metapath_str)
        _, _, dst_ntype = block.to_canonical_etype(metapath_str)

        with block.local_scope():
            if self.encoder_type == "mean":
                funcs = {}
                num_neigh = 0
                for mp in mp_list:
                    funcs[mp] = (fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                    num_neigh += block.in_degrees(etype=mp)
                block.multi_update_all(funcs, "sum")
                block.dstnodes[dst_ntype].data["h_dst_out"] = (block.dstnodes[dst_ntype].data["h_neigh"] +
                                                               block.dstnodes[dst_ntype].data["h_dst"]) / th.unsqueeze(
                    num_neigh + 1, dim=-1)
            elif self.encoder_type == "attention":
                # K, V projections for source nodes
                for ntype in src_ntypes:
                    block.srcnodes[ntype].data['k'] = self.k_linear(
                        block.srcdata["h_src"][ntype]).view(-1, self.n_heads, self.d_k)
                    if self.use_v:
                        block.srcnodes[ntype].data['v'] = self.v_linear(
                            block.srcdata["h_src"][ntype]).view(-1, self.n_heads, self.d_k)
                    else:
                        block.srcnodes[ntype].data['v'] = block.srcdata["h_src"][ntype].view(-1, self.n_heads, self.d_k)
                # K, V, Q projections for destination nodes
                dst_k = self.k_linear(
                    block.dstdata["h_dst"][dst_ntype]).view(-1, self.n_heads, self.d_k)
                if self.use_v:
                    dst_v = self.v_linear(
                        block.dstdata["h_dst"][dst_ntype]).view(-1, self.n_heads, self.d_k)
                else:
                    dst_v = block.dstdata["h_dst"][dst_ntype].view(-1, self.n_heads, self.d_k)
                block.dstnodes[dst_ntype].data['q'] = self.q_linear(
                    block.dstdata["h_dst"][dst_ntype]).view(-1, self.n_heads, self.d_k)

                # compute dot product of k and q for destination nodes, for each head
                # also divide by square root of per-head dim
                dst_t = th.sum(dst_k * block.dstnodes[dst_ntype].data['q'], dim=-1, keepdim=True) / self.sqrt_dk
                # compute dot product of k and q for all edges, for each head
                # also divide by square root of per-head dim
                for mp in mp_list:
                    block.apply_edges(fn.u_dot_v('k', 'q', 't'), etype=mp)
                    block.edges[mp].data['t'] = block.edges[mp].data['t'] / self.sqrt_dk

                # edge_softmax for all edges
                # DGL do not support edge_softmax for all edges for heterograph currently
                # 1) select etypes of interest
                sub_hetero_g = dgl.edge_type_subgraph(block, etypes=mp_list)
                # 2) convert to homogeneous graph
                sub_homo_g = dgl.to_homogeneous(sub_hetero_g, edata=['t'])
                # 3) add self loop
                offset_node = [0] + [sub_hetero_g.num_nodes(sub_hetero_g.ntypes[i]) for i in
                                     range(sub_hetero_g.get_ntype_id(dst_ntype))]
                offset_node = np.sum(offset_node)
                offset_edge = sub_homo_g.num_edges()
                u = sub_homo_g.nodes()[offset_node:offset_node + block.num_dst_nodes(dst_ntype)]
                sub_homo_g.add_edges(u, u, data={'t': dst_t})
                # 4) perform edge_softmax
                sub_homo_g.edata['a'] = edge_softmax(sub_homo_g, sub_homo_g.edata['t'], norm_by='dst')
                dst_a = sub_homo_g.edata['a'][-block.num_dst_nodes(dst_ntype):]
                # 5) remove self loop and convert back to heterograph
                sub_homo_g.remove_edges(offset_edge + th.arange(block.num_dst_nodes(dst_ntype)).to(block.device))
                sub_hetero_g2 = dgl.to_heterogeneous(sub_homo_g, sub_hetero_g.ntypes, sub_hetero_g.etypes)
                # 6) copy data
                for etype in sub_hetero_g2.canonical_etypes:
                    block.edges[etype].data['a'] = sub_hetero_g2.edges[etype].data['a']

                # aggregate neighbors' v multiplied by attention scores
                funcs = {mp: (fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'h_neigh')) for mp in mp_list}
                block.multi_update_all(funcs, "sum")

                # consider self loop, aggregate destination nodes
                if self.use_v:
                    block.dstnodes[dst_ntype].data['h_dst_out'] = F.relu(
                        (block.dstnodes[dst_ntype].data['h_neigh'] + dst_a * dst_v).view(-1, self.n_heads * self.d_k))
                else:
                    block.dstnodes[dst_ntype].data['h_dst_out'] = (
                            block.dstnodes[dst_ntype].data['h_neigh'] + dst_a * dst_v).view(-1, self.n_heads * self.d_k)
            else:
                raise NotImplementedError

            return block.dstnodes[dst_ntype].data["h_dst_out"]


class MetapathFusion(nn.Module):
    def __init__(self, n_metapaths, in_dim, out_dim, fusion_type="conv"):
        super(MetapathFusion, self).__init__()

        self.n_metapaths = n_metapaths
        self.fusion_type = fusion_type

        if fusion_type == "mean":
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "weight":
            self.weight = nn.Parameter(th.full((n_metapaths,), 1 / n_metapaths, dtype=th.float32))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "conv":
            self.conv = nn.Parameter(th.full((n_metapaths, in_dim), 1 / n_metapaths, dtype=th.float32))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "cat":
            self.linear = nn.Linear(n_metapaths * in_dim, out_dim)
        else:
            raise NotImplementedError

    def forward(self, h_list):
        if self.fusion_type == "mean":
            fused = th.mean(th.stack(h_list), dim=0)
        elif self.fusion_type == "weight":
            fused = th.sum(th.stack(h_list) * self.weight[:, None, None], dim=0)
        elif self.fusion_type == "conv":
            fused = th.sum(th.stack(h_list).transpose(0, 1) * self.conv, dim=1)
        elif self.fusion_type == "cat":
            fused = th.hstack(h_list)
        else:
            raise NotImplementedError
        return self.linear(fused), fused


class MECCHLayer(nn.Module):
    def __init__(
            self,
            metapaths_dict,
            in_dim,
            out_dim,
            n_heads=8,
            dropout=0.5,
            context_encoder="mean",
            use_v=False,
            metapath_fusion="conv",
            residual=False,
            layer_norm=False,
            activation=None
    ):
        super(MECCHLayer, self).__init__()

        self.metapaths_dict = metapaths_dict
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        if residual:
            self.alpha = nn.ParameterDict()
            self.residual = nn.ModuleDict()
            for ntype in metapaths_dict:
                self.alpha[ntype] = nn.Parameter(th.tensor(0.))
                if in_dim == out_dim:
                    self.residual[ntype] = nn.Identity()
                else:
                    self.residual[ntype] = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual = None
        if layer_norm:
            self.layer_norm = nn.ModuleDict()
            for ntype in metapaths_dict:
                self.layer_norm[ntype] = nn.LayerNorm(out_dim)
        else:
            self.layer_norm = None
        self.activation = activation

        self.context_encoders = nn.ModuleDict()
        for ntype in metapaths_dict:
            for metapath_str in metapaths_dict[ntype]:
                self.context_encoders[metapath_str] = MetapathContextEncoder(in_dim, context_encoder, use_v, n_heads)

        # Metapath fusion
        self.metapath_fuse = nn.ModuleDict()
        for ntype in metapaths_dict:
            self.metapath_fuse[ntype] = MetapathFusion(len(metapaths_dict[ntype]), in_dim, out_dim, metapath_fusion)

    def forward(self, block, h_dict):
        with block.local_scope():
            for ntype in block.srctypes:
                if block.num_src_nodes(ntype) > 0:
                    block.srcnodes[ntype].data["h_src"] = h_dict[ntype]
                    block.dstnodes[ntype].data["h_dst"] = h_dict[ntype][:block.num_dst_nodes(ntype)]

            out_h_dict = {}
            out_embs_dict = {}
            for ntype in block.dsttypes:
                if block.num_dst_nodes(ntype) > 0:
                    metapath_outs = []
                    for metapath_str in self.metapaths_dict[ntype]:
                        metapath_outs.append(self.context_encoders[metapath_str](block, h_dict, metapath_str))
                    out_h_dict[ntype], out_embs_dict[ntype] = self.metapath_fuse[ntype](metapath_outs)

            for ntype in out_h_dict:
                if self.residual is not None:
                    alpha = th.sigmoid(self.alpha[ntype])
                    out_h_dict[ntype] = out_h_dict[ntype] * alpha + self.residual[ntype](
                        h_dict[ntype][: block.num_dst_nodes(ntype)]) * (1 - alpha)
                if self.layer_norm is not None:
                    out_h_dict[ntype] = self.layer_norm[ntype](out_h_dict[ntype])
                if self.activation is not None:
                    out_h_dict[ntype] = self.activation(out_h_dict[ntype])
                out_h_dict[ntype] = self.dropout(out_h_dict[ntype])

            return out_h_dict, out_embs_dict


class MECCH(nn.Module):
    def __init__(
            self,
            g,
            metapaths_dict,
            in_dim_dict,
            hidden_dim,
            out_dim,
            n_layers,
            n_heads_list,
            dropout=0.5,
            context_encoder="mean",
            use_v=False,
            metapath_fusion="conv",
            residual=False,
            layer_norm=True
    ):
        super(MECCH, self).__init__()

        self.in_dim_dict = in_dim_dict
        self.n_layers = n_layers

        n_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes if in_dim_dict[ntype] < 0}
        self.embed_layer = HeteroEmbedLayer(n_nodes_dict, hidden_dim)
        self.linear_layer = HeteroLinearLayer(in_dim_dict, hidden_dim)

        self.MECCH_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.MECCH_layers.append(
                MECCHLayer(
                    metapaths_dict,
                    hidden_dim,
                    hidden_dim,
                    n_heads_list[i],
                    dropout=dropout,
                    context_encoder=context_encoder,
                    use_v=use_v,
                    metapath_fusion=metapath_fusion,
                    residual=residual,
                    layer_norm=layer_norm,
                    activation=F.relu,
                )
            )
        self.MECCH_layers.append(
            MECCHLayer(
                metapaths_dict,
                hidden_dim,
                out_dim,
                n_heads_list[-1],
                dropout=0.0,
                context_encoder=context_encoder,
                use_v=use_v,
                metapath_fusion=metapath_fusion,
                residual=residual,
                layer_norm=False,
                activation=None,
            )
        )

    def forward(self, blocks, x_dict):
        nids_dict = {ntype: nids for ntype, nids in blocks[0].srcdata[dgl.NID].items() if self.in_dim_dict[ntype] < 0}

        # ntype-specific embedding/projection
        h_embed_dict = self.embed_layer(nids_dict)
        h_linear_dict = self.linear_layer(x_dict)
        h_dict = h_embed_dict | h_linear_dict

        for block, layer in zip(blocks, self.MECCH_layers):
            h_dict, _ = layer(block, h_dict)

        return h_dict

    # used to get node representations for node classification tasks
    # (i.e., the node vectors just before applying the final linear layer of the last MECCH layer)
    def get_embs(self, blocks, x_dict):
        nids_dict = {ntype: nids for ntype, nids in blocks[0].srcdata[dgl.NID].items() if self.in_dim_dict[ntype] < 0}

        # ntype-specific embedding/projection
        h_embed_dict = self.embed_layer(nids_dict)
        h_linear_dict = self.linear_layer(x_dict)
        h_dict = h_embed_dict | h_linear_dict

        for block, layer in zip(blocks, self.MECCH_layers):
            h_dict, embs_dict = layer(block, h_dict)

        return h_dict, embs_dict


class khopMECCHLayer(nn.Module):
    def __init__(
            self,
            ntypes,
            in_dim,
            out_dim,
            dropout=0.5,
            residual=False,
            layer_norm=False,
            activation=None
    ):
        super(khopMECCHLayer, self).__init__()

        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim)

        if residual:
            self.alpha = nn.ParameterDict()
            self.residual = nn.ModuleDict()
            for ntype in ntypes:
                self.alpha[ntype] = nn.Parameter(th.tensor(0.))
                if in_dim == out_dim:
                    self.residual[ntype] = nn.Identity()
                else:
                    self.residual[ntype] = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual = None
        if layer_norm:
            self.layer_norm = nn.ModuleDict()
            for ntype in ntypes:
                self.layer_norm[ntype] = nn.LayerNorm(out_dim)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, block, h_dict):
        with block.local_scope():
            for ntype in block.srctypes:
                if block.num_src_nodes(ntype) > 0:
                    block.srcnodes[ntype].data["h_src"] = h_dict[ntype]
                    block.dstnodes[ntype].data["h_dst"] = h_dict[ntype][:block.num_dst_nodes(ntype)]
            funcs = {}
            num_neigh = {ntype: 0 for ntype in block.dsttypes}
            for etype in block.canonical_etypes:
                if block.num_edges(etype=etype):
                    _, _, ntype = etype
                    funcs[etype] = (fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                    num_neigh[ntype] = num_neigh[ntype] + block.in_degrees(etype=etype)
            block.multi_update_all(funcs, "sum")

            out_h_dict = {}
            for ntype in block.dsttypes:
                if block.num_dst_nodes(ntype) > 0:
                    out_h_dict[ntype] = (block.dstnodes[ntype].data["h_neigh"] + block.dstnodes[ntype].data[
                        "h_dst"]) / th.unsqueeze(num_neigh[ntype] + 1, dim=-1)
                    out_h_dict[ntype] = self.linear(out_h_dict[ntype])
                    if self.residual is not None:
                        alpha = th.sigmoid(self.alpha[ntype])
                        out_h_dict[ntype] = out_h_dict[ntype] * alpha + self.residual[ntype](
                            h_dict[ntype][: block.num_dst_nodes(ntype)]) * (1 - alpha)
                    if self.layer_norm is not None:
                        out_h_dict[ntype] = self.layer_norm[ntype](out_h_dict[ntype])
                    if self.activation is not None:
                        out_h_dict[ntype] = self.activation(out_h_dict[ntype])
                    out_h_dict[ntype] = self.dropout(out_h_dict[ntype])

            return out_h_dict


class khopMECCH(nn.Module):
    def __init__(
            self,
            g,
            in_dim_dict,
            hidden_dim,
            out_dim,
            n_layers,
            dropout=0.5,
            residual=False,
            layer_norm=True
    ):
        super(khopMECCH, self).__init__()

        self.in_dim_dict = in_dim_dict
        self.n_layers = n_layers

        n_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes if in_dim_dict[ntype] < 0}
        self.embed_layer = HeteroEmbedLayer(n_nodes_dict, hidden_dim)
        self.linear_layer = HeteroLinearLayer(in_dim_dict, hidden_dim)

        self.khopMECCH_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.khopMECCH_layers.append(
                khopMECCHLayer(
                    g.ntypes,
                    hidden_dim,
                    hidden_dim,
                    dropout=dropout,
                    residual=residual,
                    layer_norm=layer_norm,
                    activation=F.relu,
                )
            )
        self.khopMECCH_layers.append(
            khopMECCHLayer(
                g.ntypes,
                hidden_dim,
                out_dim,
                dropout=0.0,
                residual=residual,
                layer_norm=False,
                activation=None,
            )
        )

    def forward(self, blocks, x_dict):
        nids_dict = {ntype: nids for ntype, nids in blocks[0].srcdata[dgl.NID].items() if self.in_dim_dict[ntype] < 0}

        # ntype-specific embedding/projection
        h_embed_dict = self.embed_layer(nids_dict)
        h_linear_dict = self.linear_layer(x_dict)
        h_dict = h_embed_dict | h_linear_dict

        for block, layer in zip(blocks, self.khopMECCH_layers):
            h_dict = layer(block, h_dict)

        return h_dict
