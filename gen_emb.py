import argparse
import pickle
from pathlib import Path

import dgl
import torch as th
import numpy as np

from model.MECCH import MECCH, khopMECCH
from model.baselines.RGCN import RGCN
from model.baselines.HGT import HGT
from model.baselines.HAN import HAN, HAN_lp
from model.modules import LinkPrediction_minibatch, LinkPrediction_fullbatch
from utils import metapath2str, get_metapath_g, get_khop_g, load_base_config, load_model_config, load_data_nc, load_data_lp


def main_nc(args):
    # load data
    g, in_dim_dict, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = load_data_nc(args.dataset)
    print("Loaded data from dataset: {}".format(args.dataset))

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        args.device = th.device('cuda', args.gpu)
    else:
        args.device = th.device('cpu')

    # create model + model-specific data preprocessing
    if args.model == "MECCH":
        if args.ablation:
            g = get_khop_g(g, args)
            model = khopMECCH(
                g,
                in_dim_dict,
                args.hidden_dim,
                out_dim,
                args.n_layers,
                dropout=args.dropout,
                residual=args.residual,
                layer_norm=args.layer_norm
            )
        else:
            g, selected_metapaths = get_metapath_g(g, args)
            n_heads_list = [args.n_heads] * args.n_layers
            model = MECCH(
                g,
                selected_metapaths,
                in_dim_dict,
                args.hidden_dim,
                out_dim,
                args.n_layers,
                n_heads_list,
                dropout=args.dropout,
                context_encoder=args.context_encoder,
                use_v=args.use_v,
                metapath_fusion=args.metapath_fusion,
                residual=args.residual,
                layer_norm=args.layer_norm
            )
        minibatch_flag = True
    elif args.model == "RGCN":
        assert args.n_layers >= 2
        model = RGCN(
            g,
            in_dim_dict,
            args.hidden_dim,
            out_dim,
            num_bases=-1,
            num_hidden_layers=args.n_layers - 2,
            dropout=args.dropout,
            use_self_loop=args.use_self_loop
        )
        minibatch_flag = False
    elif args.model == "HGT":
        model = HGT(
            g,
            in_dim_dict,
            args.hidden_dim,
            out_dim,
            args.n_layers,
            args.n_heads
        )
        minibatch_flag = False
    elif args.model == "HAN":
        # assume the target node type has attributes
        assert args.hidden_dim % args.n_heads == 0
        target_ntype = list(g.ndata["y"].keys())[0]
        n_heads_list = [args.n_heads] * args.n_layers
        model = HAN(
            args.metapaths,
            target_ntype,
            in_dim_dict[target_ntype],
            args.hidden_dim // args.n_heads,
            out_dim,
            num_heads=n_heads_list,
            dropout=args.dropout
        )
        minibatch_flag = False
    else:
        raise NotImplementedError

    state_dict = th.load(str(Path(args.save) / 'checkpoint.pt'))
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    g = g.to(args.device)

    if minibatch_flag:
        nid_dict = {ntype: g.nodes(ntype).to(args.device) for ntype in g.ntypes}
        # Use GPU-based neighborhood sampling if possible
        num_workers = 4 if args.device.type == "CPU" else 0
        if args.n_neighbor_samples <= 0:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler([{
                etype: args.n_neighbor_samples for etype in g.canonical_etypes}] * args.n_layers)
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            nid_dict,
            sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            device=args.device
        )
        with th.no_grad():
            h_dict = {ntype: [] for ntype in nid_dict}
            for input_nodes, output_nodes, blocks in dataloader:
                input_features = blocks[0].srcdata["x"]
                _, h_dict_temp = model.get_embs(blocks, input_features)
                if not h_dict:
                    h_dict = {k: [v] for k, v in h_dict_temp.items()}
                else:
                    for k, v in h_dict_temp.items():
                        h_dict[k].append(v)
            h_dict = {k: th.cat(v, dim=0).cpu().numpy() for k, v in h_dict.items()}
    else:
        with th.no_grad():
            _, h_dict = model.get_embs(g, g.ndata['x'])
            h_dict = {k: v.cpu().numpy() for k, v in h_dict.items()}

    # save embeddings
    np.savez(Path(args.save) / 'embeddings.npz', **h_dict)


def main_lp(args):
    # load data
    (g_train, g_val, g_test), in_dim_dict, (train_eid_dict, val_eid_dict, test_eid_dict), (val_neg_uv, test_neg_uv) = load_data_lp(args.dataset)

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        args.device = th.device('cuda', args.gpu)
    else:
        args.device = th.device('cpu')

    target_etype = list(train_eid_dict.keys())[0]
    # create model + model-specific preprocessing
    if args.model == 'MECCH':
        if args.ablation:
            # Note: here we assume there is only one edge type between users and items
            train_eid_dict = {(g_train.to_canonical_etype(k)[0], '1-hop', g_train.to_canonical_etype(k)[2]): v for
                              k, v in train_eid_dict.items()}
            val_eid_dict = {(g_val.to_canonical_etype(k)[0], '1-hop', g_val.to_canonical_etype(k)[2]): v for k, v
                            in val_eid_dict.items()}
            test_eid_dict = {(g_test.to_canonical_etype(k)[0], '1-hop', g_test.to_canonical_etype(k)[2]): v for k, v
                             in test_eid_dict.items()}
            target_etype = list(train_eid_dict.keys())[0]

            g_train = get_khop_g(g_train, args)
            g_val = get_khop_g(g_val, args)
            g_test = get_khop_g(g_test, args)
            model = khopMECCH(
                g_train,
                in_dim_dict,
                args.hidden_dim,
                args.hidden_dim,
                args.n_layers,
                dropout=args.dropout,
                residual=args.residual,
                layer_norm=args.layer_norm
            )
        else:
            train_eid_dict = {metapath2str([g_train.to_canonical_etype(k)]): v for k, v in train_eid_dict.items()}
            val_eid_dict = {metapath2str([g_val.to_canonical_etype(k)]): v for k, v in val_eid_dict.items()}
            test_eid_dict = {metapath2str([g_test.to_canonical_etype(k)]): v for k, v in test_eid_dict.items()}
            target_etype = list(train_eid_dict.keys())[0]

            # cache metapath_g
            load_path = Path('./data') / args.dataset / 'metapath_g-max_mp={}'.format(args.max_mp_length)
            if load_path.is_dir():
                g_list, _ = dgl.load_graphs(str(load_path / 'graph.bin'))
                g_train, g_val, g_test = g_list
                with open(load_path / 'selected_metapaths.pkl', 'rb') as in_file:
                    selected_metapaths = pickle.load(in_file)
            else:
                g_train, _ = get_metapath_g(g_train, args)
                g_val, _ = get_metapath_g(g_val, args)
                g_test, selected_metapaths = get_metapath_g(g_test, args)
                load_path.mkdir()
                dgl.save_graphs(str(load_path / 'graph.bin'), [g_train, g_val, g_test])
                with open(load_path / 'selected_metapaths.pkl', 'wb') as out_file:
                    pickle.dump(selected_metapaths, out_file)

            n_heads_list = [args.n_heads] * args.n_layers
            model = MECCH(
                g_train,
                selected_metapaths,
                in_dim_dict,
                args.hidden_dim,
                args.hidden_dim,
                args.n_layers,
                n_heads_list,
                dropout=args.dropout,
                context_encoder=args.context_encoder,
                use_v=args.use_v,
                metapath_fusion=args.metapath_fusion,
                residual=args.residual,
                layer_norm=args.layer_norm
            )
        model_lp = LinkPrediction_minibatch(model, args.hidden_dim, target_etype)
        minibatch_flag = True
    elif args.model == 'RGCN':
        assert args.n_layers >= 2
        model = RGCN(
            g_train,
            in_dim_dict,
            args.hidden_dim,
            args.hidden_dim,
            num_bases=-1,
            num_hidden_layers=args.n_layers - 2,
            dropout=args.dropout,
            use_self_loop=args.use_self_loop
        )
        if hasattr(args, 'batch_size'):
            model_lp = LinkPrediction_minibatch(model, args.hidden_dim, target_etype)
            minibatch_flag = True
        else:
            srctype, _, dsttype = g_train.to_canonical_etype(target_etype)
            model_lp = LinkPrediction_fullbatch(model, args.hidden_dim, srctype, dsttype)
            minibatch_flag = False
    elif args.model == 'HGT':
        model = HGT(
            g_train,
            in_dim_dict,
            args.hidden_dim,
            args.hidden_dim,
            args.n_layers,
            args.n_heads
        )
        if hasattr(args, 'batch_size'):
            model_lp = LinkPrediction_minibatch(model, args.hidden_dim, target_etype)
            minibatch_flag = True
        else:
            srctype, _, dsttype = g_train.to_canonical_etype(target_etype)
            model_lp = LinkPrediction_fullbatch(model, args.hidden_dim, srctype, dsttype)
            minibatch_flag = False
    elif args.model == 'HAN':
        # assume the target node type has attributes
        # Note: this HAN version from DGL conducts full-batch training with online metapath_reachable_graph,
        #       preprocessing needed for the PubMed dataset
        assert args.hidden_dim % args.n_heads == 0
        n_heads_list = [args.n_heads] * args.n_layers
        model_lp = HAN_lp(
            g_train,
            args.metapaths_u,
            args.metapaths_u[0][0][0],
            -1,
            args.metapaths_v,
            args.metapaths_v[0][0][0],
            -1,
            args.hidden_dim // args.n_heads,
            args.hidden_dim,
            num_heads=n_heads_list,
            dropout=args.dropout
        )
        minibatch_flag = False
    else:
        raise NotImplementedError

    state_dict = th.load(str(Path(args.save) / 'checkpoint.pt'))
    model_lp.load_state_dict(state_dict)
    model_lp.to(args.device)
    model_lp.eval()
    g_test = g_test.to(args.device)

    if args.model == 'HAN':
        # should we use g_val or g_test as the input graph?
        with th.no_grad():
            # set initial node embeddings
            if hasattr(model_lp, 'feats_u'):
                x_dict_u = {model_lp.target_ntype_u: model_lp.feats_u}
            else:
                x_dict_u = {model_lp.target_ntype_u: g_test.ndata['x'][model_lp.target_ntype_u]}
            if hasattr(model_lp, 'feats_v'):
                x_dict_v = {model_lp.target_ntype_v: model_lp.feats_v}
            else:
                x_dict_v = {model_lp.target_ntype_v: g_test.ndata['x'][model_lp.target_ntype_v]}

            h_u = model_lp.model_u(g_test, x_dict_u)[model_lp.target_ntype_u]
            h_v = model_lp.model_v(g_test, x_dict_v)[model_lp.target_ntype_v]
            h_dict = {model_lp.target_ntype_u: h_u.cpu().numpy(), model_lp.target_ntype_v: h_v.cpu().numpy()}
    else:
        if minibatch_flag:
            nid_dict = {ntype: g_test.nodes(ntype).to(args.device) for ntype in g_test.ntypes}
            # Use GPU-based neighborhood sampling if possible
            num_workers = 4 if args.device.type == "CPU" else 0
            if args.n_neighbor_samples <= 0:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
            else:
                sampler = dgl.dataloading.MultiLayerNeighborSampler([{
                    etype: args.n_neighbor_samples for etype in g_test.canonical_etypes}] * args.n_layers)
            dataloader = dgl.dataloading.NodeDataLoader(
                g_test,
                nid_dict,
                sampler,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                device=args.device
            )
            with th.no_grad():
                h_dict = {ntype: [] for ntype in nid_dict}
                for input_nodes, output_nodes, blocks in dataloader:
                    input_features = blocks[0].srcdata["x"]
                    h_dict_temp = model_lp.emb_model(blocks, input_features)
                    if not h_dict:
                        h_dict = {k: [v] for k, v in h_dict_temp.items()}
                    else:
                        for k, v in h_dict_temp.items():
                            h_dict[k].append(v)
                h_dict = {k: th.cat(v, dim=0).cpu().numpy() for k, v in h_dict.items()}
        else:
            with th.no_grad():
                h_dict = model_lp.emb_model(g_test, g_test.ndata['x'])
                h_dict = {k: v.cpu().numpy() for k, v in h_dict.items()}

    # save embeddings
    np.savez(Path(args.save) / 'embeddings.npz', **h_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("My HGNNs")
    parser.add_argument('--model', '-m', type=str, required=True, help='name of model')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='name of dataset')
    parser.add_argument('--task', '-t', type=str, default='node_classification', help='type of task')
    parser.add_argument("--gpu", '-g', type=int, default=-1, help="which gpu to use, specify -1 to use CPU")
    parser.add_argument('--save', '-s', type=str, required=True, help='which save dir to use')

    args = parser.parse_args()
    configs = load_base_config()
    configs.update(load_model_config(Path(args.save) / '{}.json'.format(args.model), args.dataset))
    configs.update(vars(args))
    args = argparse.Namespace(**configs)
    print(args)

    if args.task == 'node_classification':
        main_nc(args)
    elif args.task == 'link_prediction':
        main_lp(args)
    else:
        raise NotImplementedError
