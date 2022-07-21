import argparse
import pickle
from pathlib import Path

import dgl
import numpy as np
import torch as th

from experiment.node_classification import node_classification_minibatch, node_classification_fullbatch
from experiment.link_prediction import link_prediction_minibatch, link_prediction_fullbatch
from model.MECCH import MECCH, khopMECCH
from model.baselines.RGCN import RGCN
from model.baselines.HGT import HGT
from model.baselines.HAN import HAN, HAN_lp
from model.modules import LinkPrediction_minibatch, LinkPrediction_fullbatch
from utils import metapath2str, get_metapath_g, get_khop_g, load_data_nc, load_data_lp, \
    get_save_path, load_base_config, load_model_config


def main_nc(args):
    dir_path_list = []
    for _ in range(args.repeat):
        dir_path_list.append(get_save_path(args))

    test_macro_f1_list = []
    test_micro_f1_list = []
    for i in range(args.repeat):
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

        if minibatch_flag:
            test_macro_f1, test_micro_f1 = node_classification_minibatch(model, g, train_nid_dict, val_nid_dict,
                                                                         test_nid_dict, dir_path_list[i], args)
        else:
            test_macro_f1, test_micro_f1 = node_classification_fullbatch(model, g, train_nid_dict, val_nid_dict,
                                                                         test_nid_dict, dir_path_list[i], args)
        test_macro_f1_list.append(test_macro_f1)
        test_micro_f1_list.append(test_micro_f1)

    print("--------------------------------")
    if args.repeat > 1:
        print("Macro-F1_MEAN\tMacro-F1_STDEV\tMicro-F1_MEAN\tMicro-F1_STDEV")
        print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(np.mean(test_macro_f1_list), np.std(test_macro_f1_list, ddof=0),
                                                      np.mean(test_micro_f1_list), np.std(test_micro_f1_list, ddof=0)))
    else:
        print("args.repeat <= 1, not calculating the average and the standard deviation of scores")


def main_lp(args):
    dir_path_list = []
    for _ in range(args.repeat):
        dir_path_list.append(get_save_path(args))

    test_auroc_list = []
    test_ap_list = []
    for i in range(args.repeat):
        # load data
        (g_train, g_val, g_test), in_dim_dict, (train_eid_dict, val_eid_dict, test_eid_dict), (
            val_neg_uv, test_neg_uv) = load_data_lp(args.dataset)
        print("Loaded data from dataset: {}".format(args.dataset))

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

        if minibatch_flag:
            test_auroc, test_ap = link_prediction_minibatch(model_lp, g_train, g_val, g_test, train_eid_dict,
                                                            val_eid_dict, test_eid_dict, val_neg_uv, test_neg_uv,
                                                            dir_path_list[i], args)
        else:
            test_auroc, test_ap = link_prediction_fullbatch(model_lp, g_train, g_val, g_test, train_eid_dict,
                                                            val_eid_dict, test_eid_dict, val_neg_uv, test_neg_uv,
                                                            dir_path_list[i], args)
        test_auroc_list.append(test_auroc)
        test_ap_list.append(test_ap)

    print("--------------------------------")
    if args.repeat > 1:
        print("ROC-AUC_MEAN\tROC-AUC_STDEV\tPR-AUC_MEAN\tPR-AUC_STDEV")
        print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(np.mean(test_auroc_list), np.std(test_auroc_list, ddof=0),
                                                      np.mean(test_ap_list), np.std(test_ap_list, ddof=0)))
    else:
        print("args.repeat <= 1, not calculating the average and the standard deviation of scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("My HGNNs")
    parser.add_argument('--model', '-m', type=str, required=True, help='name of model')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='name of dataset')
    parser.add_argument('--task', '-t', type=str, default='node_classification', help='type of task')
    parser.add_argument("--gpu", '-g', type=int, default=-1, help="which gpu to use, specify -1 to use CPU")
    parser.add_argument('--config', '-c', type=str, help='config file for model hyperparameters')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='repeat the training and testing for N times')

    args = parser.parse_args()
    if args.config is None:
        args.config = "./configs/{}.json".format(args.model)

    configs = load_base_config()
    configs.update(load_model_config(args.config, args.dataset))
    configs.update(vars(args))
    args = argparse.Namespace(**configs)
    print(args)

    if args.task == 'node_classification':
        main_nc(args)
    elif args.task == 'link_prediction':
        main_lp(args)
