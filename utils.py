import pickle
from collections import defaultdict
from pathlib import Path
import shutil

import dgl
import numpy as np
import torch as th


def get_all_metapaths(g, min_length=1, max_length=4):
    etype_dict = {}
    for src, e, dst in g.canonical_etypes:
        if src in etype_dict:
            etype_dict[src].append((e, dst))
        else:
            etype_dict[src] = [(e, dst)]

    metapath_dict = {src: {i + 1: [] for i in range(max_length)} for src in etype_dict}
    for src in etype_dict:
        metapath_dict[src][1].extend([(src, e[0], e[1]) for e in etype_dict[src]])
        for i in range(1, max_length):
            metapath_dict[src][i + 1].extend(
                [mp + (e[0], e[1]) for mp in metapath_dict[src][i] for e in etype_dict[mp[-1]]])

    return {src: {length: metapath_dict[src][length] for length in metapath_dict[src] if length >= min_length} for src
            in metapath_dict}


def metapath_dict2list(metapath_dict):
    return [
        [mp[i - 1: i + 2] for i in range(1, len(mp), 2)]
        for src in metapath_dict
        for length in metapath_dict[src]
        for mp in metapath_dict[src][length]
    ]


join_token = "=>"


def metapath2str(metapath):
    metapath_str = "mp:" + join_token.join(metapath[0])
    for src_ntype, etype, dst_ntype in metapath[1:]:
        metapath_str += join_token + etype + join_token + dst_ntype
    return metapath_str


# Assume metapath_g already contains all the nodes
def add_metapath_connection(g, metapath, metapath_g, add_reverse=False):
    if metapath_g is not None:
        graph_data = {e: metapath_g.edges(etype=e) for e in metapath_g.canonical_etypes}
    else:
        graph_data = {}
    num_nodes = {n: g.num_nodes(n) for n in g.ntypes}

    new_g = dgl.metapath_reachable_graph(g, metapath)
    src_nodes = new_g.edges()[0]
    dst_nodes = new_g.edges()[1]
    canonical_etype = (new_g.srctypes[0], metapath2str(metapath), new_g.dsttypes[0])
    graph_data[canonical_etype] = (src_nodes, dst_nodes)
    if add_reverse:
        src_nodes, dst_nodes = dst_nodes, src_nodes
        canonical_etype = (new_g.dsttypes[0], "mp:" + "rev:" + join_token.join(metapath), new_g.srctypes[0])
        graph_data[canonical_etype] = (src_nodes, dst_nodes)

    return dgl.heterograph(graph_data, num_nodes)


def select_metapaths(all_metapaths_list, length=4):
    # select only max-length metapath
    selected_metapaths = defaultdict(list)
    for mp in all_metapaths_list:
        if len(mp) == length:
            selected_metapaths[mp[-1][-1]].append(metapath2str(mp))
    return dict(selected_metapaths)


def load_data_nc(dataset_name, prefix="./data"):
    if dataset_name == "imdb-gtn":
        # movie*, actor, director
        glist, _ = dgl.load_graphs(str(Path(prefix, dataset_name, "graph.bin")))
        g = glist[0]

        x = g.ndata.pop('h')
        y = g.ndata.pop('label')
        train_mask = g.ndata.pop('train_mask')
        val_mask = g.ndata.pop('valid_mask')
        test_mask = g.ndata.pop('test_mask')

        g = g.long()
        g.nodes['movie'].data['x'] = x['movie'].float()
        g.nodes['actor'].data['x'] = x['actor'].float()
        g.nodes['director'].data['x'] = x['director'].float()
        g.nodes['movie'].data['y'] = y['movie'].long()

        in_dim_dict = {
            "movie": x["movie"].shape[1],
            "actor": x["actor"].shape[1],
            "director": x["director"].shape[1],
        }
        out_dim = y["movie"].max().item() + 1

        train_nid_dict = {
            "movie": train_mask["movie"].nonzero().flatten().long()
        }
        val_nid_dict = {
            "movie": val_mask["movie"].nonzero().flatten().long()
        }
        test_nid_dict = {
            "movie": test_mask["movie"].nonzero().flatten().long()
        }
    elif dataset_name == 'acm-gtn':
        # paper*, author, subject
        glist, _ = dgl.load_graphs(str(Path(prefix, dataset_name, "graph.bin")))
        g = glist[0]

        x = g.ndata.pop('h')
        y = g.ndata.pop('label')
        train_mask = g.ndata.pop('train_mask')
        val_mask = g.ndata.pop('valid_mask')
        test_mask = g.ndata.pop('test_mask')
        g.ndata.pop('pspap_m2v_emb')
        g.ndata.pop('psp_m2v_emb')
        g.ndata.pop('pap_m2v_emb')

        g = g.long()
        g.nodes['paper'].data['x'] = x['paper'].float()
        g.nodes['author'].data['x'] = x['author'].float()
        g.nodes['subject'].data['x'] = x['subject'].float()
        g.nodes['paper'].data['y'] = y['paper'].long()

        in_dim_dict = {
            "paper": x["paper"].shape[1],
            "author": x["author"].shape[1],
            "subject": x["subject"].shape[1],
        }
        out_dim = y["paper"].max().item() + 1

        train_nid_dict = {
            "paper": train_mask["paper"].nonzero().flatten().long()
        }
        val_nid_dict = {
            "paper": val_mask["paper"].nonzero().flatten().long()
        }
        test_nid_dict = {
            "paper": test_mask["paper"].nonzero().flatten().long()
        }
    elif dataset_name == 'dblp-gtn':
        # paper, author*, conference
        dir_path = Path(prefix, dataset_name)
        edges = pickle.load(dir_path.joinpath("edges.pkl").open("rb"))
        labels = pickle.load(dir_path.joinpath("labels.pkl").open("rb"))
        node_features = pickle.load(
            dir_path.joinpath("node_features.pkl").open("rb"))

        num_nodes = edges[0].shape[0]
        node_type = np.zeros(num_nodes, dtype=int)
        node_type[:] = -1

        assert len(edges) == 4
        assert len(edges[0].nonzero()) == 2
        node_type[edges[0].nonzero()[0]] = 0
        node_type[edges[0].nonzero()[1]] = 1
        node_type[edges[1].nonzero()[0]] = 1
        node_type[edges[1].nonzero()[1]] = 0
        node_type[edges[2].nonzero()[0]] = 0
        node_type[edges[2].nonzero()[1]] = 2
        node_type[edges[3].nonzero()[0]] = 2
        node_type[edges[3].nonzero()[1]] = 0
        assert (node_type == -1).sum() == 0

        data_dict = {
            ('paper', 'paper-author', 'author'): edges[0][node_type == 0, :][:, node_type == 1].nonzero(),
            ('author', 'author-paper', 'paper'): edges[1][node_type == 1, :][:, node_type == 0].nonzero(),
            ('paper', 'paper-conference', 'conference'): edges[2][node_type == 0, :][:, node_type == 2].nonzero(),
            ('conference', 'conference-paper', 'paper'): edges[3][node_type == 2, :][:, node_type == 0].nonzero()
        }
        num_nodes_dict = {
            'paper': (node_type == 0).sum(),
            'author': (node_type == 1).sum(),
            'conference': (node_type == 2).sum()
        }
        g = dgl.heterograph(data_dict, num_nodes_dict, idtype=th.int64)

        train_nid_dict = {
            'author': th.from_numpy(np.array(labels[0])[:, 0]).long()
        }
        val_nid_dict = {
            'author': th.from_numpy(np.array(labels[1])[:, 0]).long()
        }
        test_nid_dict = {
            'author': th.from_numpy(np.array(labels[2])[:, 0]).long()
        }

        g.nodes['paper'].data['x'] = th.from_numpy(
            node_features[node_type == 0]).float()
        g.nodes['author'].data['x'] = th.from_numpy(
            node_features[node_type == 1]).float()
        g.nodes['conference'].data['x'] = th.from_numpy(
            node_features[node_type == 2]).float()

        y = np.zeros((g.num_nodes('author')), dtype=int)
        y[train_nid_dict['author'].numpy()] = np.array(labels[0])[:, 1]
        y[val_nid_dict['author'].numpy()] = np.array(labels[1])[:, 1]
        y[test_nid_dict['author'].numpy()] = np.array(labels[2])[:, 1]
        g.nodes['author'].data['y'] = th.from_numpy(y).long()

        in_dim_dict = {
            'paper': g.nodes['paper'].data['x'].shape[1],
            'author': g.nodes['author'].data['x'].shape[1],
            'conference': g.nodes['conference'].data['x'].shape[1]
        }
        out_dim = g.nodes['author'].data['y'].max().item() + 1
    else:
        raise NotImplementedError

    return g, in_dim_dict, out_dim, train_nid_dict, val_nid_dict, test_nid_dict


def load_data_lp(dataset_name, prefix="./data"):
    if dataset_name == 'lastfm':
        load_path = Path(prefix, dataset_name)
        g_list, _ = dgl.load_graphs(str(load_path / 'graph.bin'))
        g_train, g_val, g_test = g_list
        train_val_test_idx = np.load(str(load_path / 'train_val_test_idx.npz'))
        train_eid_dict = {'user-artist': th.tensor(train_val_test_idx['train_idx'])}
        val_eid_dict = {'user-artist': th.tensor(train_val_test_idx['val_idx'])}
        test_eid_dict = {'user-artist': th.tensor(train_val_test_idx['test_idx'])}
        val_neg_uv = th.tensor(np.load(str(load_path / 'val_neg_user_artist.npy')))
        test_neg_uv = th.tensor(np.load(str(load_path / 'test_neg_user_artist.npy')))
        in_dim_dict = {ntype: -1 for ntype in g_test.ntypes}
    else:
        raise NotImplementedError

    g_train = g_train.long()
    g_val = g_val.long()
    g_test = g_test.long()
    train_eid_dict = {k: v.long() for k, v in train_eid_dict.items()}
    val_eid_dict = {k: v.long() for k, v in val_eid_dict.items()}
    test_eid_dict = {k: v.long() for k, v in test_eid_dict.items()}
    val_neg_uv = val_neg_uv.long()
    test_neg_uv = test_neg_uv.long()

    return (g_train, g_val, g_test), in_dim_dict, (train_eid_dict, val_eid_dict, test_eid_dict), (
        val_neg_uv, test_neg_uv)


def get_save_path(args, prefix="./saves"):
    dir_path = Path(prefix, args.model, args.dataset)
    dir_path.mkdir(parents=True, exist_ok=True)
    old_saves = [int(str(x.name)) for x in dir_path.iterdir() if x.is_dir() and str(x.name).isdigit()]
    if len(old_saves) == 0:
        save_num = 1
    else:
        save_num = max(old_saves) + 1
    dir_path = dir_path / str(save_num)
    dir_path.mkdir()

    # copy config files to the save dir
    shutil.copy("./configs/base.json", dir_path)
    shutil.copy(args.config, dir_path)

    return dir_path
