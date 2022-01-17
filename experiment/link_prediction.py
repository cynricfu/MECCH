from collections import OrderedDict

import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl

import experiment.utils as utils


def link_prediction_minibatch(model, g_train, g_val, g_test, train_eid_dict, val_eid_dict, test_eid_dict, val_neg_uv,
                              test_neg_uv, dir_path, args):
    model.to(args.device)

    target_etype = list(train_eid_dict.keys())[0]

    # GPU-based sampling results in an error, so only use CPU-based sampling here
    num_workers = 4
    if args.n_neighbor_samples <= 0:
        block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    else:
        block_sampler = dgl.dataloading.MultiLayerNeighborSampler([{
            etype: args.n_neighbor_samples for etype in g_train.canonical_etypes}] * args.n_layers)
    if args.exclude:
        exclude = "reverse_types"
        reverse_etypes = args.reverse_etypes
    else:
        exclude = None
        reverse_etypes = None
    val_eid2neg_uv = {eid: (u, v) for eid, (u, v) in
                      zip(val_eid_dict[target_etype].cpu().tolist(), val_neg_uv.cpu().tolist())}
    test_eid2neg_uv = {eid: (u, v) for eid, (u, v) in
                       zip(test_eid_dict[target_etype].cpu().tolist(), test_neg_uv.cpu().tolist())}
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        g_train,
        {target_etype: g_train.edges(etype=target_etype, form='eid')},
        block_sampler,
        exclude=exclude,
        reverse_etypes=reverse_etypes,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    val_dataloader = dgl.dataloading.EdgeDataLoader(
        g_test,
        val_eid_dict,
        block_sampler,
        g_sampling=g_train,
        negative_sampler=utils.FixedNegSampler(val_eid2neg_uv),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    test_dataloader = dgl.dataloading.EdgeDataLoader(
        g_test,
        test_eid_dict,
        block_sampler,
        g_sampling=g_val,
        negative_sampler=utils.FixedNegSampler(test_eid2neg_uv),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    early_stopping = utils.EarlyStopping(
        patience=args.patience, mode=args.early_stopping_mode, verbose=True, save_path=str(dir_path / "checkpoint.pt")
    )

    for epoch in range(args.n_epochs):
        # training
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for iteration, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq):
                blocks = [b.to(args.device) for b in blocks]
                positive_graph = positive_graph.to(args.device)
                negative_graph = negative_graph.to(args.device)

                input_features = blocks[0].srcdata["x"]
                pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
                train_loss = F.binary_cross_entropy_with_logits(pos_score, th.ones_like(
                    pos_score)) + F.binary_cross_entropy_with_logits(neg_score, th.zeros_like(neg_score))

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # print training info
                tq.set_postfix(
                    {"loss": "{:.03f}".format(train_loss.item())}, refresh=False
                )

        # validation
        model.eval()
        with tqdm.tqdm(val_dataloader) as tq, th.no_grad():
            val_loss = 0
            pos_score_list = []
            neg_score_list = []
            for iteration, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq):
                blocks = [b.to(args.device) for b in blocks]
                positive_graph = positive_graph.to(args.device)
                negative_graph = negative_graph.to(args.device)

                input_features = blocks[0].srcdata["x"]
                pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
                val_loss += F.binary_cross_entropy_with_logits(pos_score, th.ones_like(pos_score),
                                                               reduction="sum") + F.binary_cross_entropy_with_logits(
                    neg_score, th.zeros_like(neg_score), reduction="sum")

                pos_score_list.append(pos_score.cpu().numpy())
                neg_score_list.append(neg_score.cpu().numpy())
            val_loss = val_loss / val_eid_dict[target_etype].numel()
            pos_scores = np.concatenate(pos_score_list, axis=0)
            neg_scores = np.concatenate(neg_score_list, axis=0)

            val_auroc, val_ap = utils.link_prediction_scores(pos_scores, neg_scores)

            # print validation info
            print(
                "Epoch {:05d} | AUROC {:.4f} | AP {:.4f} | Val_Loss {:.4f}".format(epoch, val_auroc, val_ap, val_loss))

            # early stopping
            if args.early_stopping_mode == "score":
                quantity = val_auroc
            elif args.early_stopping_mode == "loss":
                quantity = val_loss
            else:
                raise NotImplementedError
            early_stopping(quantity, model)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

    # testing
    model.load_state_dict(th.load(str(dir_path / "checkpoint.pt")))
    model.eval()
    with tqdm.tqdm(test_dataloader) as tq, th.no_grad():
        pos_score_list = []
        neg_score_list = []
        for iteration, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq):
            blocks = [b.to(args.device) for b in blocks]
            positive_graph = positive_graph.to(args.device)
            negative_graph = negative_graph.to(args.device)

            input_features = blocks[0].srcdata["x"]
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)

            pos_score_list.append(pos_score.cpu().numpy())
            neg_score_list.append(neg_score.cpu().numpy())
        pos_scores = np.concatenate(pos_score_list, axis=0)
        neg_scores = np.concatenate(neg_score_list, axis=0)

        test_auroc, test_ap = utils.link_prediction_scores(pos_scores, neg_scores)

        # print testing info
        print("Testing Evaluation Metrics")
        print("AUROC: {:.4f}".format(test_auroc))
        print("AP: {:.4f}".format(test_ap))
        # save evaluation results
        with dir_path.joinpath("result.txt").open("w") as f:
            f.write("AUROC: {:.4f}\n".format(test_auroc))
            f.write("AP: {:.4f}\n".format(test_ap))
    return test_auroc, test_ap


def link_prediction_fullbatch(model, g_train, g_val, g_test, train_eid_dict, val_eid_dict, test_eid_dict, val_neg_uv,
                              test_neg_uv, dir_path, args):
    model.to(args.device)
    g_train = g_train.to(args.device)
    g_val = g_val.to(args.device)
    g_test = g_test.to(args.device)
    train_eid_dict = {k: v.to(args.device) for k, v in train_eid_dict.items()}
    val_eid_dict = {k: v.to(args.device) for k, v in val_eid_dict.items()}
    test_eid_dict = {k: v.to(args.device) for k, v in test_eid_dict.items()}

    target_etype = list(train_eid_dict.keys())[0]
    target_ntype_u, _, target_ntype_v = g_train.to_canonical_etype(target_etype)

    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    early_stopping = utils.EarlyStopping(
        patience=args.patience, mode=args.early_stopping_mode, verbose=True, save_path=str(dir_path / "checkpoint.pt")
    )

    with tqdm.tqdm(range(args.n_epochs)) as tq:
        for epoch in tq:
            # training
            model.train()

            pos_edges = th.stack(g_train.edges(etype=target_etype), dim=1)
            neg_edges = th.clone(pos_edges)
            neg_edges[:, 1] = th.randint(0, g_train.num_nodes(target_ntype_v), (neg_edges.shape[0],))

            pos_score, neg_score = model(pos_edges, neg_edges, g_train, g_train.ndata['x'])
            train_loss = F.binary_cross_entropy_with_logits(pos_score, th.ones_like(
                pos_score)) + F.binary_cross_entropy_with_logits(neg_score, th.zeros_like(neg_score))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # evaluation metrics
            train_auroc, train_ap = utils.link_prediction_scores(pos_score.detach().cpu().numpy(),
                                                                 neg_score.detach().cpu().numpy())

            # set validation print info
            print_info = OrderedDict()
            print_info["train_loss"] = "{:.03f}".format(train_loss.item())
            print_info["train_auroc"] = "{:.4f}".format(train_auroc)
            print_info["train_ap"] = "{:.4f}".format(train_ap)

            # validation
            model.eval()
            with th.no_grad():
                pos_edges = th.stack(g_test.find_edges(val_eid_dict[target_etype], etype=target_etype), dim=1)
                pos_score, neg_score = model(pos_edges, val_neg_uv, g_train, g_train.ndata['x'])
                val_loss = F.binary_cross_entropy_with_logits(pos_score, th.ones_like(
                    pos_score)) + F.binary_cross_entropy_with_logits(neg_score, th.zeros_like(neg_score))

                # evaluation metrics
                val_auroc, val_ap = utils.link_prediction_scores(pos_score.cpu().numpy(), neg_score.cpu().numpy())

                # set validation print info
                print_info["val_loss"] = "{:.03f}".format(val_loss.item())
                print_info["val_auroc"] = "{:.4f}".format(val_auroc)
                print_info["val_ap"] = "{:.4f}".format(val_ap)

                # print training and validation info
                tq.set_postfix(print_info, refresh=False)

                # early stopping
                if args.early_stopping_mode == "score":
                    quantity = val_auroc
                elif args.early_stopping_mode == "loss":
                    quantity = val_loss
                else:
                    raise NotImplementedError
                early_stopping(quantity, model)
                if early_stopping.early_stop:
                    print("Early stopping!")
                    break

    # testing
    model.load_state_dict(th.load(str(dir_path / "checkpoint.pt")))
    model.eval()
    with th.no_grad():
        # forward
        pos_edges = th.stack(g_test.find_edges(test_eid_dict[target_etype], etype=target_etype), dim=1)
        pos_score, neg_score = model(pos_edges, test_neg_uv, g_val, g_val.ndata['x'])

        # evaluation metrics
        test_auroc, test_ap = utils.link_prediction_scores(pos_score.cpu().numpy(), neg_score.cpu().numpy())

        # print testing info
        print("Testing Evaluation Metrics")
        print("AUROC: {:.4f}".format(test_auroc))
        print("AP: {:.4f}".format(test_ap))
        # save evaluation results
        with dir_path.joinpath("result.txt").open("w") as f:
            f.write("AUROC: {:.4f}\n".format(test_auroc))
            f.write("AP: {:.4f}\n".format(test_ap))
    return test_auroc, test_ap
