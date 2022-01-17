from collections import OrderedDict

import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl

import experiment.utils as utils


def node_classification_minibatch(model, g, train_nid_dict, val_nid_dict, test_nid_dict, dir_path, args):
    model.to(args.device)
    g = g.to(args.device)
    train_nid_dict = {k: v.to(args.device) for k, v in train_nid_dict.items()}
    val_nid_dict = {k: v.to(args.device) for k, v in val_nid_dict.items()}
    test_nid_dict = {k: v.to(args.device) for k, v in test_nid_dict.items()}

    assert len(g.ndata["y"].keys()) == 1
    target_ntype = list(g.ndata["y"].keys())[0]

    # Use GPU-based neighborhood sampling if possible
    num_workers = 4 if args.device.type == "CPU" else 0
    if args.n_neighbor_samples <= 0:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    else:
        sampler = dgl.dataloading.MultiLayerNeighborSampler([{
            etype: args.n_neighbor_samples for etype in g.canonical_etypes}] * args.n_layers)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        device=args.device,
    )
    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        val_nid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        device=args.device,
    )
    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_nid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        device=args.device,
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
            for iteration, (input_nodes, output_nodes, blocks) in enumerate(tq):
                input_features = blocks[0].srcdata["x"]
                output_labels = blocks[-1].dstdata["y"]

                logits_dict = model(blocks, input_features)
                logp = F.log_softmax(logits_dict[target_ntype], dim=-1)
                train_loss = F.nll_loss(logp, output_labels[target_ntype])

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
            logits_list = []
            y_true_list = []
            for iteration, (input_nodes, output_nodes, blocks) in enumerate(tq):
                input_features = blocks[0].srcdata["x"]
                output_labels = blocks[-1].dstdata["y"]

                logits_dict = model(blocks, input_features)
                logp = F.log_softmax(logits_dict[target_ntype], dim=-1)
                val_loss += F.nll_loss(logp, output_labels[target_ntype], reduction="sum")

                logits_list.append(logits_dict[target_ntype].cpu().numpy())
                y_true_list.append(output_labels[target_ntype].cpu().numpy())

            val_loss = val_loss / val_nid_dict[target_ntype].numel()
            logits = np.concatenate(logits_list, axis=0)
            y_true = np.concatenate(y_true_list, axis=0)

            val_acc, val_auroc, val_macro_f1, val_micro_f1 = utils.classification_scores(
                y_true, logits
            )

            # print validation info
            print(
                "Epoch {:05d} | Macro-F1 {:.4f} | Micro-F1 {:.4f} | Val_Loss {:.4f}".format(
                    epoch, val_macro_f1, val_micro_f1, val_loss.item()
                )
            )

            # early stopping
            if args.early_stopping_mode == "score":
                quantity = (val_macro_f1 + val_micro_f1) / 2
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
        logits_list = []
        y_true_list = []
        for iteration, (input_nodes, output_nodes, blocks) in enumerate(tq):
            input_features = blocks[0].srcdata["x"]
            output_labels = blocks[-1].dstdata["y"]

            logits_dict = model(blocks, input_features)

            logits_list.append(logits_dict[target_ntype].cpu().numpy())
            y_true_list.append(output_labels[target_ntype].cpu().numpy())

        logits = np.concatenate(logits_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)
        test_acc, test_auroc, test_macro_f1, test_micro_f1 = utils.classification_scores(
            y_true, logits
        )

        # print testing info
        print("Testing Evaluation Metrics")
        print("Macro-F1: {:.4f}".format(test_macro_f1))
        print("Micro-F1: {:.4f}".format(test_micro_f1))
        # save evaluation results
        with dir_path.joinpath("result.txt").open("w") as f:
            f.write("Macro-F1: {:.4f}\n".format(test_macro_f1))
            f.write("Micro-F1: {:.4f}\n".format(test_micro_f1))
    return test_macro_f1, test_micro_f1


def node_classification_fullbatch(model, g, train_nid_dict, val_nid_dict, test_nid_dict, dir_path, args):
    model.to(args.device)
    g = g.to(args.device)
    train_nid_dict = {k: v.to(args.device) for k, v in train_nid_dict.items()}
    val_nid_dict = {k: v.to(args.device) for k, v in val_nid_dict.items()}
    test_nid_dict = {k: v.to(args.device) for k, v in test_nid_dict.items()}

    assert len(g.ndata["y"].keys()) == 1
    target_ntype = list(g.ndata["y"].keys())[0]

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

            logits_dict = model(g, g.ndata['x'])
            logits = logits_dict[target_ntype][train_nid_dict[target_ntype]]
            y_true = g.ndata['y'][target_ntype][train_nid_dict[target_ntype]]
            logp = F.log_softmax(logits, dim=-1)
            train_loss = F.nll_loss(logp, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # evaluation metrics
            train_acc, train_auroc, train_macro_f1, train_micro_f1 = utils.classification_scores(
                y_true.detach().cpu().numpy(), logits.detach().cpu().numpy()
            )

            # set training print info
            print_info = OrderedDict()
            print_info["train_loss"] = "{:.03f}".format(train_loss.item())
            print_info["train_macro_f1"] = "{:.4f}".format(train_macro_f1)
            print_info["train_micro_f1"] = "{:.4f}".format(train_micro_f1)

            # validation
            model.eval()
            with th.no_grad():
                logits_dict = model(g, g.ndata['x'])
                logits = logits_dict[target_ntype][val_nid_dict[target_ntype]]
                y_true = g.ndata['y'][target_ntype][val_nid_dict[target_ntype]]
                logp = F.log_softmax(logits, dim=-1)
                val_loss = F.nll_loss(logp, y_true)

                # evaluation metrics
                val_acc, val_auroc, val_macro_f1, val_micro_f1 = utils.classification_scores(
                    y_true.cpu().numpy(), logits.cpu().numpy()
                )

                # set validation print info
                print_info["val_loss"] = "{:.03f}".format(val_loss.item())
                print_info["val_macro_f1"] = "{:.4f}".format(val_macro_f1)
                print_info["val_micro_f1"] = "{:.4f}".format(val_micro_f1)

                # print training and validation info
                tq.set_postfix(print_info, refresh=False)

                # early stopping
                if args.early_stopping_mode == "score":
                    quantity = (val_macro_f1 + val_micro_f1) / 2
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
        logits_dict = model(g, g.ndata['x'])
        logits = logits_dict[target_ntype][test_nid_dict[target_ntype]]
        y_true = g.ndata['y'][target_ntype][test_nid_dict[target_ntype]]

        # evaluation metrics
        test_acc, test_auroc, test_macro_f1, test_micro_f1 = utils.classification_scores(
            y_true.cpu().numpy(), logits.cpu().numpy()
        )

        # print testing info
        print("Testing Evaluation Metrics")
        print("Macro-F1: {:.4f}".format(test_macro_f1))
        print("Micro-F1: {:.4f}".format(test_micro_f1))
        # save evaluation results
        with dir_path.joinpath("result.txt").open("w") as f:
            f.write("Macro-F1: {:.4f}\n".format(test_macro_f1))
            f.write("Micro-F1: {:.4f}\n".format(test_micro_f1))
    return test_macro_f1, test_micro_f1
