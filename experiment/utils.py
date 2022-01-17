import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch as th
import dgl


def classification_scores(y_true, logits):
    y_pred = np.argmax(logits, axis=1)
    y_score = softmax(logits, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score, multi_class="ovr")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    return accuracy, auroc, macro_f1, micro_f1


def link_prediction_scores(pos_scores, neg_scores):
    y_true = np.concatenate((np.ones_like(pos_scores, dtype=np.int64), np.zeros_like(neg_scores, dtype=np.int64)),
                            axis=0)
    y_score = np.concatenate((pos_scores, neg_scores), axis=0)

    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return auroc, ap


class EarlyStopping:
    """Early stops the training if validation score/loss doesn't improve after a given patience."""

    def __init__(
            self,
            patience=10,
            delta=0,
            mode="score",
            save_path="checkpoint.pt",
            verbose=False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score/loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation score/loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.Inf
        self.early_stop = False

    def __call__(self, quantity, model):
        if self.mode == "score":
            score = quantity
        elif self.mode == "loss":
            score = -quantity
        else:
            raise NotImplementedError

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(quantity, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, quantity, model):
        """Saves model when validation score/loss improves."""
        if self.verbose:
            if self.mode == "score":
                print(
                    f"Validation score increased ({self.best_score:.6f} --> {quantity:.6f}).  Saving model ..."
                )
            elif self.mode == "loss":
                print(
                    f"Validation loss decreased ({-self.best_score:.6f} --> {quantity:.6f}).  Saving model ..."
                )
            else:
                raise NotImplementedError
        th.save(model.state_dict(), self.save_path)


class FixedNegSampler(dgl.dataloading.negative_sampler._BaseNegativeSampler):
    def __init__(self, eid2neg_uv):
        self.eid2neg_uv = eid2neg_uv

    def _generate(self, g, eids, canonical_etype):
        edges = th.tensor([self.eid2neg_uv[eid] for eid in eids.cpu().tolist()], dtype=g.idtype, device=g.device)
        return edges[:, 0], edges[:, 1]
