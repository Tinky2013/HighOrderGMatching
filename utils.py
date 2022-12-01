import os
import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor

root = os.path.split(__file__)[0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def evaluate_metric(pred_0, pred_1, pred_c0, pred_c1):
    tau_pred = torch.cat([pred_1, pred_c1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)
    tau_true = torch.ones(tau_pred.shape)
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    return eATE, ePEHE


def load_dataset(name: str, device=None):
    if device is None:
        device = torch.device('cpu')
    name = name.lower()
    if name in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(root=root + "/dataset/Planetoid", name=name)
    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=root + "/dataset/WikipediaNetwork", name=name)
    elif name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=root + "/dataset/WebKB", name=name)
    elif name in ["actor"]:
        dataset = Actor(root=root + "/dataset/Actor")
    else:
        raise "Please implement support for this dataset in function load_dataset()."
    data = dataset[0].to(device)
    x, y = data.x, data.y
    n = len(x)
    edge_index = data.edge_index
    nfeat = data.num_node_features
    nclass = len(torch.unique(y))
    return x, y, nfeat, nclass, eidx_to_sp(n, edge_index), data.train_mask, data.val_mask, data.test_mask


def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)


def select_mask(i: int, train: torch.Tensor, val: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    if train.dim() == 1:
        return train, val, test
    else:
        indices = torch.tensor([i]).to(train.device)
        train_idx = torch.index_select(train, 1, indices).reshape(-1)
        val_idx = torch.index_select(val, 1, indices).reshape(-1)
        test_idx = torch.index_select(test, 1, indices).reshape(-1)
        return train_idx, val_idx, test_idx
