import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
from utils import accuracy, set_seed, select_mask, load_dataset, evaluate_metric
from model import H2GCN
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--without-relu', action="store_true", help="disable relu for all H2GCN layer")
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--k', type=int, default=2, help='number of embedding rounds')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay value')
parser.add_argument('--hidden', type=int, default=64, help='embedding output dim')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--dataset', default='cora', help='dateset name')
parser.add_argument('--gpu', type=int, default=-1, help='gpu id to use while training, set -1 to use cpu')
parser.add_argument('--split-id', type=int, default=0, help='the data split to use')
args = parser.parse_args()


def train():
    model.train()
    optimizer.zero_grad()
    pred_0, pred_1, pred_c0, pred_c1, lbn_loss = model(adj, features, t, idx_train)
    true_0, true_1 = y[torch.where(t[idx_train] == 0)[0]], y[torch.where(t[idx_train] == 1)[0]]
    pred_loss = F.mse_loss(pred_0, true_0.reshape(-1,1)) + F.mse_loss(pred_1, true_1.reshape(-1,1))
    loss_train = pred_loss + lbn_loss * PARAM['ibn_reg']

    loss_train.backward()
    optimizer.step()
    # evaluation
    eATE_train, ePEHE_train = evaluate_metric(pred_0, pred_1, pred_c0, pred_c1)
    return loss_train.item(), eATE_train, ePEHE_train

def validate():
    model.eval()
    with torch.no_grad():
        pred_0, pred_1, pred_c0, pred_c1, lbn_loss = model(adj, features, t, idx_val)
        true_0, true_1 = y[torch.where(t[idx_val] == 0)[0]], y[torch.where(t[idx_val] == 1)[0]]

        loss_val = F.mse_loss(pred_0, true_0.reshape(-1,1)) + F.mse_loss(pred_1, true_1.reshape(-1,1)) + lbn_loss * PARAM['ibn_reg']
        # evaluation
        eATE_val, ePEHE_val = evaluate_metric(pred_0, pred_1, pred_c0, pred_c1)
        return loss_val.item(), eATE_val, ePEHE_val

def test():
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        pred_0, pred_1, pred_c0, pred_c1, lbn_loss = model(adj, features, t, idx_test)
        true_0, true_1 = y[torch.where(t[idx_test] == 0)[0]], y[torch.where(t[idx_test] == 1)[0]]
        loss_test = F.mse_loss(pred_0, true_0.reshape(-1,1)) + F.mse_loss(pred_1, true_1.reshape(-1,1)) + lbn_loss * PARAM['ibn_reg']
        # evaluation
        eATE_test, ePEHE_test = evaluate_metric(pred_0, pred_1, pred_c0, pred_c1)
        return loss_test.item(), eATE_test, ePEHE_test

def main():
    begin_time = time.time()
    tolerate = 0
    best_loss = 1000
    for epoch in range(args.epochs):
        loss_train, eATE_train, ePEHE_train = train()
        loss_validate, eATE_val, ePEHE_val = validate()
        if (epoch + 1) % 1 == 0:
            print(
                'Epoch {:03d}'.format(epoch + 1),
                '|| train',
                'loss : {:.3f}'.format(loss_train),
                ', eATE : {:.3f}'.format(eATE_train),
                ', ePEHE : {:.3f}'.format(ePEHE_train),
                '|| val',
                'loss : {:.3f}'.format(loss_validate),
                ', eATE : {:.3f}'.format(eATE_val),
                ', ePEHE : {:.3f}'.format(ePEHE_val),
            )
        if loss_validate < best_loss:
            best_loss = loss_validate
            torch.save(model.state_dict(), checkpoint_path)
            tolerate = 0
        else:
            tolerate += 1
        if tolerate == args.patience:
            break
    print("Train cost : {:.2f}s".format(time.time() - begin_time))
    _, eATE_test, ePEHE_test = test()
    print("Test eATE : {:.3f} ".format(eATE_test), "Test ePEHE : {:.3f} ".format(ePEHE_test))


if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device('cpu' if args.gpu == -1 else "cuda:%s" % args.gpu)

    graph = 'A_0_3_0.1_100_N'
    i=11

    PARAM = {
        'feature': 'data/synthetic_dt/' + graph + '/gendt_'+str(i)+'.csv',
        'network': 'data/synthetic_dt/' + graph + '/net_' + str(i) + '.csv',
        'has_feature': False,
        'num_nodes': 100,
        'ibn_reg': 10,
    }

    t = torch.tensor(np.array(pd.read_csv(PARAM['feature'])['T'])).to(device)
    y = torch.tensor(np.array(pd.read_csv(PARAM['feature'])['Y'])).to(device)

    # features, labels, feat_dim, class_dim, adj, train_mask, val_mask, test_mask = load_dataset(
    #     args.dataset,
    #     device
    # )

    # features: (num_nodes, feature_dim)
    if PARAM['has_feature'] == True:
        features = np.array(pd.read_csv(PARAM['feature'])[PARAM['feature_col']])
        if len(features.shape) == 1:
            features = features[:, np.newaxis]
    else:
        features = np.eye(PARAM['num_nodes'], PARAM['num_nodes'])

    # prepare torch_geometric data
    A = np.array(pd.read_csv(PARAM['network']))
    A, D = sp.coo_matrix(A), sp.coo_matrix(A - np.eye(PARAM['num_nodes']))
    edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    adj = torch.sparse_coo_tensor(indices=edge_index, values=values, size=[PARAM['num_nodes'], PARAM['num_nodes']]).to(device)


    checkpoint_path = utils.root + '/checkpoint/%s.pt' % args.dataset
    idx_train, idx_val, idx_test = range(0, 60), range(60, 80), range(80, 100)
    if not os.path.exists(utils.root + '/checkpoint'):
        os.makedirs(utils.root + '/checkpoint')
    model = H2GCN(
        feat_dim=features.shape[-1],
        hidden_dim=args.hidden,
        use_relu=not args.without_relu
    ).to(device)
    optimizer = optim.Adam([{'params': model.params, 'weight_decay': args.wd}], lr=args.lr)
    main()