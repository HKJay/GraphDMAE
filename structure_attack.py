import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack, DICE, Random
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, AmazonPyg, Pyg2Dpr
import argparse
import time
import scipy.sparse as sp



parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer','cora_ml', 'blogcatalog', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train', 'DICE', 'random'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)
data = Dataset(root='./tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

elif 'DICE' in args.model:
    model = DICE()

elif 'random' in args.model:
    model = Random()

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    # gcn.fit(features, adj, labels, idx_train) # train without model picking
    gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    if 'DICE' in args.model:
        # Convert tensor adj back to scipy sparse matrix for DICE attack
        if torch.is_tensor(adj):
            adj_scipy = to_scipy(adj.cpu())
        else:
            adj_scipy = adj
        model.attack(adj_scipy, labels, n_perturbations=perturbations)
        modified_adj = model.modified_adj

        # Convert scipy sparse matrix to tensor for saving
        if not torch.is_tensor(modified_adj):
            modified_adj = torch.FloatTensor(modified_adj.toarray())
    
        torch.save(modified_adj.cpu().to_sparse(), "./ptb_graphs/%s/%s_%s_%s.pt" % ('DICE', 'DICE', args.dataset, args.ptb_rate))
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_train" % ('DICE', 'DICE', args.dataset, args.ptb_rate), idx_train)
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_val" % ('DICE', 'DICE', args.dataset, args.ptb_rate), idx_val)
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_test" % ('DICE', 'DICE', args.dataset, args.ptb_rate), idx_test)
    
    elif 'random' in args.model:
        if torch.is_tensor(adj):
            adj_scipy = to_scipy(adj.cpu())
        else:
            adj_scipy = adj
        model.attack(adj_scipy, n_perturbations=perturbations)
        modified_adj = model.modified_adj

        # Convert scipy sparse matrix to tensor for saving
        if not torch.is_tensor(modified_adj):
            modified_adj = torch.FloatTensor(modified_adj.toarray())
    
        torch.save(modified_adj.cpu().to_sparse(), "./ptb_graphs/%s/%s_%s_%s.pt" % ('random', 'random', args.dataset, args.ptb_rate))
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_train" % ('random', 'random', args.dataset, args.ptb_rate), idx_train)
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_val" % ('random', 'random', args.dataset, args.ptb_rate), idx_val)
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_test" % ('random', 'random', args.dataset, args.ptb_rate), idx_test)
    
    else:
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True)
        print('=== testing GCN on original(clean) graph ===')
        # test(adj)
        modified_adj = model.modified_adj
        #print("modified_adj: ",modified_adj)
        # test(modified_adj)
    
        # Save features and labels
        sp.save_npz("./ptb_graphs/%s_features.npz" % args.dataset, sp.csr_matrix(features.cpu().numpy()))
        np.save("./ptb_graphs/%s_labels.npy" % args.dataset, labels.cpu().numpy())
    
        torch.save(modified_adj.cpu().to_sparse(), "./ptb_graphs/%s/%s_%s_%s.pt" % ('mettack', 'mettack', args.dataset, args.ptb_rate))
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_train" % ('mettack', 'mettack', args.dataset, args.ptb_rate), idx_train)
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_val" % ('mettack', 'mettack', args.dataset, args.ptb_rate), idx_val)
        np.save("./ptb_graphs/%s/%s_%s_%s_idx_test" % ('mettack', 'mettack', args.dataset, args.ptb_rate), idx_test)


if __name__ == '__main__':
    main()
