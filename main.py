import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import time
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GAT
import json
from copy import deepcopy
from models import GraphDMAE, GraphMAETrainer
from utils import cal_degree, get_logger, norm_cosine_similarity, graph_augmentation, adj_filter, idx_to_mask, evaluate, compute_sorted_spectrum, reconstruct_adj_from_spectrum, k_neighbors


parser = argparse.ArgumentParser()

parser.add_argument("--use_config", action='store_true', help='use config file or not')
parser.add_argument("--config", type=str, default="cora_mettack_0.15", help='config file')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')
parser.add_argument('--attack', type=str, default='mettack',  help='attack method')
parser.add_argument('--jt', type=float, default=0.6,  help='jaccard threshold')
parser.add_argument('--cos', type=float, default=0.6,  help='positive cosine similarity threshold')
parser.add_argument('--cos_add', type=float, default=0.85,  help='negative cosine similarity threshold')
parser.add_argument('--DMAE_hidden', type=int, default=256,  help='hidden dimension for DMAE')
parser.add_argument('--DMAE_epochs', type=int, default=300,  help='epochs for DMAE')
parser.add_argument('--recover_threshold', type=float, default=0.6,  help='recover threshold for edge recovery')
parser.add_argument('--lap_threshold', type=float, default=0.4,  help='lap threshold for edge recovery')
parser.add_argument('--L_dim', type=int, default=20,  help='L dimension')
parser.add_argument('--k_l', type=int, default=50,  help='k neighbors for k-nearest neighbors')
parser.add_argument('--GAT_layers', type=int, default=2,  help='number of GAT layers')
parser.add_argument("--hidden_dim", type=int, default=16 ,  help='dimension of hidden layers')
parser.add_argument("--dropout", type=float, default=0.6 ,  help='dropout rate')
parser.add_argument("--log", action='store_true', help='run prepare_data or not')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')

args = parser.parse_args()

if args.use_config:
    config = json.load(open('./config.json'))[args.config]
    args.dataset = config['dataset']
    args.ptb_rate = config['ptb_rate']
    args.attack = config['attack']
    args.jt = config['jt']
    args.cos = config['cos']
    args.cos_add = config['cos_add']
    args.DMAE_hidden = config['DMAE_hidden']
    args.DMAE_epochs = config['DMAE_epochs']
    args.recover_threshold = config['recover_threshold']
    args.L_dim = config['L_dim']
    args.GAT_layers = config['GAT_layers']
    args.hidden_dim = config['hidden_dim']
    args.dropout = config['dropout']
    args.k_l = config['k_l']
    args.lap_threshold = config['lap_threshold']

if args.log:
    logger = get_logger('./log/' + args.attack + '/' + args.dataset + '_' + str(args.ptb_rate) + '.log')
else:
    logger = get_logger('./log/try.log')
    

# Set random seed
if args.seed is not None:
    seed = args.seed
else:
    seed = int(time.time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
logger.info(f'Random seed: {seed}')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    logger.info(f'Using GPU')
else:
    logger.info('Using CPU')

#Loading dataset
dataset = args.dataset
ptb_rate = args.ptb_rate
logger.info(f'Loading dataset {dataset} with pertubation rate {ptb_rate}')
features = torch.tensor(sp.load_npz('./ptb_graphs/%s_features.npz' % (dataset)).toarray())
features = features.to(device)
labels = torch.tensor(np.load('./ptb_graphs/%s_labels.npy' % (dataset))).to(device)
n_nodes = features.shape[0]
n_class = labels.max() + 1
idx_train = np.load('./ptb_graphs/%s/%s_%s_%s_idx_train.npy' % (args.attack, args.attack, args.dataset, args.ptb_rate))
idx_val = np.load('./ptb_graphs/%s/%s_%s_%s_idx_val.npy' % (args.attack, args.attack, args.dataset, args.ptb_rate))
idx_test = np.load('./ptb_graphs/%s/%s_%s_%s_idx_test.npy' % (args.attack, args.attack, args.dataset, args.ptb_rate))
perturbed_adj = torch.load('./ptb_graphs/%s/%s_%s_%s.pt' % (args.attack, args.attack, args.dataset, args.ptb_rate)).to(device)
perturbed_adj = perturbed_adj.indices()
train_mask, val_mask, test_mask = idx_to_mask(idx_train, n_nodes).to(device), idx_to_mask(idx_val, n_nodes).to(device), \
                                  idx_to_mask(idx_test, n_nodes).to(device)
logger.info('train nodes:%d' % train_mask.sum())
logger.info('val nodes:%d' % val_mask.sum())
logger.info('test nodes:%d' % test_mask.sum())

loss = nn.CrossEntropyLoss()
# Train model
def train_MAE(model, trainer, filter_adj, adj, logger, features, clean_lap, lam, recover_threshold, epochs, k, verbose=True):
    logger.info(f"clean_lap shape: {clean_lap.shape}")
    neighbors = k_neighbors(features, k)
    neighbors = torch.tensor(neighbors).to(device)
    for epoch in range(epochs):
        l = trainer.train_epoch(features, clean_lap, filter_adj, neighbors)

        if verbose:
            if epoch % 10 == 0:
                logger.info("Epoch {:05d}/{:05d} | Loss {:.4f}"
                      .format(epoch, epochs, l))
    
    model.eval()
    node_x = model.encode(features, clean_lap, filter_adj)
    lap_corrected = model.lap_correct(node_x, filter_adj)
    lap_corrected = lap_corrected.to(device)
    recover_adj = torch.zeros([n_nodes, n_nodes]).long()
    degree = cal_degree(filter_adj, n_nodes)
    if args.lap_threshold >= 0.0:
        index, edge_weight= reconstruct_adj_from_spectrum(clean_lap+lap_corrected, lam, degree, threshold=args.lap_threshold, device=device)
        similarity_lap = norm_cosine_similarity(node_x[index[0]], node_x[index[1]]).detach()
        recover_edge_lap = index[:, similarity_lap>=recover_threshold]
        recover_adj[recover_edge_lap[0], recover_edge_lap[1]] = 1

    similarity = norm_cosine_similarity(node_x[adj[0]], node_x[adj[1]]).detach()
    recover_edge = adj[:, similarity>=recover_threshold]

    recover_adj = torch.zeros([n_nodes, n_nodes]).long()

    recover_adj[filter_adj[0], filter_adj[1]] = 1
    recover_adj[recover_edge[0], recover_edge[1]] = 1
    recover_adj = recover_adj.nonzero().t()
    recover_adj = graph_augmentation(recover_adj, node_x)
    logger.info(f'Edge recovery: {recover_adj.shape[1]}')
    return recover_adj, node_x.detach()


def train_GAT(model, optim, adj, run, logger, labels, train_mask, val_mask, test_mask, features, epochs, verbose=True):  
    best_loss_val = 9999
    best_acc_val = 0


    for epoch in range(epochs):
        model.train()
        out = model(features, adj)
        l = loss(out[train_mask], labels[train_mask].long())
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, features, adj, labels, val_mask)
        val_loss = loss(out[val_mask], labels[val_mask].long()).item()
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 100 == 0:
                logger.info("Epoch {:05d}/{:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, epochs, l.item(), acc))
    model.load_state_dict(weights)
    acc = evaluate(model, features, adj, labels, test_mask)
    logger.info("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return acc

if __name__ == '__main__':
    DMAE_hidden = args.DMAE_hidden
    MAE_epochs = args.DMAE_epochs
    GAT_epochs = 200

    logger.info(args)

    logger.info("=== Start filter adj ===")
    positive_adj = adj_filter(features, perturbed_adj, args.jt, args.cos, args.cos_add).to(device)

    logger.info("=== Recover adj ===")
    DMAE = GraphDMAE(feature_dim=features.shape[1],
                           hidden_dim=DMAE_hidden,
                           num_encoder_layers=3,
                           num_decoder_layers=2,
                           output_dim=features.shape[1],
                           L_dim=args.L_dim).to(device)
    trainer = GraphMAETrainer(DMAE, device=device)
    lam, clean_lap = compute_sorted_spectrum(n_nodes, positive_adj, args.L_dim)
    lam = lam.to(device)
    clean_lap = clean_lap.to(device)
    recover_adj, node_x = train_MAE(DMAE, trainer, positive_adj, perturbed_adj, logger, features, clean_lap, lam, args.recover_threshold, MAE_epochs, args.k_l, verbose=True)
    recover_adj = recover_adj.to(device)
    node_x = node_x.to(device)

    logger.info("=== Train Classifier ===")
    acc_total = []
    for run in range(10):
        GAT_model = GAT(
            in_channels=node_x.shape[1]+features.shape[1],
            hidden_channels=args.hidden_dim,
            num_layers=args.GAT_layers,
            out_channels=n_class,
            dropout=args.dropout,
            residual=True
        ).to(device)
        optim = torch.optim.Adam(GAT_model.parameters(), lr=2e-3, weight_decay=5e-4)
        acc = train_GAT(GAT_model, optim, recover_adj, run, logger, labels, train_mask, val_mask, test_mask, torch.cat((node_x, features), dim=1), GAT_epochs, verbose=True)
        acc_total.append(acc)
       
    logger.info('Mean Accuracy:%f' % np.mean(acc_total))
    logger.info('Standard Deviation:%f' % np.std(acc_total, ddof=1))