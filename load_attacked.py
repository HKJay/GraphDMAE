import torch
import torch_geometric.transforms as T
import numpy as np
from deeprobust.graph.data import Dataset as DeepRobust_Dataset
from deeprobust.graph.data import PrePtbDataset as DeepRobust_PrePtbDataset
from torch_geometric.data import Data
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')
parser.add_argument('--attack', type=str, default='mettack',  choices=['mettack', 'nettack'], help='attack method')


args = parser.parse_args()

def mask_to_index(mask):
    index = torch.where(mask == True)[0].cuda()
    return index

def get_dataset(args, sparse=True):
    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform = None

    dataset, perturbed_data= get_adv_dataset(args.dataset, transform=transform, ptb_rate=args.ptb_rate, attack=args.attack)

    data = dataset.data

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test'] = mask_to_index(data.test_mask)

    return dataset, data, split_idx, perturbed_data


def get_adv_dataset(name, transform, ptb_rate, attack):
    dataset = DeepRobust_Dataset(root='./tmp', name=name, setting='nettack', require_mask=True, seed=15)
    dataset.x = torch.FloatTensor(dataset.features.todense())
    dataset.y = torch.LongTensor(dataset.labels)
    dataset.num_classes = dataset.y.max().item() + 1

    if ptb_rate > 0:
        if args.attack == 'mettack':
            perturbed_data = DeepRobust_PrePtbDataset(
                root='./tmp',
                name=name,
                attack_method='meta',
                ptb_rate=ptb_rate)
        else:
            perturbed_data = DeepRobust_PrePtbDataset(
                root='./tmp',
                name=name,
                attack_method='nettack',
                ptb_rate=ptb_rate)
        edge_index = torch.LongTensor(perturbed_data.adj.nonzero())
    else:
        perturbed_data = dataset
        edge_index = torch.LongTensor(dataset.adj.nonzero())
    data = Data(x=dataset.x, edge_index=edge_index, y=dataset.y)

    clean_edge_index = torch.LongTensor(dataset.adj.nonzero())
    clean_data = Data(x=dataset.x, edge_index=clean_edge_index, y=dataset.y)

    data.train_mask = torch.tensor(dataset.train_mask)
    data.val_mask = torch.tensor(dataset.val_mask)
    if attack == 'mettack':
        data.test_mask = torch.tensor(dataset.test_mask)
    else:
        if ptb_rate == 0.0:
            target_data = DeepRobust_PrePtbDataset(
                root='./tmp',
                name=name,
                attack_method='nettack',
                ptb_rate=1.0)
            perturbed_data.target_nodes = target_data.target_nodes
        test_mask = np.zeros(dataset.y.shape[0], dtype=int)
        test_mask[perturbed_data.target_nodes] = 1
        data.test_mask = torch.tensor(test_mask)
        data.idx_test = perturbed_data.target_nodes
        dataset.idx_test = perturbed_data.target_nodes

    dataset.data = data
    dataset.clean_data = clean_data
    dataset.data.clean_adj = dataset.clean_data.edge_index
    return dataset, perturbed_data

def main():
    
    dataset, data, split_idx, perturbed_data = get_dataset(args)
    print("perturbed_data: ", perturbed_data.adj)
    idx_train, idx_val, idx_test = dataset.idx_train, dataset.idx_val, dataset.idx_test
    modified_adj = torch.zeros([data.x.shape[0], data.x.shape[0]]).long()
    modified_adj[data.edge_index[0], data.edge_index[1]] = 1
    # modified_adj = modified_adj.cpu().to_sparse()
    print("modified_adj: ", modified_adj)
    torch.save(modified_adj.cpu().to_sparse(), "./ptb_graphs/%s/%s_%s_%s.pt" % (args.attack, args.attack, args.dataset, args.ptb_rate))
    np.save("./ptb_graphs/%s/%s_%s_%s_idx_train" % (args.attack, args.attack, args.dataset, args.ptb_rate), idx_train)
    np.save("./ptb_graphs/%s/%s_%s_%s_idx_val" % (args.attack, args.attack, args.dataset, args.ptb_rate), idx_val)
    np.save("./ptb_graphs/%s/%s_%s_%s_idx_test" % (args.attack, args.attack, args.dataset, args.ptb_rate), idx_test)
    
    # Export features and labels files
    import scipy.sparse as sp
    # Convert features to sparse matrix and save as npz
    features_sparse = sp.csr_matrix(dataset.x.cpu().numpy())
    # sp.save_npz("./ptb_graphs/%s_features.npz" % (args.dataset), features_sparse)
    # Save labels as npy
    labels = dataset.y.cpu().numpy()
    # np.save("./ptb_graphs/%s_labels.npy" % (args.dataset), labels)


if __name__ == '__main__':
    main()