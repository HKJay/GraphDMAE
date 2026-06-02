import os
import numpy as np
import scipy as sp
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric import utils as TGutils
from torch_geometric.utils import degree, to_scipy_sparse_matrix
from deeprobust.graph.data import Dataset, AmazonPyg
import logging

def adj_filter(feature, edge_index, jt, cos, cos_add):
    r"""Get positive example according to cosine similarity and jaccard similarity.
    Theta is the threshold.
    Args:
        feature (Tensor): feature of nodes
        edge_index (Tensor): edge index of nodes
        jt (float): threshold for jaccard similarity
        cos (float): threshold for cosine similarity
        cos_add (float): threshold for cosine similarity addition

    Returns:
        positive example
    """
    n_nodes = feature.shape[0]
    
    adj = torch.zeros([n_nodes, n_nodes], device=feature.device)
    adj[edge_index[0], edge_index[1]] = 1
    
    positive_adj = torch.zeros_like(adj)
    
    cosine_matrix = norm_cosine_similarity(feature, feature, all_pairs=True)
    positive_mask = (cosine_matrix >= cos_add).to(torch.bool)
    positive_adj[positive_mask] = 1
    
    unique_edges = torch.unique(torch.stack([edge_index[0], edge_index[1]]).t(), dim=0)
    
    node_a = unique_edges[:, 0]
    node_b = unique_edges[:, 1]
    
    edge_cosine_similarities = cosine_matrix[node_a, node_b]
    
    adj_sparse = adj.to_sparse()
    
    # Jaccard(A,B) = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    
    degrees = adj.sum(dim=1)
    
    intersection_sizes = (adj @ adj.T)[node_a, node_b]
    
    union_sizes = degrees[node_a] + degrees[node_b] - intersection_sizes
    
    union_sizes = torch.clamp(union_sizes, min=1e-8)
    
    jaccard_similarities = intersection_sizes / union_sizes
    
    cosine_threshold_mask = edge_cosine_similarities >= cos
    jaccard_threshold_mask = jaccard_similarities >= jt
    
    add_edge_mask = cosine_threshold_mask | jaccard_threshold_mask
    
    nodes_a_to_add = node_a[add_edge_mask]
    nodes_b_to_add = node_b[add_edge_mask]
    
    positive_adj[nodes_a_to_add, nodes_b_to_add] = 1
    positive_adj[nodes_b_to_add, nodes_a_to_add] = 1
    
    has_nonzero = (positive_adj != 0).any(dim=1)
    zero_indices = ~has_nonzero
    
    if zero_indices.any():
        supply_adj_cos = cosine_matrix[zero_indices, :].argmax(dim=1)
        positive_adj[zero_indices, supply_adj_cos] = 1
        positive_adj[supply_adj_cos, zero_indices] = 1

    return positive_adj.nonzero().t()

def save_adj_filter(feature, adj, jt_pos, jt_neg, cos_pos, cos_neg, dir, dataset_name):
    r"""Get positive example according to cosine similarity and jaccard similarity, and save it in dir.
        Theta is the threshold.
        Args:
            feature (Tensor): feature of nodes
            adj (Tensor): adjacency matrix of nodes
            jt_pos (float): threshold for positive cosine jaccard similarity
            jt_neg (float): threshold for negative cosine jaccard similarity
            cos_pos (float): threshold for positive cosine similarity
            cos_neg (float): threshold for negative cosine similarity
            dir (str): save directory
            dataset_name (str): dataset name

        Returns:
            positive example
            [positive_adj]
        """

    positive_adj = example_extraction(feature, adj, jt_pos, jt_neg, cos_pos, cos_neg)

    pos_path = os.path.join(dir, '%s_positive_adj_%s_%s_%s_%s_%s.pt' % (dataset_name, jt_pos, jt_neg, cos_pos, cos_neg))
    torch.save(positive_adj, pos_path)

    return positive_adj


def norm_cosine_similarity(feature_a, feature_b, all_pairs=False):
    """Calculate similarity between two nodes using cosine similarity, normalized."""
    if not all_pairs:
        cosine = F.cosine_similarity(feature_a, feature_b, dim=-1).to(feature_a.device)
        norm_cosine = (cosine + 1.) / 2.
        return norm_cosine
    else:

        feature_a_norm = F.normalize(feature_a, p=2, dim=1)
        feature_b_norm = F.normalize(feature_b, p=2, dim=1)

        similarity_matrix = feature_a_norm @ feature_b_norm.T
        norm_similarity_matrix = (similarity_matrix + 1.) / 2.
        return norm_similarity_matrix


def norm_Jaccard_similarity(neighbors_a, neighbors_b)->float:
    """Calculate similarity between two nodes using jaccard similarity, normalized."""

    intersection = torch.count_nonzero(neighbors_a * neighbors_b, dim=-1)
    return intersection / (torch.count_nonzero(neighbors_a) + torch.count_nonzero(neighbors_b, dim=-1) - intersection)

def edge_index2adj(edge_idx, num_nodes):
    """Convert edge index to adjacency matrix"""

    return torch.sparse_coo_tensor(edge_idx, torch.ones(edge_idx.size(1)), [num_nodes, num_nodes]).to_dense()

def adj2edge_index(adj):
    """Convert adjacency matrix to edge index"""

    return adj.nonzero().t()


def dpr2prg(name, data_dir):
    """Convert dpr dataset to pyg dataset"""

    # Validate file paths
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load dataset based on type
    if name == 'amazon':
        data = AmazonPyg(root=data_dir, name='computers')
        x = data.x
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        edge_index = data.edge_index

    else:
        dpr_data = Dataset(data_dir, name, setting="prognn", require_mask=True)
        # Handle sparse features
        if hasattr(dpr_data.features, 'todense'):
            x = torch.FloatTensor(dpr_data.features.todense())
        else:
            x = torch.FloatTensor(dpr_data.features)

        edge_index = torch.tensor(dpr_data.adj.todense()).nonzero().t()
        y = torch.LongTensor(dpr_data.labels)
        train_mask = dpr_data.train_mask
        val_mask = dpr_data.val_mask
        test_mask = dpr_data.test_mask

    # Load and process attacked edges

    prg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return prg_data

def dpr2prg_from_data(name, data_dir, data_adj):
    """Convert dpr dataset to pyg dataset with pretrained adjacency matrix"""

    # Validate file paths
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load dataset based on type
    if name == 'amazon':
        data = AmazonPyg(root=data_dir, name='computers')
        x = data.x
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        dpr_data = Dataset(data_dir, name, setting="prognn", require_mask=True)
        # Handle sparse features
        if hasattr(dpr_data.features, 'todense'):
            x = torch.FloatTensor(dpr_data.features.todense())
        else:
            x = torch.FloatTensor(dpr_data.features)
        y = torch.LongTensor(dpr_data.labels)
        train_mask = dpr_data.train_mask
        val_mask = dpr_data.val_mask
        test_mask = dpr_data.test_mask

    # Load and process attacked edges
    edge_index = torch.tensor(data_adj.adj.todense()).nonzero().t()

    prg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return prg_data

def attack_dpr2prg(name, data_dir, att_edge_path):
    """Convert dpr dataset to pyg dataset using attacked edges"""

    # Validate file paths
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(att_edge_path):
        raise FileNotFoundError(f"Attack edge file not found: {att_edge_path}")

    # Load dataset based on type
    if name == 'amazon':
        data = AmazonPyg(root=data_dir, name='computers')
        x = data.x
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        dpr_data = Dataset(data_dir, name, setting="prognn", require_mask=True)
        # Handle sparse features
        if hasattr(dpr_data.features, 'todense'):
            x = torch.FloatTensor(dpr_data.features.todense())
        else:
            x = torch.FloatTensor(dpr_data.features)
        y = torch.LongTensor(dpr_data.labels)
        train_mask = dpr_data.train_mask
        val_mask = dpr_data.val_mask
        test_mask = dpr_data.test_mask

    # Load and process attacked edges
    edge = torch.load(att_edge_path)

    edge_index = edge.indices()

    prg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return prg_data

def idx_to_mask(idx, nodes_num):
    """Convert a indices array to a tensor mask matrix"""
    mask = torch.zeros(nodes_num)
    mask[idx] = 1
    return mask.bool()

def load_all(model, optimizer=None, dir=None, file_name=None):
    """Load model and optimizer"""
    if dir is None or file_name is None:
        print("dir and file_name must be specified")

    path = str(os.path.join(dir, file_name))
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])


def save_all(model, optimizer, dir, file_name):
    """Save model and optimizer"""
    path = str(os.path.join(dir, file_name))
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state_dict, path)

def graph_augmentation(edge_index, x):
    """Augment graph with cosine similarity and k cycles."""

    n = x.shape[0]
    
    adj = torch.zeros(n, n, dtype=torch.long, device=x.device)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1

    return adj.nonzero().t()

def _k_step_reachability(adj: torch.Tensor, n:int, k: int):
        reach = torch.eye(n, device=adj.device, dtype=adj.dtype)
        for _ in range(k):
            reach = reach @ adj
        return reach


def _find_single_k_cycle(start: int, k: int, adj: torch.Tensor):
    from collections import deque

    queue = deque()
    queue.append((start, [start], {start}))

    while queue:
        current, path, visited = queue.popleft()

        if len(path) == k + 1:
            if path[0] == path[-1]:
                return set(path[1:-1])
            continue

        if len(path) > k + 1:
            continue

        neighbors = torch.where(adj[current] > 0)[0].tolist()
        neighbors.sort(key=lambda x: torch.sum(adj[x]).item())

        for neighbor in neighbors:
            if len(path) >= 2 and neighbor == path[-2]:
                continue

            if len(path) == k and neighbor != path[0]:
                continue

            if neighbor in visited and neighbor != path[0]:
                continue

            new_path = path + [neighbor]
            new_visited = visited | {neighbor} if neighbor != path[0] else visited
            queue.append((neighbor, new_path, new_visited))

    return set()

def compute_sorted_spectrum(num_nodes, edge_index, k=None, normalized=True, return_tensors=True, device='cpu'):
    """Compute sorted spectrum of Laplacian matrix."""
    adj_sparse = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    degrees = adj_sparse.sum(axis=1).A1

    if normalized:
        degrees_sqrt_inv = 1.0 / np.sqrt(np.maximum(degrees, 1))
        D_sqrt_inv = diags(degrees_sqrt_inv, format='csr')
        L = eye(num_nodes, format='csr') - D_sqrt_inv @ adj_sparse @ D_sqrt_inv
    else:
        D = diags(degrees, format='csr')
        L = D - adj_sparse

    if k is None:
        k = num_nodes - 1
        if k <= 0:
            k = 1

    try:
        eigenvalues, eigenvectors = eigsh(
            L,
            k=k,
            which='SA',       
            maxiter=5000,
            tol=1e-5          
        )
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

    except Exception:
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        eigenvalues = eigenvalues[1:k+1]
        eigenvectors = eigenvectors[:, 1:k+1]

    pos_counts = (eigenvectors > 0).sum(axis=0)
    neg_counts = (eigenvectors < 0).sum(axis=0)
    flip_mask = pos_counts < neg_counts
    eigenvectors[:, flip_mask] *= -1

    if return_tensors:
        eigenvalues = torch.from_numpy(eigenvalues).float().to(device)
        eigenvectors = torch.from_numpy(eigenvectors).float().to(device)

    return eigenvalues, eigenvectors


def spectrum_map(x, eigenvectors):
    """
    map feature vector to k_min spectrums
    x: n x d
    eigenvectors: n x k
    """

    w = x.t() @ eigenvectors  # d x k
    # w = F.normalize(w, p=2, dim=1)  # d x k

    s_map = w @ eigenvectors.t()  # d x n
    return s_map.t()  # n x d



def visualize_spectrum(eigenvalues, eigenvectors, title="Spectrum Visualization", save_path=None):
    """Visualize spectrum of Laplacian matrix."""

    import matplotlib.pyplot as plt
    
    if isinstance(eigenvalues, torch.Tensor):
        eigenvalues = eigenvalues.cpu().numpy()
    if isinstance(eigenvectors, torch.Tensor):
        eigenvectors = eigenvectors.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(range(len(eigenvalues)), eigenvalues, 'bo-', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Eigenvalue')
    axes[0, 0].set_title('Eigenvalue Spectrum')
    axes[0, 0].grid(True, alpha=0.3)
    
    counts, bins, patches = axes[0, 1].hist(eigenvalues, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Eigenvalue')
    axes[0, 1].set_ylabel('Proportion')
    axes[0, 1].grid(True, alpha=0.3)
    
    total = len(eigenvalues)
    axes[0, 1].set_yticks(axes[0, 1].get_yticks())
    axes[0, 1].set_yticklabels([f'{tick/total*100:.1f}%' for tick in axes[0, 1].get_yticks()])
    
    mean_val = np.mean(eigenvalues)
    median_val = np.median(eigenvalues)
    var_val = np.var(eigenvalues)
    std_val = np.std(eigenvalues)
    
    axes[0, 1].axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    axes[0, 1].axvline(x=median_val, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
    
    axes[0, 1].axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='yellow', label=f'±1 Std: {std_val:.4f}')
    
    axes[0, 1].legend()
    
    axes[0, 1].set_title(f'Eigenvalue Distribution\n(Variance: {var_val:.4f})')
    
    num_to_show = min(8, eigenvectors.shape[1])
    im = axes[1, 0].imshow(eigenvectors[:, :num_to_show], aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_xlabel('Eigenvector Index')
    axes[1, 0].set_ylabel('Node Index')
    axes[1, 0].set_title(f'First {num_to_show} Eigenvectors')
    plt.colorbar(im, ax=axes[1, 0])
    
    if eigenvectors.shape[1] > 1:
        fiedler = eigenvectors[:, 1]
        axes[1, 1].bar(range(len(fiedler)), fiedler, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Node Index')
        axes[1, 1].set_ylabel('Fiedler Vector Value')
        axes[1, 1].set_title('Fiedler Vector (2nd Eigenvector)')
        axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def get_logger(filename, verbosity=1, name=None):
    """Get logger."""

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def evaluate(model, features, adj, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices == test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]

def reconstruct_adj_from_spectrum(phi, lam, deg, threshold=0.4, device='cpu'):
    """
    phi: (n,k)
    lam: (k,)
    deg: (n,)
    """
    n = phi.size(0)
    L_sym = (phi @ torch.diag(lam) @ phi.t()).to(device)   # (n,n)
    D_sqrt_inv = (torch.diag(1.0 / torch.sqrt(deg + 1e-10))).to(device)
    A = D_sqrt_inv @ (torch.eye(n, device=device) - L_sym) @ D_sqrt_inv
    A = (A + A.t()) / 2
    A = torch.clamp(A, min=0, max=1)
    edge_index = A.nonzero().t()
    edge_weight = A[edge_index[0], edge_index[1]]
    mask = edge_weight >= threshold
    return edge_index[:, mask], edge_weight[mask]

def cal_degree(edge_index, n_nodes):
    adj_sparse = to_scipy_sparse_matrix(edge_index, num_nodes=n_nodes)
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    return torch.from_numpy(degrees)
   
def k_neighbors(embeddings, k):
    emb = embeddings.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(emb)
    distances, indices = nbrs.kneighbors(emb)
    return indices[:, 1:]  # (n, k)

def neighbor_error(anchor, neighbors, temperature=1.0):
    n, d = anchor.shape
    k = neighbors.shape[1]
    pos = anchor[neighbors]  # (n, k, d)
    anchor_norm = F.normalize(anchor, dim=1).unsqueeze(1)  # (n, 1, d)
    pos_norm = F.normalize(pos, dim=2)  # (n, k, d)
    all_norm = F.normalize(anchor, dim=1)  # (n, d)
    # Compute similarity between each anchor and its own neighbors
    sim_pos = torch.sum(anchor_norm * pos_norm, dim=2) / temperature  # (n, k)
    # Compute similarity between each anchor and all nodes
    sim_all = torch.mm(anchor_norm.squeeze(1), all_norm.T) / temperature  # (n, n)

    pos_exp = torch.exp(sim_pos).sum(dim=1)  # (n,)
    all_exp = torch.exp(sim_all).sum(dim=1)  # (n,)
    loss = -torch.log(pos_exp / all_exp).mean()
    return loss


def visualize_embeddings(embeddings, labels=None, save_path=None, perplexity=30,
                         random_state=42, figsize=(3.5, 3.5)):
    r"""Visualize node embeddings via t-SNE.

    Args:
        embeddings (Tensor or ndarray): Node embeddings of shape (n, d).
        labels (Tensor or ndarray, optional): Node labels for coloring, shape (n,).
        save_path (str, optional): If provided, save figure to this path.
        perplexity (float): t-SNE perplexity parameter.
        random_state (int): Random seed for reproducibility.
        figsize (tuple): Figure size in inches.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n = embeddings.shape[0]
    perp = min(perplexity, (n - 1) / 3)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state,
                init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(embeddings)

    x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
    y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
    span = max(x_max - x_min, y_max - y_min) * 1.08
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    x_lim = (x_mid - span / 2, x_mid + span / 2)
    y_lim = (y_mid - span / 2, y_mid + span / 2)

    if labels is not None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        if n_labels <= 10:
            cmap = plt.cm.tab10
        elif n_labels <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.gist_ncar

    plt.rcParams.update({
        'font.family': 'serif',
        'lines.linewidth': 0.8,
        'lines.markersize': 2,
    })

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        for label in unique_labels:
            mask = labels == label
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=3, alpha=0.7,
                       color=cmap(label % cmap.N),
                       edgecolors='none', rasterized=True)
    else:
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=1.5, alpha=0.7,
                   color='steelblue', edgecolors='none', rasterized=True)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.show()


def lineplot_from_excel(filepath, xlabel=None, ylabel=None, save_path=None,
                        figsize=(3.5, 3.0), markersize=4, linewidth=1.2, multiplelpcator=5):
    r"""Read data from an Excel file and produce an line chart.

    The Excel file is expected to have the first column as series names and the
    remaining columns as data values (column headers are used as x-axis ticks).

    Args:
        filepath (str): Path to the .xlsx file.
        xlabel (str, optional): x-axis label.
        ylabel (str, optional): y-axis label.
        save_path (str, optional): If provided, save figure to this path.
        figsize (tuple): Figure size in inches.
        markersize (float): Marker size.
        linewidth (float): Line width.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_excel(filepath)
    model_names = df.iloc[:, 0].values
    x_ticks = [str(c) for c in df.columns[1:]]
    data = df.iloc[:, 1:].values.astype(float)

    # B&W-friendly line styles with distinct markers for ICDM
    styles = [
        {'marker': 'o', 'linestyle': '-',   'color': 'black'},
        {'marker': 's', 'linestyle': '--',  'color': '#2c3e50'},
        {'marker': '^', 'linestyle': '-.',  'color': '#7f8c8d'},
        {'marker': 'D', 'linestyle': ':',   'color': '#e74c3c'},
        {'marker': 'v', 'linestyle': (0, (3, 1, 1, 1)), 'color': '#2980b9'},
        {'marker': 'p', 'linestyle': (0, (5, 2)),       'color': '#8e44ad'},
        {'marker': '*', 'linestyle': (0, (1, 1)),       'color': '#16a085'},
        {'marker': 'X', 'linestyle': (0, (3, 2, 1, 2)), 'color': '#c0392b'},
        {'marker': 'h', 'linestyle': (0, (4, 1, 2, 1)), 'color': '#27ae60'},
        {'marker': 'P', 'linestyle': '--',  'color': 'red'},
    ]

    x = list(range(len(x_ticks)))

    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': linewidth,
        'lines.markersize': markersize,
        'legend.fontsize': 8,
    })

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(model_names):
        style = styles[i % len(styles)]
        ax.plot(x, data[i], marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], label=name, markersize=markersize,
                linewidth=linewidth, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.tick_params(pad=1)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(multiplelpcator))

    leg = ax.legend(frameon=True, fancybox=False, edgecolor='black',
                    fontsize=6.5, handlelength=1.2, handletextpad=0.4,
                    borderpad=0.2, labelspacing=0.15, columnspacing=0.4,
                    loc='lower left', ncol=2 if len(model_names) > 5 else 1)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.show()


def ablation_lineplot(filepath, xlabel=None, ylabel=None, save_path=None,
                      figsize=(3.5, 3.0), markersize=4, linewidth=1.2,
                      multiplelpcator=5, full_model='GraphDMAE'):
    r"""Read ablation study data from Excel and produce an line chart.

    The full model is drawn with a prominent style (solid, bold); ablated variants
    use subdued styles. Excel format: first column = variant names, remaining
    columns = data values with numeric headers as x-ticks.

    Args:
        filepath (str): Path to the .xlsx file.
        xlabel (str, optional): x-axis label.
        ylabel (str, optional): y-axis label.
        save_path (str, optional): If provided, save figure to this path.
        figsize (tuple): Figure size in inches.
        markersize (float): Marker size.
        linewidth (float): Line width.
        multiplelpcator (int): Y-axis tick step size.
        full_model (str): Name of the full model row to highlight.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_excel(filepath)
    model_names = df.iloc[:, 0].values
    x_ticks = [str(c) for c in df.columns[1:]]
    data = df.iloc[:, 1:].values.astype(float)

    ablate_styles = [
        {'marker': 's', 'linestyle': '--',  'color': '#7f8c8d', 'lw': linewidth * 0.9},
        {'marker': '^', 'linestyle': '-.',  'color': '#95a5a6', 'lw': linewidth * 0.9},
        {'marker': 'D', 'linestyle': ':',   'color': '#bdc3c7', 'lw': linewidth * 0.9},
        {'marker': 'v', 'linestyle': (0, (3, 1, 1, 1)), 'color': '#7f8c8d', 'lw': linewidth * 0.9},
        {'marker': 'p', 'linestyle': (0, (5, 2)),       'color': '#95a5a6', 'lw': linewidth * 0.9},
        {'marker': 'h', 'linestyle': (0, (4, 1, 2, 1)), 'color': '#bdc3c7', 'lw': linewidth * 0.9},
        {'marker': '*', 'linestyle': (0, (1, 1)),       'color': '#7f8c8d', 'lw': linewidth * 0.9},
        {'marker': 'X', 'linestyle': (0, (3, 2, 1, 2)), 'color': '#95a5a6', 'lw': linewidth * 0.9},
        {'marker': 'P', 'linestyle': '--',  'color': '#bdc3c7', 'lw': linewidth * 0.9},
    ]
    full_style = {'marker': 'o', 'linestyle': '-', 'color': '#c0392b',
                  'lw': linewidth * 1.2, 'ms': markersize * 1.1, 'zorder': 10}

    x = list(range(len(x_ticks)))

    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': linewidth,
        'lines.markersize': markersize,
        'legend.fontsize': 6.5,
    })

    fig, ax = plt.subplots(figsize=figsize)

    ablate_idx = 0
    for i, name in enumerate(model_names):
        if name == full_model:
            ax.plot(x, data[i], marker=full_style['marker'],
                    linestyle=full_style['linestyle'], color=full_style['color'],
                    label=name, markersize=full_style['ms'],
                    linewidth=full_style['lw'], zorder=full_style['zorder'],
                    clip_on=False)
        else:
            style = ablate_styles[ablate_idx % len(ablate_styles)]
            ablate_idx += 1
            ax.plot(x, data[i], marker=style['marker'],
                    linestyle=style['linestyle'], color=style['color'],
                    label=name, markersize=markersize,
                    linewidth=style['lw'], clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.tick_params(pad=1)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(multiplelpcator))

    leg = ax.legend(frameon=True, fancybox=False, edgecolor='black',
                    fontsize=6.5, handlelength=1.2, handletextpad=0.4,
                    borderpad=0.2, labelspacing=0.15, columnspacing=0.4,
                    loc='lower left', ncol=2 if len(model_names) > 5 else 1)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.show()


def contour_from_excel(filepath, xlabel=r'$k$', ylabel=r'$k_l$', save_path=None,
                       figsize=(3.5, 3.0), levels=30, cmap='viridis'):
    r"""Read a 2D parameter grid from Excel and produce an contour plot.

    Excel format: first column = y-axis parameter values, column headers = x-axis
    parameter values, interior cells = metric values (e.g., accuracy).

    Args:
        filepath (str): Path to the .xlsx file.
        xlabel (str, optional): x-axis label (column parameter name).
        ylabel (str, optional): y-axis label (row parameter name).
        save_path (str, optional): If provided, save figure to this path.
        figsize (tuple): Figure size in inches.
        levels (int): Number of contour levels.
        cmap (str): Matplotlib colormap name.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_excel(filepath)
    y_params = df.iloc[:, 0].values.astype(float)
    x_params = df.columns[1:].values.astype(float)
    z_values = df.iloc[:, 1:].values.astype(float)

    X, Y = np.meshgrid(x_params, y_params)

    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 8,
        'axes.labelsize': 16,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cf = ax.contourf(X, Y, z_values, levels=levels, cmap=cmap, alpha=0.9)
    cs = ax.contour(X, Y, z_values, levels=6, colors='black', linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')

    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.tick_params(pad=1)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=20)
    cbar.set_label('Accuracy (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.outline.set_linewidth(0.5)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.show()
