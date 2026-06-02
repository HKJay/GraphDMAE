import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GCNConv
import torch.nn.utils.parametrizations as param_utils


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))


        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))


        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = dropout
        self.activation = nn.PReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphDMAE(nn.Module):
    def __init__(self,
                 feature_dim,
                 L_dim,
                 hidden_dim,
                 output_dim,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 mask_ratio=0.5,
                 replace_ratio=0.1,
                 gamma=3.0):
        super().__init__()

        self.input_dim = feature_dim+L_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.L_dim = L_dim
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio
        self.gamma = gamma

        self.mask_token = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.xavier_uniform_(self.mask_token.data)

        self.decoder_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.decoder_mask_token.data)

        self.mask_lap = nn.Parameter(torch.zeros(1, L_dim))
        nn.init.xavier_uniform_(self.mask_lap.data)

        self.encoder = self._build_gnn(
            self.input_dim, hidden_dim, hidden_dim, num_encoder_layers)

        self.decoder = self._build_gnn(
            hidden_dim, hidden_dim, output_dim, num_decoder_layers)

        self.lap_corrector = self._build_gnn(
            hidden_dim, hidden_dim, L_dim, num_decoder_layers)

        self.output_proj = nn.Linear(output_dim, feature_dim)



    def _build_gnn(self, input_dim, hidden_dim, output_dim, num_layers):
        return GCNModel(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x, lap, edge_index, return_h=False):
        x_masked, mask_indices = self.mask_input(x, self.mask_token)
        lap_masked, mask_indices_lap = self.mask_input(lap, self.mask_lap)

        x_masked = torch.cat([x_masked, lap_masked], dim=1)

        edge_mask_indices = torch.randperm(edge_index.shape[1])[:int(edge_index.shape[1]*0.9)]
        edge_index = edge_index[:, edge_mask_indices]


        h = self.encoder(x_masked, edge_index)

        h_remasked = self.remask_encoding(h, mask_indices, self.decoder_mask_token)

        z = self.decoder(h_remasked, edge_index)
        lap_corrected = self.lap_corrector(h, edge_index)

        x_recon = self.output_proj(z)

        if return_h:
            return x_recon, lap_corrected, mask_indices, mask_indices_lap, h
        return x_recon, lap_corrected, mask_indices, mask_indices_lap

    def encode(self, x, lap, edge_index):
        x = torch.cat([x, lap], dim=1)

        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.output_proj(self.decoder(z, edge_index))

    def lap_correct(self, h, edge_index):
        return self.lap_corrector(h, edge_index)

    def mask_input(self, x, mask_token):
        num_nodes = x.shape[0]

        num_mask_nodes = int(num_nodes * self.mask_ratio)
        mask_indices = torch.randperm(num_nodes)[:num_mask_nodes]

        x_masked = x.clone()

        if self.replace_ratio > 0:
            num_replace = int(num_mask_nodes * self.replace_ratio)
            replace_indices = mask_indices[:num_replace]

            random_indices = torch.randperm(num_nodes)[:num_replace]
            x_masked[replace_indices] = x[random_indices]

            remaining_indices = mask_indices[num_replace:]
            x_masked[remaining_indices] = mask_token
        else:
            x_masked[mask_indices] = mask_token

        return x_masked, mask_indices

    def remask_encoding(self, h, mask_indices, mask_token):
        h_remasked = h.clone()
        h_remasked[mask_indices] = mask_token
        return h_remasked

    def scaled_cosine_error(self, x_original, x_reconstructed, mask_indices):
        x_original_masked = x_original[mask_indices]
        x_reconstructed_masked = x_reconstructed[mask_indices]

        cosine_sim = F.cosine_similarity(x_original_masked, x_reconstructed_masked, dim=-1)

        cosine_error = 1 - cosine_sim

        scaled_loss = torch.pow(cosine_error, self.gamma)

        return scaled_loss.mean()

class GraphMAETrainer:
    def __init__(self,
                 model,
                 lr: float = 1e-3,
                 device: torch.device = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def train_epoch(self, features, clean_lap, adj, neighbors):
        self.model.train()
        total_loss = 0

        features = features.to(self.device)
        clean_lap = clean_lap.to(self.device)
        self.optimizer.zero_grad()

        x_recon, lap_corrected, mask_indices, mask_indices_lap, h = self.model(features, clean_lap, adj, return_h=True)
        loss_recon = self.model.scaled_cosine_error(features, x_recon, mask_indices)
        loss_lap = self.neighbor_error(lap_corrected+clean_lap, neighbors)
        loss_smooth = self.laplacian_smooth_loss(features, adj)

        loss = loss_recon + 0.5 * loss_lap +  0.1 * loss_smooth
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def neighbor_error(self, anchor, neighbors, temperature=1.0):
       n, d = anchor.shape
       k = neighbors.shape[1]
       # neighbors is (n, k) indices, gather neighbor embeddings
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
    
    # def laplacian_smooth_loss(self, h, edge_index):
    #     src, dst = edge_index[0], edge_index[1]
    #     diff = h[src] - h[dst]
    #     loss = torch.mean(diff.pow(2)) 
    #     return loss
    def laplacian_smooth_loss(self, h, edge_index, threshold=0.8):
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]
        h_dst = h[dst]

        sim = F.cosine_similarity(h_src, h_dst, dim=-1)  # [E]

        weight = torch.sigmoid((sim - threshold) * 5.0)
    
        diff = h_src - h_dst
        loss = (weight * diff.pow(2).sum(dim=-1)).mean()
        return loss
