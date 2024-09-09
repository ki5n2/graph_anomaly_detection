#%%
import os
import wandb
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import DataLoader, Dataset, Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, GAE, InnerProductDecoder
from torch_geometric.utils import add_self_loops, k_hop_subgraph, to_dense_adj, subgraph, to_undirected, to_networkx


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, subgraph
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


#%%
class OptimizedGraphAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.1):
        super(OptimizedGraphAutoencoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder.append(GCNConv(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Decoder
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
        self.decoder.append(nn.Linear(hidden_dims[0], num_features))
        
        self.edge_predictor = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x, edge_index):
        for conv in self.encoder:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        return x

    def decode(self, z):
        for linear in self.decoder[:-1]:
            z = F.relu(linear(z))
        return self.decoder[-1](z)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encoding
        z = self.encode(x, edge_index)
        
        # Node feature reconstruction
        x_recon = self.decode(z)
        
        # Edge prediction
        edge_logits = self.edge_predictor(z)
        adj_recon = torch.sigmoid(torch.matmul(edge_logits, edge_logits.t()))
        
        # Graph-level representation
        z_g = global_max_pool(z, batch)
        
        return z, z_g, x_recon, adj_recon


#%%
def combined_loss(x, x_recon, adj_orig, adj_recon, z):
    recon_loss = F.mse_loss(x_recon, x)
    edge_loss = F.binary_cross_entropy(adj_recon, adj_orig)
    
    # KL 손실 계산 수정
    kl_loss = -0.5 * torch.mean(1 + z.var(dim=1) - z.mean(dim=1).pow(2) - z.var(dim=1).exp())
    
    return recon_loss + edge_loss + 0.1 * kl_loss


#%%
def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        z, z_g, x_recon, adj_recon = model(data)
        
        batch_size = data.num_graphs
        loss = 0
        start = 0
        for i in range(batch_size):
            num_nodes = (data.batch == i).sum().item()
            end = start + num_nodes
            
            # 현재 그래프에 해당하는 노드 인덱스 추출
            node_mask = (data.batch == i)
            nodes = node_mask.nonzero().squeeze()
            
            # 현재 그래프의 edge_index 추출
            edge_mask = (data.batch[data.edge_index[0]] == i) & (data.batch[data.edge_index[1]] == i)
            sub_edge_index = data.edge_index[:, edge_mask]
            
            if sub_edge_index.numel() > 0:
                # 노드 인덱스 재매핑
                _, sub_edge_index = subgraph(nodes, sub_edge_index, relabel_nodes=True)
                adj_orig = to_dense_adj(sub_edge_index)[0]
            else:
                # 엣지가 없는 경우 모든 엔트리가 0인 인접 행렬 생성
                adj_orig = torch.zeros((num_nodes, num_nodes), device=device)
            
            loss += combined_loss(data.x[start:end], x_recon[start:end], adj_orig, adj_recon[start:end, start:end], z[start:end])
            start = end
        
        loss = loss / batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def compute_anomaly_score(model, data, device):
    model.eval()
    with torch.no_grad():
        z, z_g, x_recon, adj_recon = model(data.to(device))
        batch_size = data.num_graphs
        anomaly_scores = []
        start = 0
        for i in range(batch_size):
            num_nodes = (data.batch == i).sum().item()
            end = start + num_nodes
            
            # 현재 그래프에 해당하는 노드 인덱스 추출
            node_mask = (data.batch == i)
            nodes = node_mask.nonzero().squeeze()
            
            # 현재 그래프의 edge_index 추출
            edge_mask = (data.batch[data.edge_index[0]] == i) & (data.batch[data.edge_index[1]] == i)
            sub_edge_index = data.edge_index[:, edge_mask]
            
            if sub_edge_index.numel() > 0:
                # 노드 인덱스 재매핑
                _, sub_edge_index = subgraph(nodes, sub_edge_index, relabel_nodes=True)
                adj_orig = to_dense_adj(sub_edge_index)[0]
            else:
                # 엣지가 없는 경우 모든 엔트리가 0인 인접 행렬 생성
                adj_orig = torch.zeros((num_nodes, num_nodes), device=device)
            
            recon_error = F.mse_loss(x_recon[start:end], data.x[start:end], reduction='none').mean()
            edge_error = F.binary_cross_entropy(adj_recon[start:end, start:end], adj_orig, reduction='none').mean()
            anomaly_score = recon_error + edge_error
            anomaly_scores.append(anomaly_score.item())
            start = end
    return np.array(anomaly_scores)


def evaluate(model, data_loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    for data in data_loader:
        scores = compute_anomaly_score(model, data, device)
        all_scores.extend(scores)
        all_labels.extend(data.y.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_scores)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)
    
    return auroc, auprc


#%%
data_name='AIDS'
graph_dataset = TUDataset(root='./dataset', name=data_name).shuffle()

labels = np.array([data.y.item() for data in graph_dataset])

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of edge features: {graph_dataset.num_edge_features}')
print(f'labels: {labels}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = graph_dataset.num_features
dataset_AN = True


#%%
# Main training loop
def main():
    for fold, (train_idx, val_idx) in enumerate(skf.split(graph_dataset, labels)):
        print(f"Fold {fold + 1}")
        
        train_normal_idx = [idx for idx in train_idx if labels[idx] == 1]
        print(len(train_idx))
        print(len(train_normal_idx))
        
        val_normal_idx = [idx for idx in val_idx if labels[idx] == 1]
        val_anormal_idx = [idx for idx in val_idx if labels[idx] == 0]
        print(len(val_normal_idx))
        print(len(val_anormal_idx))
        
        train_dataset = [graph_dataset[i] for i in train_normal_idx]
        val_dataset = [graph_dataset[i] for i in val_idx]
        # val_dataset == 0
        
        if dataset_AN:
            idx = 0
            for data in train_dataset:
                data.y = 0
                data['idx'] = idx
                idx += 1

            for data in val_dataset:
                data.y = 1 if data.y == 0 else 0
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
        
        print(f"  Training set size (normal only): {len(train_dataset)}")
        print(f"  Validation set size (normal + abnormal): {len(val_dataset)}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OptimizedGraphAutoencoder(num_features=graph_dataset.num_features, hidden_dims=[256, 128]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
        for epoch in range(100):
            train_loss = train(model, optimizer, train_loader, device)
            auroc, auprc = evaluate(model, val_loader, device)
            print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')

if __name__ == "__main__":
    main()

