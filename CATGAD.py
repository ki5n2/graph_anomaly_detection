#%%
from torch_geometric.nn import GINConv, global_add_pool
from torch_scatter import scatter_mean, scatter_std

import os
import re
import sys
import json
import math
import time
import wandb
import torch
import random
import argparse
import numpy as np
import gudhi as gd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.utils as utils

from tqdm import tqdm
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, GINConv
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch
from torch_geometric.transforms import Compose

from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, silhouette_score, silhouette_samples

from functools import partial
from multiprocessing import Pool

from module.loss import loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, compute_persistence_and_betti, process_batch_graphs, get_ad_dataset_Tox21

import networkx as nx


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='AIDS')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--gamma-cluster", type=float, default=0.5)
parser.add_argument("--node-theta", type=float, default=0.03)
parser.add_argument("--adj-theta", type=float, default=0.01)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
'''OPTIONS'''
data_root: str = args.data_root
assets_root: str = args.assets_root
dataset_name: str = args.dataset_name

epochs: int = args.epochs
patience: int = args.patience
step_size: int = args.step_size
batch_size: int = args.batch_size
n_cross_val: int = args.n_cross_val
random_seed: int = args.random_seed
hidden_dims: list = args.hidden_dims
log_interval: int = args.log_interval
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
dropout_rate: float = args.dropout_rate
learning_rate: float = args.learning_rate

alpha: float = args.alpha
beta: float = args.beta
gamma: float = args.gamma
gamma_cluster: float = args.gamma_cluster
node_theta: float = args.node_theta
adj_theta: float = args.adj_theta

set_seed(random_seed)

device = set_device()
# device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#%%
class FeatureTransform:
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, x):
        mask = torch.rand(x.size()) < self.p
        x_aug = x.clone()
        x_aug[mask] = torch.randn_like(x[mask])
        return x_aug

class StructureTransform:
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, edge_index):
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) >= self.p
        return edge_index[:, mask]


class FeatureTransform:
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, x):
        mask = torch.rand(x.size()) < self.p
        x_aug = x.clone()
        x_aug[mask] = torch.randn_like(x[mask])
        return x_aug
    

class StructureTransform:
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, edge_index):
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) >= self.p
        return edge_index[:, mask]


class CVTGAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_type='gin', alpha=1.0, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.dropout = dropout
        
        # GNN Encoders
        if gnn_type == 'gin':
            self.gnn_f = GINConv(MLP(input_dim, hidden_dim, hidden_dim))
            self.gnn_s = GINConv(MLP(input_dim, hidden_dim, hidden_dim))
        else:
            self.gnn_f = GCNConv(input_dim, hidden_dim)
            self.gnn_s = GCNConv(input_dim, hidden_dim)
        
        # Transformer modules
        self.transformer_f = SimplifiedTransformer(hidden_dim, dropout=dropout)
        self.transformer_s = SimplifiedTransformer(hidden_dim, dropout=dropout)
        
        # Cross-view attention
        self.cross_attn = CrossViewAttention(hidden_dim, dropout=dropout)
    
    def augment_features(self, x):
        """Feature view augmentation - perturbation-free strategy"""
        x_aug = x.clone()
        feature_mask = torch.bernoulli(torch.ones_like(x) * 0.1).bool()
        x_aug[feature_mask] = torch.randn_like(x[feature_mask]) * 0.1 + x[feature_mask]
        return x_aug
    
    def augment_structure(self, edge_index, num_nodes):
        """Structure view augmentation - perturbation-free strategy"""
        # Dense adjacency matrix로 변환
        adj = to_dense_adj(edge_index)[0]
        
        # 랜덤 마스크 생성 (10%의 엣지를 수정)
        mask = torch.bernoulli(torch.ones_like(adj) * 0.1).bool()
        
        # 마스크된 위치의 엣지 플립
        adj_aug = adj.clone()
        adj_aug[mask] = 1 - adj[mask]
        
        # Sparse format으로 다시 변환
        edge_index_aug = torch.nonzero(adj_aug).t()
        
        return edge_index_aug

    def loss_function(self, h_f_final, h_s_final, h_f, h_s, batch, tau=0.5):
        """Calculate the total loss combining node-level and graph-level losses."""
        # Node-level loss
        l_node = self._node_level_loss(h_f, h_s, batch, tau)
        
        # Graph-level loss
        l_graph = self._graph_level_loss(h_f_final, h_s_final, tau)
        
        # Adaptive weighting (식 12)
        sigma_node = torch.std(l_node.detach(), unbiased=False) + 1e-6
        sigma_graph = torch.std(l_graph.detach(), unbiased=False) + 1e-6
        
        lambda1 = sigma_node.pow(self.alpha)
        lambda2 = sigma_graph.pow(self.alpha)
        
        # Final loss combining both levels
        total_loss = lambda1 * l_node.mean() + lambda2 * l_graph.mean()
        
        return total_loss

    def _node_level_loss(self, h_f, h_s, batch, tau):
        """Calculate node-level contrastive loss (식 8, 9)"""
        losses = []
        
        for b in torch.unique(batch):
            mask = (batch == b)
            if not mask.any():
                continue
                
            h_f_i = h_f[mask]
            h_s_i = h_s[mask]
            
            # Normalize embeddings
            h_f_i = F.normalize(h_f_i, dim=1, eps=1e-8)
            h_s_i = F.normalize(h_s_i, dim=1, eps=1e-8)
            
            # Positive pairs similarity
            pos_sim = (h_f_i * h_s_i).sum(dim=1) / tau
            
            # Negative pairs similarity
            sim_matrix = torch.mm(h_f_i, h_s_i.t()) / tau
            
            # Remove diagonal elements for proper negative sampling
            mask_neg = ~torch.eye(h_f_i.size(0), dtype=bool, device=h_f_i.device)
            neg_sim = sim_matrix[mask_neg].reshape(h_f_i.size(0), -1)
            
            # Loss calculation
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)
            loss = -torch.log(numerator / denominator).mean()
            
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=h_f.device, requires_grad=True)
            
        return torch.stack(losses)

    def _graph_level_loss(self, h_f, h_s, tau):
        """Calculate graph-level contrastive loss (식 10, 11)"""
        # Normalize embeddings
        h_f = F.normalize(h_f, dim=1, eps=1e-8)
        h_s = F.normalize(h_s, dim=1, eps=1e-8)
        
        # Positive pairs similarity
        pos_sim = (h_f * h_s).sum(dim=1) / tau
        
        # Negative pairs similarity
        sim_matrix = torch.mm(h_f, h_s.t()) / tau
        
        # Remove diagonal elements for proper negative sampling
        mask_neg = ~torch.eye(h_f.size(0), dtype=bool, device=h_f.device)
        neg_sim = sim_matrix[mask_neg].reshape(h_f.size(0), -1)
        
        # Loss calculation
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)
        losses = -torch.log(numerator / denominator)
        
        return losses


    def calculate_anomaly_score(self, h_f_final, h_s_final, h_f, h_s, batch):
        """Calculate anomaly scores for each graph in the batch"""
        scores = []
        batch_size = h_f_final.size(0)
        eps = 1e-8  # 숫자 안정성을 위한 작은 값
        
        # 각 그래프에 대한 이상 점수 계산
        for i in range(batch_size):
            # 현재 그래프의 노드 선택
            mask = (batch == i)
            if not mask.any():
                scores.append(0.0)  # 빈 그래프에 대한 기본값
                continue
                
            h_f_i = h_f[mask]
            h_s_i = h_s[mask]
            
            try:
                # Node-level similarity 계산
                h_f_i = F.normalize(h_f_i, dim=1, eps=eps)
                h_s_i = F.normalize(h_s_i, dim=1, eps=eps)
                node_sims = torch.sum(h_f_i * h_s_i, dim=1)
                node_sim = torch.mean(node_sims)
                
                # Graph-level similarity 계산
                h_f_g = F.normalize(h_f_final[i].unsqueeze(0), dim=1, eps=eps)
                h_s_g = F.normalize(h_s_final[i].unsqueeze(0), dim=1, eps=eps)
                graph_sim = torch.sum(h_f_g * h_s_g)
                
                # Cross-view discrepancy as anomaly score
                # 높은 점수 = 높은 이상치 가능성
                score = -(node_sim + graph_sim).item() / 2
                
                # NaN 값 처리
                if torch.isnan(torch.tensor(score)):
                    score = float('inf')  # NaN의 경우 가장 높은 이상치 점수 할당
                    
            except (RuntimeError, ValueError):
                score = float('inf')
                
            scores.append(score)
        
        # Min-max normalization for better comparison
        scores = torch.tensor(scores, device=h_f.device)
        if scores.size(0) > 1:  # 배치 크기가 1보다 큰 경우에만 정규화
            scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)
        
        return scores    
    
    
    def forward(self, x, edge_index, batch):
        # Generate views using perturbation-free augmentation
        x_f = self.augment_features(x)
        edge_index_s = self.augment_structure(edge_index, x.size(0))
        
        # GNN encoding
        h_f = self.gnn_f(x_f, edge_index)
        h_s = self.gnn_s(x, edge_index_s)
        
        # Graph-level pooling
        h_f_g = global_mean_pool(h_f, batch)
        h_s_g = global_mean_pool(h_s, batch)
        
        # Get batch information for receptive field
        batch_size = torch.unique(batch).size(0)
        node_counts = torch.bincount(batch)
        max_nodes = node_counts.max()
        
        # Transformer processing with receptive field consideration
        h_f_t = self.transformer_f(h_f_g, batch_size)
        h_s_t = self.transformer_s(h_s_g, batch_size)
        
        # Cross-view attention
        h_f_final, h_s_final = self.cross_attn(h_f_t, h_s_t)
        
        return h_f_final, h_s_final, h_f, h_s
    

#%%
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimplifiedTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection networks
        self.projection = MLP(hidden_dim, hidden_dim*2, hidden_dim)
        self.residual = MLP(hidden_dim, hidden_dim*2, hidden_dim)
        
        # Transformer components
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.inter_graph_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, batch_size):
        # Projection
        proj = self.projection(x)
        res = self.residual(proj)
        
        # Self-attention (intra-graph)
        attn_out1, _ = self.self_attn(proj, proj, proj)
        out = self.norm1(attn_out1 + res)
        
        # Inter-graph attention
        attn_out2, _ = self.inter_graph_attn(out, out, out)
        out = self.norm2(attn_out2 + out)
        
        # Feed-forward
        ffn_out = self.ffn(out)
        out = self.norm3(ffn_out + out)
        
        return out


class CrossViewAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projections for cross-view attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # View co-occurrence projection
        self.cooccur_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_f, x_s):
        # Feature view -> Structure view attention
        Q_f = self.q_proj(x_f)
        K_s = self.k_proj(x_s)
        V_f = self.v_proj(x_f)
        
        # Cross-view attention computation (식 6)
        attn = torch.matmul(Q_f, K_s.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = F.normalize(attn, p=1, dim=-1)  # L1 normalization
        attn = attn / torch.sqrt(torch.tensor(self.hidden_dim, device=x_f.device))
        
        # View co-occurrence modeling
        co_occur = self.cooccur_proj(torch.cat([x_f, x_s], dim=-1))
        
        # Combine attention and co-occurrence
        out_f = torch.matmul(attn, V_f) + co_occur
        
        # Structure view -> Feature view attention (식 7)
        Q_s = self.q_proj(x_s)
        K_f = self.k_proj(x_f)
        V_s = self.v_proj(x_s)
        
        attn = torch.matmul(Q_s, K_f.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = F.normalize(attn, p=1, dim=-1)
        attn = attn / torch.sqrt(torch.tensor(self.hidden_dim, device=x_f.device))
        
        out_s = torch.matmul(attn, V_s) + co_occur
        
        return self.dropout(out_f), self.dropout(out_s)
    
    
#%%
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        h_f_final, h_s_final, h_f, h_s = model(data.x, data.edge_index, data.batch)
        
        # Calculate loss
        loss = model.loss_function(h_f_final, h_s_final, h_f, h_s, data.batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    anomaly_scores = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            h_f_final, h_s_final, h_f, h_s = model(data.x, data.edge_index, data.batch)
            
            # Calculate anomaly scores for each graph
            scores = model.calculate_anomaly_score(h_f_final, h_s_final, h_f, h_s, data.batch)
            
            anomaly_scores.extend(scores.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    anomaly_scores = np.array(anomaly_scores)
    labels = np.array(labels)
    
    # Calculate metrics
    auroc = roc_auc_score(labels, anomaly_scores)
    auprc = average_precision_score(labels, anomaly_scores)
    
    return {
        'AUC-ROC': auroc,
        'AP': auprc,
        'scores': anomaly_scores
    }


#%%
'''DATASETS'''
if dataset_name == 'AIDS' or dataset_name == 'NCI1' or dataset_name == 'DHFR':
    dataset_AN = True
else:
    dataset_AN = False

splits = get_ad_split_TU(dataset_name, n_cross_val)
loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, splits[0], dataset_AN)
num_train = meta['num_train']
num_features = meta['num_feat']
num_edge_features = meta['num_edge_feat']
max_nodes = meta['max_nodes']

print(f'Number of graphs: {num_train}')
print(f'Number of features: {num_features}')
print(f'Number of edge features: {num_edge_features}')
print(f'Max nodes: {max_nodes}')

current_time_ = time.localtime()
current_time = time.strftime("%Y_%m_%d_%H_%M", current_time_)
print(f'random number saving: {current_time}')


#%%
def run(dataset_name, random_seed, trial, device, epoch_results=None):
    if epoch_results is None:
        epoch_results = {}
    
    epoch_interval = 10
    set_seed(random_seed)
    split = splits[trial]
    
    # Load data
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    
    # Initialize model
    model = CVTGAD(
        input_dim=num_features,
        hidden_dim=32,
        gnn_type='gin',
        alpha=1.0
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, epochs+1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate periodically
        if epoch % log_interval == 0:
            # Test
            test_results = test(model, test_loader, device)
            auroc, auprc = test_results['AUC-ROC'], test_results['AP']
            
            print(f'Epoch {epoch}: Training Loss = {train_loss:.4f}, Test AUC = {auroc:.4f}, Test AUPRC = {auprc:.4f}')
            
            # Save results every epoch_interval
            if epoch % epoch_interval == 0:
                if epoch not in epoch_results:
                    epoch_results[epoch] = {
                        'aurocs': [],
                        'auprcs': []
                    }
                epoch_results[epoch]['aurocs'].append(auroc)
                epoch_results[epoch]['auprcs'].append(auprc)
    
    return auroc, epoch_results


#%%
if __name__ == '__main__':
    ad_aucs = []
    fold_times = []
    epoch_results = {}
    
    splits = get_ad_split_TU(dataset_name, n_cross_val)
    
    start_time = time.time()
    
    for trial in range(n_cross_val):
        fold_start = time.time()
        print(f"Starting fold {trial + 1}/{n_cross_val}")
        
        ad_auc, epoch_results = run(
            dataset_name,
            random_seed,
            trial,
            device=device,
            epoch_results=epoch_results
        )
        
        ad_aucs.append(ad_auc)
        fold_duration = time.time() - fold_start
        fold_times.append(fold_duration)
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds")
    
    # 각 에폭별 평균 성능 계산
    epoch_means = {}
    epoch_stds = {}
    for epoch in epoch_results.keys():
        epoch_means[epoch] = {
            'auroc': np.mean(epoch_results[epoch]['aurocs']),
            'auprc': np.mean(epoch_results[epoch]['auprcs'])
        }
        epoch_stds[epoch] = {
            'auroc': np.std(epoch_results[epoch]['aurocs']),
            'auprc': np.std(epoch_results[epoch]['auprcs'])
        }
    
    # 최고 성능을 보인 에폭 찾기
    best_epoch = max(epoch_means.keys(), key=lambda x: epoch_means[x]['auroc'])
    
    # 결과 출력
    print("\n=== Performance at every 10 epochs (averaged over all folds) ===")
    for epoch in sorted(epoch_means.keys()):
        print(f"Epoch {epoch}: AUROC = {epoch_means[epoch]['auroc']:.4f} ± {epoch_stds[epoch]['auroc']:.4f}")
    
    print(f"\nBest average performance achieved at epoch {best_epoch}:")
    print(f"AUROC = {epoch_means[best_epoch]['auroc']:.4f} ± {epoch_stds[best_epoch]['auroc']:.4f}")
    print(f"AUPRC = {epoch_means[best_epoch]['auprc']:.4f} ± {epoch_stds[best_epoch]['auprc']:.4f}")
    
    # 최종 결과 저장
    total_time = time.time() - start_time
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")








# %%
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='Tox21_MMP')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--gamma-cluster", type=float, default=0.5)
parser.add_argument("--node-theta", type=float, default=0.03)
parser.add_argument("--adj-theta", type=float, default=0.01)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
'''OPTIONS'''
data_root: str = args.data_root
assets_root: str = args.assets_root
dataset_name: str = args.dataset_name

epochs: int = args.epochs
patience: int = args.patience
step_size: int = args.step_size
batch_size: int = args.batch_size
n_cross_val: int = args.n_cross_val
random_seed: int = args.random_seed
hidden_dims: list = args.hidden_dims
log_interval: int = args.log_interval
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
dropout_rate: float = args.dropout_rate
learning_rate: float = args.learning_rate

alpha: float = args.alpha
beta: float = args.beta
gamma: float = args.gamma
gamma_cluster: float = args.gamma_cluster
node_theta: float = args.node_theta
adj_theta: float = args.adj_theta

set_seed(random_seed)

device = set_device()
# device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#%%
if dataset_name == 'AIDS' or dataset_name == 'NCI1' or dataset_name == 'DHFR':
    dataset_AN = True
else:
    dataset_AN = False

loaders, meta = get_ad_dataset_Tox21(dataset_name, batch_size, test_batch_size)

num_train = meta['num_train']
num_features = meta['num_feat']
max_nodes = meta['max_nodes']

print(f'Number of graphs: {num_train}')
print(f'Number of features: {num_features}')
print(f'Max nodes: {max_nodes}')

current_time_ = time.localtime()
current_time = time.strftime("%Y_%m_%d_%H_%M", current_time_)
print(f'random number saving: {current_time}')


# %%
def run(dataset_name, random_seed, device=device, epoch_results=None):
    if epoch_results is None:
        epoch_results = {}
    epoch_interval = 10  # 10 에폭 단위로 비교
    
    set_seed(random_seed)
    all_results = []

    loaders, meta = get_ad_dataset_Tox21(dataset_name, batch_size, test_batch_size)

    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    
    # Initialize model
    model = CVTGAD(
        input_dim=num_features,
        hidden_dim=32,
        gnn_type='gin',
        alpha=1.0
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, epochs+1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate periodically
        if epoch % log_interval == 0:
            # Test
            test_results = test(model, test_loader, device)
            auroc, auprc = test_results['AUC-ROC'], test_results['AP']
            
            print(f'Epoch {epoch}: Training Loss = {train_loss:.4f}, Test AUC = {auroc:.4f}, Test AUPRC = {auprc:.4f}')
            
            # Save results every epoch_interval
            if epoch % epoch_interval == 0:
                if epoch not in epoch_results:
                    epoch_results[epoch] = {
                        'aurocs': [],
                        'auprcs': []
                    }
                epoch_results[epoch]['aurocs'].append(auroc)
                epoch_results[epoch]['auprcs'].append(auprc)
    
    return auroc, epoch_results


#%%
if __name__ == '__main__':
    ad_aucs = []
    epoch_results = {}
    
    loaders, meta = get_ad_dataset_Tox21(dataset_name, batch_size, test_batch_size)
    
    start_time = time.time()
    
    ad_auc, epoch_results = run(
        dataset_name,
        random_seed,
        device=device,
        epoch_results=epoch_results
        )
        
    ad_aucs.append(ad_auc)
        
    # 각 에폭별 평균 성능 계산
    epoch_means = {}
    epoch_stds = {}
    for epoch in epoch_results.keys():
        epoch_means[epoch] = {
            'auroc': np.mean(epoch_results[epoch]['aurocs']),
            'auprc': np.mean(epoch_results[epoch]['auprcs'])
        }
        epoch_stds[epoch] = {
            'auroc': np.std(epoch_results[epoch]['aurocs']),
            'auprc': np.std(epoch_results[epoch]['auprcs'])
        }
    
    # 최고 성능을 보인 에폭 찾기
    best_epoch = max(epoch_means.keys(), key=lambda x: epoch_means[x]['auroc'])
    
    # 결과 출력
    print("\n=== Performance at every 10 epochs (averaged over all folds) ===")
    for epoch in sorted(epoch_means.keys()):
        print(f"Epoch {epoch}: AUROC = {epoch_means[epoch]['auroc']:.4f} ± {epoch_stds[epoch]['auroc']:.4f}")
    
    print(f"\nBest average performance achieved at epoch {best_epoch}:")
    print(f"AUROC = {epoch_means[best_epoch]['auroc']:.4f} ± {epoch_stds[best_epoch]['auroc']:.4f}")
    print(f"AUPRC = {epoch_means[best_epoch]['auprc']:.4f} ± {epoch_stds[best_epoch]['auprc']:.4f}")
    
    # 최종 결과 저장
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)

# %%
def visualize_molecular_graph(x, edge_index, batch, y, idx=0):
    """
    분자 그래프를 레이블에 따라 다르게 시각화하는 함수
    
    Parameters:
    x: 노드 특성 텐서
    edge_index: 엣지 인덱스 텐서
    batch: 배치 할당 텐서
    y: 레이블 텐서 (0 또는 1)
    idx: 시각화할 그래프의 인덱스 (배치 내)
    """
    # 특정 분자 그래프의 노드 인덱스 추출
    mask = batch == idx
    sub_x = x[mask]
    
    # edge_index 필터링
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    sub_edge_index = edge_index[:, edge_mask]
    
    # 노드 인덱스 매핑 생성
    idx_map = torch.zeros(mask.size(0), dtype=torch.long)
    idx_map[mask] = torch.arange(mask.sum())
    sub_edge_index = idx_map[sub_edge_index]
    
    # NetworkX 그래프 생성
    G = nx.Graph()
    
    # 노드 추가
    for i in range(len(sub_x)):
        G.add_node(i)
    
    # 엣지 추가
    edges = sub_edge_index.t().tolist()
    G.add_edges_from(edges)
    
    # 그래프 레이아웃 계산
    pos = nx.spring_layout(G)
    
    # 레이블에 따른 색상 및 제목 설정
    if y[idx] == 0:
        node_color = 'lightblue'
        title = 'Molecular Graph (Normal)'
        edge_color = 'gray'
    else:
        node_color = 'salmon'
        title = 'Molecular Graph (Anomaly)'
        edge_color = 'darkred'
    
    # 시각화
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, 
            node_color=node_color,
            node_size=500,
            with_labels=True,
            font_size=12,
            font_weight='bold',
            edge_color=edge_color,
            width=2,
            alpha=0.7)
    
    # 레이블 정보 추가
    plt.title(f'{title} (Label: {y[idx].item()})')
    plt.axis('off')
    plt.show()

def visualize_batch_examples(x, edge_index, batch, y, num_examples=4):
    """
    배치에서 여러 예제를 한 번에 시각화하는 함수
    
    Parameters:
    x: 노드 특성 텐서
    edge_index: 엣지 인덱스 텐서
    batch: 배치 할당 텐서
    y: 레이블 텐서
    num_examples: 시각화할 예제 수
    """
    num_graphs = batch.max().item() + 1
    num_examples = min(num_examples, num_graphs)
    
    # 서브플롯 격자 크기 계산
    grid_size = int(np.ceil(np.sqrt(num_examples)))
    
    plt.figure(figsize=(5*grid_size, 5*grid_size))
    for i in range(num_examples):
        plt.subplot(grid_size, grid_size, i+1)
        
        # 특정 분자 그래프의 노드 인덱스 추출
        mask = batch == i
        sub_x = x[mask]
        
        # edge_index 필터링
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        sub_edge_index = edge_index[:, edge_mask]
        
        # 노드 인덱스 매핑 생성
        idx_map = torch.zeros(mask.size(0), dtype=torch.long)
        idx_map[mask] = torch.arange(mask.sum())
        sub_edge_index = idx_map[sub_edge_index]
        
        # NetworkX 그래프 생성
        G = nx.Graph()
        
        # 노드 추가
        for j in range(len(sub_x)):
            G.add_node(j)
        
        # 엣지 추가
        edges = sub_edge_index.t().tolist()
        G.add_edges_from(edges)
        
        # 그래프 레이아웃 계산
        pos = nx.spring_layout(G)
        
        # 레이블에 따른 색상 설정
        node_color = 'salmon' if y[i] == 1 else 'lightblue'
        edge_color = 'darkred' if y[i] == 1 else 'gray'
        
        # 시각화
        nx.draw(G, pos, 
                node_color=node_color,
                node_size=300,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color=edge_color,
                width=1.5,
                alpha=0.7)
        
        plt.title(f'Label: {y[i].item()}')
    
    plt.tight_layout()
    plt.show()



data.x
data.edge_index
data.node_label


# %%
visualize_batch_examples(data.x, data.edge_index, data.batch, data.y, num_examples=32)
# %%
