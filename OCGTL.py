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
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch

from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

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
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, compute_persistence_and_betti, process_batch_graphs

import networkx as nx


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='BZR')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=500)
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
class GraphNorm(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x, batch):
        # Calculate mean and std per graph
        batch_size = batch.max().item() + 1
        
        mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)
        std = scatter_std(x, batch, dim=0, dim_size=batch_size)
        
        # Expand mean and std to match x's shape
        mean = mean[batch]
        std = std[batch]
        
        # Normalize
        x = (x - mean) / (std + 1e-5)
        return self.scale * x + self.bias

class GIN(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GIN, self).__init__()
        
        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.conv = GINConv(self.mlp)
        self.norm = GraphNorm(hidden_dim)
        
    def forward(self, x, edge_index, batch):
        # GIN convolution
        x = self.conv(x, edge_index)
        # Graph normalization
        x = self.norm(x, batch)
        return x

class OCGTL(nn.Module):
    def __init__(self, num_features, hidden_dim=32, num_layers=4, num_transformations=6):
        super(OCGTL, self).__init__()
        self.num_layers = num_layers
        self.num_transformations = num_transformations
        self.hidden_dim = hidden_dim
        
        # Reference feature extractor
        self.reference_layers = nn.ModuleList([
            GIN(num_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # K transformation feature extractors
        self.transform_layers = nn.ModuleList([
            nn.ModuleList([
                GIN(num_features if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            for _ in range(num_transformations)
        ])
        
        # Final projection for concatenated representations
        self.proj = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable center for OCC
        self.center = nn.Parameter(torch.zeros(hidden_dim))
        
        # Temperature parameter for GTL
        self.temperature = 0.1
        
    def get_graph_repr(self, x, edge_index, batch, layers):
        """Hierarchical graph representation according to equations 4-6"""
        layer_representations = []
        h = x
        
        # Get representations from each layer
        for gin in layers:
            # Update node embeddings (eq. 4)
            h = gin(h, edge_index, batch)
            
            # Get graph-level embedding for this layer (eq. 5)
            h_graph = global_add_pool(h, batch)
            layer_representations.append(h_graph)
        
        # Concatenate all layer representations (eq. 6)
        graph_repr = torch.cat(layer_representations, dim=1)
        
        # Final projection
        graph_repr = self.proj(graph_repr)
        
        return graph_repr

    def forward(self, x, edge_index, batch):
        # Get reference embedding
        ref_embedding = self.get_graph_repr(x, edge_index, batch, self.reference_layers)
        
        # Get transformation embeddings
        trans_embeddings = [
            self.get_graph_repr(x, edge_index, batch, trans_layers)
            for trans_layers in self.transform_layers
        ]
        
        return ref_embedding, trans_embeddings

    def loss(self, ref_embedding, trans_embeddings):
        batch_size = ref_embedding.size(0)
        
        # OCC Loss (eq. 3)
        occ_loss = sum(
            torch.norm(embedding - self.center, p=2, dim=1).mean()
            for embedding in trans_embeddings
        )
        
        # GTL Loss (eq. 2)
        gtl_loss = 0
        for k, trans_emb in enumerate(trans_embeddings):
            # Compute similarities
            pos_sim = F.cosine_similarity(trans_emb, ref_embedding) / self.temperature
            
            # Compute similarities with other transformations
            neg_sims = torch.stack([
                F.cosine_similarity(trans_emb, other_emb) / self.temperature
                for i, other_emb in enumerate(trans_embeddings)
                if i != k
            ])
            
            # Compute loss terms
            pos_term = torch.exp(pos_sim)
            neg_term = torch.sum(torch.exp(neg_sims), dim=0)
            
            # Add to GTL loss
            gtl_loss -= torch.mean(torch.log(pos_term / (pos_term + neg_term)))
        
        # Total loss (eq. 1)
        total_loss = occ_loss + gtl_loss
        
        return total_loss
    
    def score_samples(self, x, edge_index, batch):
        """
        Compute anomaly scores for test samples using only OCC distance
        Higher scores indicate higher likelihood of being anomalous
        """
        ref_embedding, trans_embeddings = self.forward(x, edge_index, batch)
        
        # Compute anomaly scores as the average distance to center for all transformations
        scores = torch.stack([
            torch.norm(embedding - self.center, p=2, dim=1)
            for embedding in trans_embeddings
        ]).mean(dim=0)
        
        return scores


#%%
def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_samples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        ref_emb, trans_embs = model(batch.x, batch.edge_index, batch.batch)
        
        # Compute loss
        loss = model.loss(ref_emb, trans_embs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    
    avg_loss = total_loss / num_samples
    return avg_loss


def test(model, test_loader, device):
    """Evaluate the model on test set"""
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Compute anomaly scores (distance to center only)
            scores = model.score_samples(batch.x, batch.edge_index, batch.batch)
            
            # Forward pass (for monitoring overall loss)
            ref_emb, trans_embs = model(batch.x, batch.edge_index, batch.batch)
            loss = model.loss(ref_emb, trans_embs)
            
            # Collect results
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            num_samples += batch_size
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)
    avg_loss = total_loss / num_samples
    
    return {
        'AUC-ROC': auroc,
        'AP': auprc,
        'loss': avg_loss,
        'scores': all_scores,
        'labels': all_labels
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


# %%
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
    model = OCGTL(
        num_features=num_features,
        hidden_dim=32,
        num_layers=4,
        num_transformations=6
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
                        'aurocs': [], 'auprcs': []
                    }
                epoch_results[epoch]['aurocs'].append(auroc)
                epoch_results[epoch]['auprcs'].append(auprc)
    
    return auroc, epoch_results


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
    
    