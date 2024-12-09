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
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, compute_persistence_and_betti, process_batch_graphs

import networkx as nx


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=150)
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
class GmapAD(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_candidates=64, 
                 mutation_rate=0.5, crossover_rate=0.9, pool_size=30):
        super(GmapAD, self).__init__()
        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.num_candidates = num_candidates
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate 
        self.pool_size = pool_size
        
    def encode_nodes(self, x, edge_index, batch):
        # Node representation learning - 수식 (1) 구현
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def get_initial_candidates(self, node_embeddings, batch, labels, k):
        """
        Args:
            node_embeddings: 모든 노드의 임베딩 [num_nodes, hidden_dim]
            batch: 각 노드가 어느 그래프에 속하는지 나타내는 벡터 [num_nodes]
            labels: 각 그래프의 레이블 [batch_size]
            k: 선택할 candidate 수
        """
        # 먼저 normal 그래프들을 찾음
        normal_graphs = (labels == 1).nonzero().squeeze()
        
        # normal 그래프에 속한 노드들의 마스크 생성
        normal_node_mask = torch.zeros_like(batch, dtype=torch.bool)
        for graph_idx in normal_graphs:
            normal_node_mask |= (batch == graph_idx)
        
        # normal 그래프의 노드 임베딩 선택
        normal_nodes = node_embeddings[normal_node_mask]
        
        # 모든 노드와 normal 노드 간의 cosine similarity 계산
        similarities = F.cosine_similarity(
            node_embeddings.unsqueeze(1),
            normal_nodes.unsqueeze(0),
            dim=2
        )
        
        # 각 노드에 대해 평균 similarity 계산
        scores = similarities.mean(dim=1)
        
        # Top-k 노드 선택
        _, indices = torch.topk(scores, k=min(k, len(scores)))
        return indices

    
    def differential_evolution(self, candidates, graph_reprs, labels):
        # 초기 population 생성
        population = torch.zeros((self.pool_size, len(candidates))).bernoulli_(p=0.5)
        best_score = float('inf')
        best_candidate = None
        
        for _ in range(2000):
            for i in range(self.pool_size):
                # Mutation - 수식 (7) 구현
                r_indices = np.random.choice(self.pool_size, 5, replace=False)
                d = (population[r_indices[0]] + 
                     self.mutation_rate * (population[r_indices[1]] + 
                                         population[r_indices[2]] + 
                                         population[r_indices[3]] + 
                                         population[r_indices[4]]))
                mutant = torch.where(d >= 2, torch.ones_like(d), torch.zeros_like(d))
                
                # Crossover - 수식 (8) 구현
                r = torch.randint(0, len(candidates), (1,))[0]
                cross_points = torch.rand(len(candidates)) < self.crossover_rate
                cross_points[r] = True
                trial = torch.where(cross_points, mutant, population[i])
                
                # Objective function - 수식 (5), (6) 구현
                trial_candidates = candidates[trial.bool()]
                if len(trial_candidates) > 0:
                    similarities = self.compute_similarity(graph_reprs, trial_candidates)
                    normal_mask = labels == 1
                    anomaly_mask = labels == 0
                    
                    loss = 0
                    if normal_mask.any():
                        normal_loss = F.hinge_embedding_loss(
                            similarities[normal_mask].mean(1),
                            torch.ones(normal_mask.sum()).to(similarities.device),
                            margin=1.0
                        )
                        loss += normal_loss
                    
                    if anomaly_mask.any():
                        anomaly_loss = F.hinge_embedding_loss(
                            similarities[anomaly_mask].mean(1),
                            -torch.ones(anomaly_mask.sum()).to(similarities.device),
                            margin=1.0
                        )
                        loss += anomaly_loss
                        
                    trial_score = loss.item()
                    
                    if trial_score < best_score:
                        best_score = trial_score
                        best_candidate = trial.clone()
                    
                    if trial_score < self.compute_score(population[i], candidates, graph_reprs, labels):
                        population[i] = trial
        
        return candidates[best_candidate.bool()]
    
    def compute_similarity(self, graph_repr, candidate_nodes):
        """수식 (3) 구현"""
        similarities = []
        for node in candidate_nodes:
            # L2 distance 계산
            sim = torch.norm(graph_repr.unsqueeze(1) - node, p=2, dim=2)
            similarities.append(sim)
        return torch.stack(similarities, dim=1)


    def compute_score(self, candidate_indices, candidates, graph_reprs, labels):
        """수식 (5), (6) 구현"""
        if not candidate_indices.any():
            return float('inf')
            
        selected_candidates = candidates[candidate_indices.bool()]
        similarities = self.compute_similarity(graph_reprs, selected_candidates)
        
        normal_mask = labels == 1
        anomaly_mask = labels == 0
        
        loss = 0
        if normal_mask.any():
            normal_loss = F.hinge_embedding_loss(
                similarities[normal_mask].mean(1),
                torch.ones(normal_mask.sum()).to(similarities.device),
                margin=1.0
            )
            loss += normal_loss
        
        if anomaly_mask.any():
            anomaly_loss = F.hinge_embedding_loss(
                similarities[anomaly_mask].mean(1),
                -torch.ones(anomaly_mask.sum()).to(similarities.device),
                margin=1.0
            )
            loss += anomaly_loss
            
        return loss.item()


    def forward(self, x, edge_index, batch, labels=None):
        # 1. Node representation learning
        node_embeddings = self.encode_nodes(x, edge_index, batch)
        graph_repr = global_mean_pool(node_embeddings, batch)
        
        if self.training and labels is not None:
            # 2. Initial candidate selection - batch 정보 추가
            init_candidates = self.get_initial_candidates(
                node_embeddings, 
                batch,  # batch 정보 전달
                labels, 
                self.num_candidates
            )
            candidate_nodes = node_embeddings[init_candidates]
            
            # 3. DE optimization
            self.optimized_candidates = self.differential_evolution(candidate_nodes, graph_repr, labels)
            
            # 4. Final mapping and prediction
            similarities = self.compute_similarity(graph_repr, self.optimized_candidates)
            return similarities
        else:
            if hasattr(self, 'optimized_candidates'):
                similarities = self.compute_similarity(graph_repr, self.optimized_candidates)
                return similarities
            else:
                raise RuntimeError("Model needs to be trained first")
            

#%%            
# Loss function for anomaly-aware training
class AnomalyAwareLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(AnomalyAwareLoss, self).__init__()
        self.margin = margin
        
    def forward(self, similarities, labels):
        normal_mask = labels == 1
        anomaly_mask = labels == 0
        
        loss = 0
        if normal_mask.any():
            normal_loss = F.hinge_embedding_loss(
                similarities[normal_mask].mean(1),
                torch.ones(normal_mask.sum()).to(similarities.device),
                margin=self.margin
            )
            loss += normal_loss
        
        if anomaly_mask.any():
            anomaly_loss = F.hinge_embedding_loss(
                similarities[anomaly_mask].mean(1),
                -torch.ones(anomaly_mask.sum()).to(similarities.device),
                margin=self.margin
            )
            loss += anomaly_loss
            
        return loss
    

#%%
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.batch, data.y)
        
        # Compute loss using anomaly-aware loss
        normal_mask = data.y == 1
        anomaly_mask = data.y == 0
        
        loss = 0
        if normal_mask.any():
            normal_loss = F.hinge_embedding_loss(
                out[normal_mask].mean(1),
                torch.ones(normal_mask.sum()).to(device),
                margin=1.0
            )
            loss += normal_loss
        
        if anomaly_mask.any():
            anomaly_loss = F.hinge_embedding_loss(
                out[anomaly_mask].mean(1),
                -torch.ones(anomaly_mask.sum()).to(device),
                margin=1.0
            )
            loss += anomaly_loss
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def test(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # Use mean similarity as anomaly score
            anomaly_scores = out.mean(1)
            
            predictions.append(anomaly_scores.cpu().numpy())
            labels.append(data.y.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    
    # Compute metrics
    auroc = roc_auc_score(labels, -predictions)  # Negative because lower similarity = more anomalous
    auprc = average_precision_score(labels, -predictions)
    
    return {
        'AUC-ROC': auroc,
        'AP': auprc
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
    model = GmapAD(
        input_dim=num_features,
        hidden_dim=128,
        output_dim=64,
        num_candidates=64
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, epochs+1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate periodically
        if epoch % log_interval == 0:
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
    
    # Calculate epoch-wise performance
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
    
    # Find best epoch
    best_epoch = max(epoch_means.keys(), key=lambda x: epoch_means[x]['auroc'])
    
    # Print results
    print("\n=== Performance at every 10 epochs (averaged over all folds) ===")
    for epoch in sorted(epoch_means.keys()):
        print(f"Epoch {epoch}: AUROC = {epoch_means[epoch]['auroc']:.4f} ± {epoch_stds[epoch]['auroc']:.4f}")
    
    print(f"\nBest average performance achieved at epoch {best_epoch}:")
    print(f"AUROC = {epoch_means[best_epoch]['auroc']:.4f} ± {epoch_stds[best_epoch]['auroc']:.4f}")
    print(f"AUPRC = {epoch_means[best_epoch]['auprc']:.4f} ± {epoch_stds[best_epoch]['auprc']:.4f}")
    
    # Final results
    total_time = time.time() - start_time
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")


# %%
