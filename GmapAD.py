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
    
    def get_initial_candidates(self, node_embeddings, labels, k):
        # 초기 candidate 선택 - Section 4.2.2
        normal_mask = labels == 1
        normal_nodes = node_embeddings[normal_mask]
        
        # 수식 (4) 구현 - cosine similarity
        similarities = F.cosine_similarity(
            node_embeddings.unsqueeze(1),
            normal_nodes.unsqueeze(0),
            dim=2
        )
        scores = similarities.mean(dim=1)
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
    
    def forward(self, x, edge_index, batch, labels=None):
        # 1. Node representation learning
        node_embeddings = self.encode_nodes(x, edge_index, batch)
        graph_repr = global_mean_pool(node_embeddings, batch)
        
        if self.training and labels is not None:
            # 2. Initial candidate selection
            init_candidates = self.get_initial_candidates(node_embeddings, labels, self.num_candidates)
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
    def __init__(self):
        super(AnomalyAwareLoss, self).__init__()
        
    def forward(self, pred, labels):
        normal_mask = labels == 1
        anomaly_mask = labels == 0
        
        normal_loss = F.nll_loss(pred[normal_mask], labels[normal_mask])
        anomaly_loss = F.nll_loss(pred[anomaly_mask], labels[anomaly_mask])
        
        return (normal_loss + anomaly_loss) / 2


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
    # Configuration
    dataset_name = 'MUTAG'  # or your dataset
    n_cross_val = 10
    random_seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    test_batch_size = 32
    epochs = 100
    log_interval = 10
    
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

