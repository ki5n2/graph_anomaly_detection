#%%
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

from torch.nn import init
from typing import List, Tuple, Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch

from scipy.linalg import eigh
from scipy.stats import linregress
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, KernelDensity
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, silhouette_score, silhouette_samples

from functools import partial
from multiprocessing import Pool

from module.loss import loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, compute_persistence_and_betti, process_batch_graphs, loocv_bandwidth_selection

import networkx as nx

from torch_geometric.utils import negative_sampling, degree
from typing import Tuple, Optional


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='AIDS')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--n-head-BERT", type=int, default=2)
parser.add_argument("--n-layer-BERT", type=int, default=2)
parser.add_argument("--n-head", type=int, default=2)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--BERT-epochs", type=int, default=100)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=1)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=300)
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
parser.add_argument("--gamma", type=float, default=1.0)
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

BERT_epochs: int = args.BERT_epochs
epochs: int = args.epochs
n_head_BERT: int = args.n_head_BERT
n_layer_BERT: int = args.n_layer_BERT
n_head: int = args.n_head
n_layer: int = args.n_layer
patience: int = args.patience
n_cluster: int = args.n_cluster
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


# %%
def pretrain_graph_bert(model, train_loader, optimizer, device):
    """MaskGAE 프리트레이닝 로직으로 변경"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # MaskGAE forward pass
        z, edge_pred, deg_pred = model(
            data.x, data.edge_index, data.batch, data.num_graphs,
            is_pretrain=True
        )
        
        # MaskGAE loss 계산 - encoder의 loss 메서드 사용
        loss = model.encoder.loss(
            edge_pred=edge_pred,
            masked_edges=model.encoder.masked_edges,
            deg_pred=deg_pred,
            edge_index=data.edge_index
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
        
    return total_loss / num_samples, num_samples


#%%
def train(model, train_loader, recon_optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_sample = 0
    reconstruction_errors = []
    
    for data in train_loader:
        data = process_batch_graphs(data)
        # data = TopER_Embedding(data)
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs, node_label, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label, data.true_stats
        # toper_embeddings = data.toper_embeddings
        
        # dtype 확인 및 변환
        x = x.float()  # float32로 변환
        node_label = node_label.float()  # float32로 변환
        
        train_cls_outputs, x_recon, stats_pred = model(x, edge_index, batch, num_graphs)
        
        if epoch % 5 == 0:
            cls_outputs_np = train_cls_outputs.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_cluster, random_state=random_seed)
            kmeans.fit(cls_outputs_np)
            cluster_centers = kmeans.cluster_centers_
            
        loss = 0
        node_loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            node_loss += node_loss_
            
            if epoch % 5 == 0:
                node_loss_scaled = node_loss_.item() * alpha
                cls_vec = train_cls_outputs[i].detach().cpu().numpy()
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                reconstruction_errors.append({
                    'reconstruction': node_loss_scaled,
                    'clustering': min_distance,
                    'type': 'train_normal'  # 훈련 데이터는 모두 정상
                })
            
            start_node = end_node
            
        stats_loss = persistence_stats_loss(stats_pred, true_stats)
        
        alpha_ = 1
        stats_loss = alpha_ * stats_loss

        loss = node_loss + stats_loss
        
        print(f'node_loss: {node_loss}')
        print(f'stats_loss: {stats_loss}')
        
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), num_sample, train_cls_outputs.detach().cpu(), reconstruction_errors


#%%
def evaluate_model(model, test_loader, cluster_centers, n_clusters, gamma_clusters, random_seed, reconstruction_errors, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors_test = []  # 새로 추가
    
    with torch.no_grad():
        for data in test_loader:
            x, edge_index, batch, num_graphs, node_label, y = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label, data.y
            data = data.to(device)
            e_cls_output, x_recon, stats_pred = model(x, edge_index, batch, num_graphs)
            
            e_cls_outputs_np = e_cls_output.detach().cpu().numpy()  # [num_graphs, hidden_dim]
            
            recon_errors = []
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # Reconstruction error 계산
                node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                node_loss = node_loss.item() * alpha
                
                cls_vec = e_cls_outputs_np[i].reshape(1, -1)
                distances = cdist(cls_vec, cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                # 변환된 값들 저장
                reconstruction_errors_test.append({
                    'reconstruction': node_loss,
                    'clustering': min_distance,
                    'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
                })

                # 전체 에러는 변환된 값들의 평균으로 계산
                recon_error = node_loss + min_distance              
                recon_errors.append(recon_error)
                
                print(f'test_node_loss: {node_loss}')
                print(f'test_min_distance: {min_distance}')
                
                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error
                    
                start_node = end_node
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())
    
    # 시각화를 위한 데이터 변환
    visualization_data = {
        'normal': [
            {'reconstruction': error['reconstruction'], 
             'clustering': error['clustering']}
            for error in reconstruction_errors if error['type'] == 'test_normal'
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'clustering': error['clustering']}
            for error in reconstruction_errors if error['type'] == 'test_anomaly'
        ]
    }

    # # 데이터 분리 및 특징 벡터 구성
    # train_data = np.array([[error['reconstruction'], error['clustering']] 
    #                       for error in reconstruction_errors if error['type'] == 'train_normal'])
    # test_normal = np.array([[error['reconstruction'], error['clustering']] 
    #                        for error in reconstruction_errors_test if error['type'] == 'test_normal'])
    # test_anomaly = np.array([[error['reconstruction'], error['clustering']] 
    #                         for error in reconstruction_errors_test if error['type'] == 'test_anomaly'])
    
    # # 밀도 기반 스코어링 적용
    # # # Scott의 규칙 적용
    # # bandwidth = scott_rule_bandwidth(train_data)
    # bandwidth, _ = loocv_bandwidth_selection(train_data)
    # print(f'bandwidth : {bandwidth}')
    # # LOOCV 적용
    # density_scorer = DensityBasedScoring(bandwidth=bandwidth)
    # density_scorer.fit(train_data)
    
    # # 이상 스코어 계산
    # normal_scores = density_scorer.score_samples(test_normal)
    # anomaly_scores = density_scorer.score_samples(test_anomaly)
    
    # # 전체 스코어 및 라벨 구성
    # all_scores = np.concatenate([normal_scores, anomaly_scores])
    # all_labels = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    
    # 메트릭 계산
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 성능 메트릭 계산
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)
    
    # 최적 임계값 찾기
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (all_scores > optimal_threshold).astype(int)
    
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    
    total_loss_mean = total_loss_ / sum(all_labels == 0)
    total_loss_anomaly_mean = total_loss_anomaly_ / sum(all_labels == 1)

    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, visualization_data, reconstruction_errors


def persistence_stats_loss(pred_stats, true_stats):
    continuous_loss = F.mse_loss(pred_stats[:, :2], true_stats[:, :2])
    
    return continuous_loss


#%%
class PyGTopER:
    def __init__(self, thresholds: List[float]):
        # thresholds를 float32로 변환
        self.thresholds = torch.tensor(sorted(thresholds), dtype=torch.float32)

    
    def _get_graph_structure(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           batch: torch.Tensor, graph_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 모든 텐서를 float32로 변환
        x = x.float()  # float32로 명시적 변환
        
        # CPU로 이동 및 처리
        x = x.cpu()
        edge_index = edge_index.cpu()
        batch = batch.cpu()
        
        mask = batch == graph_idx
        nodes = x[mask]
        
        node_idx = torch.arange(len(batch), dtype=torch.long, device='cpu')[mask]
        idx_map = {int(old): new for new, old in enumerate(node_idx)}
        
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        graph_edges = edge_index[:, edge_mask]
        
        graph_edges = torch.tensor([[idx_map[int(i)] for i in graph_edges[0]],
                                  [idx_map[int(i)] for i in graph_edges[1]]], 
                                 dtype=torch.long,  # long 타입 명시
                                 device='cpu')
        
        return nodes, graph_edges
    
    def _get_node_filtration(self, nodes: torch.Tensor, edges: torch.Tensor, 
                            node_values: torch.Tensor) -> List[Tuple[int, int]]:
        """Compute filtration sequence for a single graph"""
        sequences = []
        for threshold in self.thresholds:
            # Get nodes below threshold
            mask = node_values <= threshold
            if not torch.any(mask):
                sequences.append((0, 0))
                continue
                
            # Get induced edges
            edge_mask = mask[edges[0]] & mask[edges[1]]
            filtered_edges = edges[:, edge_mask]
            
            sequences.append((torch.sum(mask).item(), 
                            filtered_edges.shape[1] // 2))
            
        return sequences

    def _compute_degree_values(self, num_nodes: int, edges: torch.Tensor) -> torch.Tensor:
        degrees = torch.zeros(num_nodes, dtype=torch.float32, device='cpu')  # float32 명시
        unique, counts = torch.unique(edges[0], return_counts=True)
        degrees[unique] += counts.float()  # counts를 float32로 변환
        return degrees
    
    def _compute_popularity_values(self, num_nodes: int, edges: torch.Tensor) -> torch.Tensor:
        """Compute popularity values as defined in the paper"""
        degrees = self._compute_degree_values(num_nodes, edges)
        
        popularity = torch.zeros(num_nodes, device='cpu')
        for i in range(num_nodes):
            neighbors = edges[1][edges[0] == i]
            if len(neighbors) > 0:
                neighbor_degrees = degrees[neighbors]
                popularity[i] = degrees[i] + neighbor_degrees.mean()
            else:
                popularity[i] = degrees[i]
                
        return popularity
    
    def _refine_sequences(self, x_vals: np.ndarray, y_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Refine sequences according to the paper's methodology"""
        refined_x = []
        refined_y = []
        i = 0
        while i < len(y_vals):
            # Find consecutive points with same y value
            j = i + 1
            while j < len(y_vals) and y_vals[j] == y_vals[i]:
                j += 1
            
            # Calculate mean of x values
            x_mean = x_vals[i:j].mean()
            refined_x.append(x_mean)
            refined_y.append(y_vals[i])
            
            i = j
            
        return np.array(refined_x), np.array(refined_y)
    
    def _fit_line(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Fit line to refined sequences with float32 precision"""
        if len(x) < 2:
            return 0.0, 0.0
            
        slope, intercept, _, _, _ = linregress(x.astype(np.float32), y.astype(np.float32))
        return float(intercept), float(slope)  # float32 반환

    def compute_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor, filtration: str = 'degree') -> torch.Tensor:
        device = x.device
        num_graphs = batch.max().item() + 1
        embeddings = []
        
        for graph_idx in range(num_graphs):
            nodes, edges = self._get_graph_structure(x, edge_index, batch, graph_idx)
            
            if filtration == 'degree':
                values = self._compute_degree_values(len(nodes), edges)
            elif filtration == 'popularity':
                values = self._compute_popularity_values(len(nodes), edges)
            else:
                raise ValueError(f"Unknown filtration type: {filtration}")
            
            sequences = self._get_node_filtration(nodes, edges, values)
            x_vals, y_vals = zip(*sequences)
            
            # numpy 배열을 float32로 변환
            x_refined, y_refined = self._refine_sequences(
                np.array(x_vals, dtype=np.float32), 
                np.array(y_vals, dtype=np.float32)
            )
            
            pivot, growth = self._fit_line(x_refined, y_refined)
            embeddings.append([pivot, growth])
            
        # 최종 결과를 float32 텐서로 변환
        return torch.tensor(embeddings, dtype=torch.float32, device=device)

    
def TopER_Embedding(data):
    # TopER 임베딩 계산
    thresholds = np.linspace(0, 5, 20)  # 논문에서 사용한 값 범위로 조정 필요
    toper = PyGTopER(thresholds)
    toper_embeddings = toper.compute_embeddings(data.x, data.edge_index, data.batch)
    
    # 데이터에 TopER 임베딩 추가
    data.toper_embeddings = toper_embeddings
    return data


class DensityBasedScoring:
    def __init__(self, bandwidth=0.5):
        self.kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """
        정상 데이터의 2D 특징(재구성 오차, 클러스터링 거리)에 대해 KDE를 학습
        
        Args:
            X: shape (n_samples, 2) 형태의 array. 
               각 행은 [reconstruction_error, clustering_distance]
        """
        # 특징 정규화
        X_scaled = self.scaler.fit_transform(X)
        # KDE 학습
        self.kde.fit(X_scaled)
        
    def score_samples(self, X):
        """
        샘플들의 밀도 기반 이상 스코어 계산
        """
        X_scaled = self.scaler.transform(X)
        
        # log density 계산
        log_density = self.kde.score_samples(X_scaled)
        
        # -inf 값을 처리
        log_density = np.nan_to_num(log_density, neginf=-10000)
        
        # 이상 스코어 계산 및 클리핑
        anomaly_scores = -log_density  # 더 낮은 밀도 = 더 높은 이상 스코어
        anomaly_scores = np.clip(anomaly_scores, 0, 10000)  # 매우 큰 값 제한
        
        return anomaly_scores


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--n-head-BERT", type=int, default=2)
parser.add_argument("--n-layer-BERT", type=int, default=2)
parser.add_argument("--n-head", type=int, default=2)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--BERT-epochs", type=int, default=100)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=1)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=300)
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

BERT_epochs: int = args.BERT_epochs
epochs: int = args.epochs
n_head_BERT: int = args.n_head_BERT
n_layer_BERT: int = args.n_layer_BERT
n_head: int = args.n_head
n_layer: int = args.n_layer
patience: int = args.patience
n_cluster: int = args.n_cluster
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


# %%
class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        # z를 버퍼 대신 인스턴스 변수로 저장
        self.z = None
        
    def forward(self, x, edge_index):
        # First GCN layer with batch normalization and ELU
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        self.z = x  # 임베딩 저장
        return x


class StructureDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        # 입력 차원을 2배로 설정 (두 노드의 임베딩을 연결하므로)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),  # in_channels*2로 수정
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 디바이스 일치시키기
        device = z.device
        edge_index = edge_index.to(device)
        
        # Get node pairs for prediction
        row, col = edge_index
        
        # 노드 임베딩 연결 (concatenate)
        edge_features = torch.cat([z[row], z[col]], dim=1)  # 곱셈 대신 연결
        
        # Pass through MLP and apply sigmoid
        return torch.sigmoid(self.mlp(edge_features).squeeze())


class DegreeDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze()


class EdgeMasking:
    def __init__(self, p: float = 0.7):
        self.p = p
    
    def __call__(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > self.p
        
        masked_edges = edge_index[:, ~mask]
        visible_edges = edge_index[:, mask]
        
        return masked_edges, visible_edges


class PathMasking:
    def __init__(self, p: float = 0.7, walk_length: int = 3):
        self.p = p
        self.walk_length = walk_length
    
    def random_walk(self, edge_index: torch.Tensor, start_nodes: torch.Tensor) -> torch.Tensor:
        device = edge_index.device  # 현재 디바이스 가져오기
        num_nodes = edge_index.max().item() + 1
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[edge_index[0], edge_index[1]] = 1
        
        masked_edges = []
        current_nodes = start_nodes.to(device)  # start_nodes를 올바른 디바이스로 이동
        
        for _ in range(self.walk_length):
            # Get possible next nodes for each current node
            next_probs = adj[current_nodes]
            next_nodes = torch.multinomial(next_probs, 1).squeeze()
            
            # Add edges to masked set
            new_edges = torch.stack([current_nodes, next_nodes], dim=0)  # dim=0 명시
            masked_edges.append(new_edges)
            
            current_nodes = next_nodes
            
        return torch.cat(masked_edges, dim=1)
    
    def __call__(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device  # 현재 디바이스 가져오기
        num_nodes = edge_index.max().item() + 1
        num_start_nodes = int(num_nodes * self.p)
        
        # Randomly select start nodes (on CPU first, then move to correct device)
        start_nodes = torch.randperm(num_nodes)[:num_start_nodes].to(device)
        
        # Get edges to mask through random walks
        masked_edges = self.random_walk(edge_index, start_nodes)
        
        # Get visible edges (those not in masked_edges)
        edge_mask = ~(edge_index.unsqueeze(-1) == masked_edges.unsqueeze(1)).all(dim=0).any(dim=1)
        visible_edges = edge_index[:, edge_mask]
        
        return masked_edges, visible_edges


# MaskGAE 클래스의 __init__ 부분도 수정
class MaskGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_channels, masking_ratio=0.7, alpha=1e-3, use_path_masking=True):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, embedding_channels)
        self.structure_decoder = StructureDecoder(embedding_channels, hidden_channels)
        self.degree_decoder = DegreeDecoder(embedding_channels, hidden_channels)
        
        if use_path_masking:
            self.masking = PathMasking(masking_ratio)
        else:
            self.masking = EdgeMasking(masking_ratio)
            
        self.alpha = alpha
        # masked_edges를 버퍼 대신 인스턴스 변수로 저장
        self.masked_edges = None

    def save_pretrained(self, path):
        # 저장할 때 state_dict에서 불필요한 키 제외
        state_dict = self.encoder.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.z')}
        torch.save(state_dict, path)
        
    def load_pretrained(self, path):
        # 불러올 때 strict=False로 설정하여 불필요한 키 무시
        self.encoder.load_state_dict(torch.load(path), strict=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        edge_index = edge_index.to(device)
        masked_edges, visible_edges = self.masking(edge_index)
        self.masked_edges = masked_edges
        
        # Encode visible graph
        z = self.encoder(x, visible_edges)
        
        # Decode structure
        edge_pred = self.structure_decoder(z, masked_edges)
        
        # Decode degrees
        deg_pred = self.degree_decoder(z)
        self.z = z  # 노드 임베딩 저장
        
        return z, edge_pred, deg_pred
    
    def loss(
        self,
        edge_pred: torch.Tensor,
        masked_edges: torch.Tensor,
        deg_pred: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        device = edge_pred.device
        
        # Generate negative samples
        num_nodes = edge_index.max().item() + 1
        num_neg_samples = masked_edges.size(1)
        
        # 네거티브 샘플링을 위해 엣지 인덱스를 CPU로 이동
        neg_edges = negative_sampling(
            edge_index=edge_index.cpu(),
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples
        ).to(device)
        
        # Calculate reconstruction loss for positive edges
        pos_loss = F.binary_cross_entropy(
            edge_pred,
            torch.ones(masked_edges.size(1), device=device)
        )
        
        # Calculate reconstruction loss for negative edges
        # GCN encoder에서 저장된 z를 사용
        neg_pred = self.structure_decoder(self.encoder.z, neg_edges)
        
        neg_loss = F.binary_cross_entropy(
            neg_pred,
            torch.zeros(neg_edges.size(1), device=device)
        )
        
        reconstruction_loss = pos_loss + neg_loss
        
        # Calculate degree loss
        true_degrees = degree(
            masked_edges[0],
            num_nodes=num_nodes
        ).to(device)
        
        degree_loss = F.mse_loss(deg_pred, true_degrees)
        
        # Total loss
        total_loss = reconstruction_loss + self.alpha * degree_loss
        
        return total_loss
        
    
#%%
class FeatureDecoder(nn.Module):
    def __init__(self, embed_dim, num_features, dropout_rate=0.1):
        super(FeatureDecoder, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc2 = nn.Linear(embed_dim//2, num_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        z = self.leaky_relu(self.fc1(z))
        z = self.dropout(z)
        z = self.fc2(z)
        return z
    

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, max_nodes):
        super(BilinearEdgeDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        actual_nodes = z.size(0)
        
        z_norm = F.normalize(z, p=2, dim=1) # 각 노드 벡터를 정규화
        cos_sim = torch.mm(z_norm, z_norm.t()) # 코사인 유사도 계산 (내적으로 계산됨)
        adj = self.sigmoid(cos_sim)
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        
        # max_nodes 크기로 패딩
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj


#%%
class BatchUtils:
    @staticmethod
    def process_batch(x, edge_index, batch, num_graphs=None):
        """배치 데이터 전처리를 위한 유틸리티 메서드"""
        batch_size = num_graphs if num_graphs is not None else batch.max().item() + 1
        
        # 그래프별 노드 특징과 엣지 인덱스 분리
        z_list = [x[batch == i] for i in range(batch_size)]
        edge_index_list = []
        start_idx = 0
        
        for i in range(batch_size):
            num_nodes = z_list[i].size(0)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes
            
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list)
        
        return z_list, edge_index_list, max_nodes_in_batch

    @staticmethod
    def add_cls_token(z_list, cls_token, max_nodes_in_batch, device):
        """CLS 토큰 추가 및 패딩 처리"""
        z_with_cls_list = []
        mask_list = []
        
        for z_graph in z_list:
            num_nodes = z_graph.size(0)
            cls_token = cls_token.to(device)
            z_graph = z_graph.unsqueeze(1)
            
            # 패딩
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)
            
            # CLS 토큰 추가
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)
            z_with_cls_list.append(z_with_cls)
            
            # 마스크 생성
            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)
            
        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)
        mask = torch.stack(mask_list).to(device)
        
        return z_with_cls_batch, mask


#%%
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first = True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask):
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output
    

class ClusteringAnalyzer:
    def __init__(self, n_clusters, random_seed):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        
    def perform_clustering_with_analysis(self, train_cls_outputs, epoch):
        """
        K-means 클러스터링 수행 및 분석을 위한 통합 함수
        """
        # CPU로 이동 및 Numpy 배열로 변환
        cls_outputs_np = train_cls_outputs.detach().cpu().numpy()
        
        # k-평균 클러스터링 수행
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(cls_outputs_np)
        
        # Silhouette Score 계산
        if len(np.unique(cluster_labels)) > 1:  # 클러스터가 2개 이상일 때만 계산
            sil_score = silhouette_score(cls_outputs_np, cluster_labels)
            sample_silhouette_values = silhouette_samples(cls_outputs_np, cluster_labels)
        else:
            sil_score = 0
            sample_silhouette_values = np.zeros(len(cluster_labels))
        
        cluster_centers = kmeans.cluster_centers_
        
        clustering_metrics = {
            'silhouette_score': sil_score,
            'sample_silhouette_values': sample_silhouette_values.tolist(),
            'cluster_sizes': [sum(cluster_labels == i) for i in range(self.n_clusters)],
            'n_clusters': self.n_clusters
        }
        
        return cluster_centers, clustering_metrics


def perform_clustering(train_cls_outputs, random_seed, n_clusters, epoch):
    analyzer = ClusteringAnalyzer(n_clusters, random_seed)
    cluster_centers, clustering_metrics = analyzer.perform_clustering_with_analysis(
        train_cls_outputs, epoch)
    
    return cluster_centers, clustering_metrics

            
#%%
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT, nhead, num_layers_BERT, num_layers, dropout_rate=0.1):
        super().__init__()
        # MaskGAE 인코더 사용
        self.encoder = MaskGAE(
            in_channels=num_features,
            hidden_channels=hidden_dims[-1],
            embedding_channels=hidden_dims[-1],
            masking_ratio=0.7,
            alpha=1e-3,
            use_path_masking=True
        )
        
        self.transformer_d = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dims[-1] * 4,
            max_nodes=max_nodes,
            dropout=dropout_rate
        )
        
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.stats_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 5)
        )
        
    def save_pretrained(self, path):
        # 인코더의 state_dict에서 불필요한 키 제외하고 저장
        encoder_state_dict = self.encoder.state_dict()
        filtered_state_dict = {k: v for k, v in encoder_state_dict.items() 
                             if not k.endswith('.z') and 'masked_edges' not in k}
        torch.save(filtered_state_dict, path)
        
    def load_pretrained(self, path):
        # strict=False로 설정하여 불필요한 키 무시하고 로드
        pretrained_dict = torch.load(path)
        self.encoder.load_state_dict(pretrained_dict, strict=False)
        
    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, is_pretrain=False):
        device = x.device
        edge_index = edge_index.to(device)
        if batch is not None:
            batch = batch.to(device)
            
        if is_pretrain:
            # MaskGAE의 프리트레이닝 로직 사용
            # edge_index의 디바이스를 확인하고 필요한 경우 이동
            return self.encoder(x, edge_index)
        
        else:
            # Fine-tuning phase
            # MaskGAE의 임베딩 얻기
            z, _, _ = self.encoder(x, edge_index)
            
            # 배치 처리
            z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(z, edge_index, batch, num_graphs)
            z_with_cls_batch, mask = BatchUtils.add_cls_token(
                z_list, self.cls_token, max_nodes_in_batch, device
            )
            
            # Transformer 처리
            encoded = self.transformer_d(z_with_cls_batch, mask)
            
            # 출력 처리
            cls_output = encoded[:, 0, :]
            node_outputs = [encoded[i, 1:z_list[i].size(0)+1, :] for i in range(num_graphs)]
            u = torch.cat(node_outputs, dim=0)
    
            stats_pred = self.stats_predictor(cls_output)

            # 디코딩
            u_prime = self.u_mlp(u)
            x_recon = self.feature_decoder(u_prime)
            
            return cls_output, x_recon, stats_pred


# %%
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
def run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=None):
    if epoch_results is None:
        epoch_results = {}
    epoch_interval = 10  # 10 에폭 단위로 비교
    
    set_seed(random_seed)    
    all_results = []
    split=splits[trial]
    
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    max_node_label = meta['max_node_label']
    
    # BERT 모델 저장 경로
    bert_save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/BERT_model/Class/pretrained_bert_{dataset_name}_fold{trial}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_gcn{hidden_dims[-1]}_edge_train_try12_.pth'
    
    model = GRAPH_AUTOENCODER(
        num_features=num_features, 
        hidden_dims=hidden_dims, 
        max_nodes=max_nodes,
        nhead_BERT=n_head_BERT,
        nhead=n_head,
        num_layers_BERT=n_layer_BERT,
        num_layers=n_layer,
        dropout_rate=dropout_rate
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []
    
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        model.load_pretrained(bert_save_path)  # 수정된 로드 함수 사용
    else:
        print("Training BERT from scratch...")
        # 1단계: BERT 임베딩 학습
        print("Stage 1: Training BERT embeddings...")

        pretrain_params = list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=learning_rate)
        
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample = pretrain_graph_bert(
                model, train_loader, bert_optimizer, device
            )
            
            if epoch % log_interval == 0:
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}')
        
        # 학습된 BERT 저장
        print("Saving pretrained BERT...")
        model.save_pretrained(bert_save_path)  # 수정된 저장 함수 사용        

    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cls_outputs, train_errors = train(model, train_loader, recon_optimizer, device, epoch)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            cluster_centers, clustering_metrics = perform_clustering(
                train_cls_outputs, random_seed, n_clusters=n_cluster, 
                epoch=epoch
            )

            # auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, visualization_data = evaluate_model(model, test_loader, max_nodes, cluster_centers, device)
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, visualization_data, test_errors = evaluate_model(model, test_loader, cluster_centers, n_cluster, gamma_cluster, random_seed, train_errors, device)
            
            save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}/error_distribution_epoch_{epoch}_fold_{trial}.json'
            with open(save_path, 'w') as f:
                json.dump(visualization_data, f)
            
            all_results.append((auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly))
            print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation loss anomaly = {test_loss_anomaly:.4f}, Validation AUC = {auroc:.4f}, Validation AUPRC = {auprc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
            
            info_test = 'AD_AUC:{:.4f}, AD_AUPRC:{:.4f}, Test_Loss:{:.4f}, Test_Loss_Anomaly:{:.4f}'.format(auroc, auprc, test_loss, test_loss_anomaly)
            print(info_train + '   ' + info_test)

            # 10 에폭 단위일 때 결과 저장
            if epoch % epoch_interval == 0:
                if epoch not in epoch_results:
                    epoch_results[epoch] = {'aurocs': [], 'auprcs': [], 'precisions': [], 'recalls': [], 'f1s': []}
                
                epoch_results[epoch]['aurocs'].append(auroc)
                epoch_results[epoch]['auprcs'].append(auprc)
                epoch_results[epoch]['precisions'].append(precision)
                epoch_results[epoch]['recalls'].append(recall)
                epoch_results[epoch]['f1s'].append(f1)

            # # run() 함수 내에서 평가 시점에 다음 코드 추가
            # cls_embeddings_pca, explained_variance_ratio, eigenvalues = analyze_cls_embeddings(model, test_loader, epoch, device)

    return auroc, epoch_results


#%%
'''MAIN'''
if __name__ == '__main__':
    ad_aucs = []
    fold_times = []
    epoch_results = {}  # 모든 폴드의 에폭별 결과를 저장
    splits = get_ad_split_TU(dataset_name, n_cross_val)    
    start_time = time.time()  # 전체 실행 시작 시간

    for trial in range(n_cross_val):
        fold_start = time.time()  # 현재 폴드 시작 시간
        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc, epoch_results = run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=epoch_results)
        ad_aucs.append(ad_auc)
        
        fold_end = time.time()  # 현재 폴드 종료 시간   
        fold_duration = fold_end - fold_start  # 현재 폴드 실행 시간
        fold_times.append(fold_duration)
        
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds.")
    
    # 각 에폭별 평균 성능 계산
    epoch_means = {}
    epoch_stds = {}
    for epoch in epoch_results.keys():
        epoch_means[epoch] = {
            'auroc': np.mean(epoch_results[epoch]['aurocs']),
            'auprc': np.mean(epoch_results[epoch]['auprcs']),
            'precision': np.mean(epoch_results[epoch]['precisions']),
            'recall': np.mean(epoch_results[epoch]['recalls']),
            'f1': np.mean(epoch_results[epoch]['f1s'])
        }
        epoch_stds[epoch] = {
            'auroc': np.std(epoch_results[epoch]['aurocs']),
            'auprc': np.std(epoch_results[epoch]['auprcs']),
            'precision': np.std(epoch_results[epoch]['precisions']),
            'recall': np.std(epoch_results[epoch]['recalls']),
            'f1': np.std(epoch_results[epoch]['f1s'])
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
    print(f"F1 = {epoch_means[best_epoch]['f1']:.4f} ± {epoch_stds[best_epoch]['f1']:.4f}")
    
    # 최종 결과 저장
    total_time = time.time() - start_time
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # 모든 결과를 JSON으로 저장
    results_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/cross_val_results/epoch_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'epoch_means': epoch_means,
            'epoch_stds': epoch_stds,
            'best_epoch': int(best_epoch),
            'final_auroc_mean': float(np.mean(ad_aucs)),
            'final_auroc_std': float(np.std(ad_aucs))
        }, f, indent=4)


# %%
def visualize_molecular_graph(x, edge_index, batch, idx=0):
    """
    분자 그래프를 시각화하는 함수
    
    Parameters:
    x: 노드 특성 텐서
    edge_index: 엣지 인덱스 텐서
    batch: 배치 할당 텐서
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
    
    # 시각화
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, 
            node_color='lightblue',
            node_size=500,
            with_labels=True,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            width=2,
            alpha=0.7)
    plt.title('Molecular Graph Visualization')
    plt.axis('off')
    plt.show()


#%%
visualize_molecular_graph(x, edge_index, batch, idx=100)  # 첫 번째 분자 시각화

# %%
visualize_molecular_graph(x, edge_index, batch, idx=5)  # 첫 번째 분자 시각화

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

# %%
visualize_molecular_graph(x, edge_index, batch, y, idx=1)

# %%
visualize_batch_examples(x, edge_index, batch, y, num_examples=32)

# %%
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 예시 데이터 생성 (4개의 점)
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# 거리 행렬 계산
distance_matrix = squareform(pdist(points, metric='euclidean'))

# Rips Complex 생성 및 Simplex Tree 계산
rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=2.0)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# Persistent Homology 계산
simplex_tree.compute_persistence()

# Persistence Diagram 얻기
persistence_diagram = simplex_tree.persistence()

# 시각화
gd.plot_persistence_diagram(persistence_diagram)
plt.show()

# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw


# 분자 그래프를 예시로 생성 (아스피린을 모델링한 예시)
# 아스피린의 SMILES 문자열: "CC(=O)OC1=CC=CC=C1C(=O)O"
# 이를 기반으로 단순화된 분자 그래프를 만듭니다.

# 원자 번호와 결합을 나타내는 데이터 (간단한 예시)
nodes = ['C1', 'C2', 'C3', 'C4', 'C5', 'O1', 'O2']
edges = [('C1', 'C2'), ('C2', 'C3'), ('C3', 'C4'), ('C4', 'C5'), 
         ('C5', 'O1'), ('O1', 'C1'), ('C3', 'O2')]

# 그래프 생성
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# 그래프 시각화
pos = nx.spring_layout(G)  # 자동 배치
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', edge_color='gray')
plt.title("Molecular Graph", fontsize=16)
plt.show()
#%%

# 실제 분자 구조인 아스피린 (Aspirin)
aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
aspirin_mol = Chem.MolFromSmiles(aspirin_smiles)

# 2D 구조 그리기
Draw.MolToImage(aspirin_mol, size=(300, 300))
#%%
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 1. 분자 구조 가져오기 (예: 아스피린)
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # 아스피린의 SMILES
AllChem.Compute2DCoords(mol)  # 2D 좌표 계산

# 2. 원자 좌표 추출 (2D 좌표)
coords = np.array([atom.GetIdx() for atom in mol.GetAtoms()])
atom_positions = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

# 3. 거리 행렬 계산
distance_matrix = squareform(pdist(atom_positions, metric='euclidean'))

# 4. Rips Complex 생성 및 Simplex Tree 계산
rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=2.0)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# 5. Persistent Homology 계산
simplex_tree.compute_persistence()

# 6. Persistence Diagram 얻기
persistence_diagram = simplex_tree.persistence()

# 7. 시각화
gd.plot_persistence_diagram(persistence_diagram)
plt.show()

# %%
