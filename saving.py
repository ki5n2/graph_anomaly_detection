#%%
'''IMPORTS'''
import os
import re
import sys
import math
import wandb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.utils as utils

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from functools import partial
from scipy.linalg import eigh
from multiprocessing import Pool

from modules.loss import loss_cal
from modules.utils import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original

import networkx as nx

#%%
'''TRAIN'''
def train(model, train_loader, optimizer, max_nodes, device):
    model.train()
    total_loss = 0
    num_sample = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

        adj = adj_original(edge_index, batch, max_nodes)
        x_recon, adj_recon_list, train_cls_outputs, z_ = model(x, edge_index, batch, num_graphs)
        
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(x_recon[start_node:end_node] - x[start_node:end_node], p='fro')**2 / num_nodes
            node_loss = node_loss * node_theta
            print(f'train_node loss: {node_loss}')
            
            adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / num_nodes
            adj_loss = adj_loss * adj_theta
            print(f'train_adj_loss: {adj_loss}')
            
            # z_node_loss = torch.norm(z_tilde[start_node:end_node] - z_[start_node:end_node], p='fro')**2 / num_nodes
            # z_node_loss = z_node_loss * 0.3
            # print(f'train_z_node loss: {z_node_loss}')
            
            loss += node_loss + adj_loss

            start_node = end_node

        num_sample += num_graphs

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), num_sample, train_cls_outputs.detach().cpu()


#%%
'''EVALUATION'''
def evaluate_model(model, test_loader, max_nodes, cluster_centers, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    total_loss_mean = 0
    total_loss_anomaly_mean = 0

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

            adj = adj_original(edge_index, batch, max_nodes)

            x_recon, adj_recon_list, e_cls_output, z_ = model(x, edge_index, batch, num_graphs)

            recon_errors = []
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes

                node_loss = (torch.norm(x_recon[start_node:end_node] - x[start_node:end_node], p='fro')**2) / num_nodes
                
                adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / num_nodes
                
                # cls_vec = e_cls_output[i].cpu().numpy()  # [hidden_dim]
                cls_vec = e_cls_output[i].detach().cpu().numpy()  # [hidden_dim]
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')  # [1, n_clusters]
                min_distance = distances.min()  # 가장 가까운 클러스터까지의 거리

                # recon_error = node_loss.item() * 0.1 + adj_loss.item() * 1 + min_distance * 0.5
                recon_error = (node_loss.item() * alpha) + (adj_loss.item() * beta) + (min_distance * gamma)
                recon_errors.append(recon_error.item())
        
                print(f'test_node_loss: {node_loss.item() * alpha}')
                print(f'test_adj_loss: {adj_loss.item() * beta}')
                print(f'test_min_distance: {min_distance * gamma}')

                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error

                start_node = end_node

            total_loss = total_loss_ / sum(data.y == 0)
            total_loss_anomaly = total_loss_anomaly_ / sum(data.y == 1)
            
            total_loss_mean += total_loss
            total_loss_anomaly_mean += total_loss_anomaly
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Compute metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (all_scores > optimal_threshold).astype(int)

    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)

    return auroc, auprc, precision, recall, f1, total_loss_mean / len(test_loader), total_loss_anomaly_mean / len(test_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='DHFR')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=3)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.001)

parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=0.025)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--node-theta", type=float, default=0.3)
parser.add_argument("--adj-theta", type=int, default=0.05)

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

wandb.init(project="graph anomaly detection", entity="ki5n2")

wandb.config.update(args)

wandb.config = {
  "random_seed": random_seed,
  "learning_rate": learning_rate,
  "epochs": epochs
}


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, improved=True, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.conv.lin.weight, gain=gain)
        nn.init.zeros_(self.conv.bias)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight, gain=1.0)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        
        # 정규화 트릭 적용
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.size(0))
        deg = utils.degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        
        # 정규화된 인접 행렬을 사용하여 합성곱 적용
        x = self.conv(x, edge_index, norm)
        
        x = F.relu(self.bn(x))
        x = self.dropout(x)
        return F.relu(x + residual)

    # def forward(self, x, edge_index):
    #     residual = self.shortcut(x)
    #     x = self.conv(x, edge_index)  # GCNConv에서 정규화 트릭 적용
    #     x = F.relu(self.bn(x))
    #     x = self.dropout(x)
    #     return F.relu(x + residual)
    

class Encoder(nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()
        dims = [num_features] + hidden_dims
        for i in range(len(dims) - 1):
            self.blocks.append(ResidualBlock(dims[i], dims[i+1], dropout_rate))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        for block in self.blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)


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
        
        adj = torch.mm(z, z.t())
        adj = self.sigmoid(adj)
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj


class GraphBertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        
        # WSP와 LE 각각에 d_model/2 차원을 할당
        self.wsp_encoder = nn.Linear(max_nodes, d_model // 2)
        self.le_encoder = nn.Linear(max_nodes, d_model // 2)
        
    def get_wsp_encoding(self, edge_index, num_nodes):
        # Weighted Shortest Path 계산
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = list(zip(edge_index_np[0], edge_index_np[1]))
        G.add_edges_from(edges)
        
        spl_matrix = torch.zeros((num_nodes, self.max_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(G, source=i, target=j)
                    except nx.NetworkXNoPath:
                        path_length = self.max_nodes  # 연결되지 않은 경우 최대 거리 할당
                    spl_matrix[i, j] = path_length

        return spl_matrix.to(edge_index.device)
    
    # def get_wsp_encoding(self, edge_index, num_nodes):
    #     # Weighted Shortest Path 계산
    #     edge_index_np = edge_index.cpu().numpy()
    #     G = nx.Graph()
    #     G.add_nodes_from(range(num_nodes))
    #     edges = list(zip(edge_index_np[0], edge_index_np[1]))
    #     G.add_edges_from(edges)
        
    #     spl_matrix = torch.full((num_nodes, self.max_nodes), self.max_nodes)
        
    #     # 모든 쌍의 최단 경로를 한 번에 계산
    #     lengths = dict(nx.all_pairs_shortest_path_length(G))
        
    #     for i in lengths:
    #         for j, length in lengths[i].items():
    #             spl_matrix[i, j] = length
        
    #     wsp_matrix = spl_matrix.to(edge_index.device)
    #     wsp_matrix = wsp_matrix.float()
        
    #     return wsp_matrix
    
    def get_laplacian_encoding(self, edge_index, num_nodes):
        # Laplacian Eigenvector 계산
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', 
                                            num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(edge_index, edge_weight, 
                                (num_nodes, num_nodes)).to_dense()
        
        # CUDA 텐서를 CPU로 이동 후 NumPy로 변환
        L_np = L.cpu().numpy()
        eigenvals, eigenvecs = eigh(L_np)
        
        # 결과를 다시 텐서로 변환하고 원래 디바이스로 이동
        le_matrix = torch.from_numpy(eigenvecs).float().to(edge_index.device)
        
        padded_le = torch.zeros((num_nodes, self.max_nodes), device=edge_index.device)
        padded_le[:, :num_nodes] = le_matrix
        
        return padded_le
    
    def forward(self, edge_index, num_nodes):
        # WSP 인코딩
        wsp_matrix = self.get_wsp_encoding(edge_index, num_nodes)
        wsp_encoding = self.wsp_encoder(wsp_matrix)
        
        # LE 인코딩
        le_matrix = self.get_laplacian_encoding(edge_index, num_nodes)
        le_encoding = self.le_encoder(le_matrix)
        
        # WSP와 LE 결합
        pos_encoding = torch.cat([wsp_encoding, le_encoding], dim=-1)
        
        return pos_encoding


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = GraphBertPositionalEncoding(d_model, max_nodes)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model


    def forward(self, src, edge_index_list, src_key_padding_mask):
        batch_size = src.size(0)
        max_seq_len = src.size(1)
        
        # 각 그래프에 대해 포지셔널 인코딩 계산
        pos_encodings = []
        for i in range(batch_size):
            # CLS 토큰을 위한 더미 인코딩
            cls_pos_encoding = torch.zeros(1, self.d_model).to(src.device)
            
            # 실제 노드들의 포지셔널 인코딩
            num_nodes = (~src_key_padding_mask[i][1:]).sum().item()
            
            # 문제 발생 위치
            if num_nodes > 0:
                graph_pos_encoding = self.positional_encoding( 
                    edge_index_list[i], num_nodes
                )
                # 패딩
                padded_pos_encoding = F.pad(
                    graph_pos_encoding, 
                    (0, 0, 0, max_seq_len - num_nodes - 1), 
                    'constant', 0
                )
            else:
                padded_pos_encoding = torch.zeros(max_seq_len - 1, self.d_model).to(src.device)
            
            # CLS 토큰 인코딩과 노드 인코딩 결합
            full_pos_encoding = torch.cat([cls_pos_encoding, padded_pos_encoding], dim=0)
            pos_encodings.append(full_pos_encoding)
        
        # 모든 배치의 포지셔널 인코딩 결합
        pos_encoding_batch = torch.stack(pos_encodings)
        
        # 포지셔널 인코딩 추가
        src_ = src + pos_encoding_batch
        
        # 트랜스포머 인코딩
        src_ = src_.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        src_key_padding_mask_ = src_key_padding_mask.transpose(0, 1)
        output = self.transformer_encoder(src_, src_key_padding_mask=src_key_padding_mask_)
        output = output.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        return output


class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(ImprovedTransformerEncoder, self).__init__()
        self.positional_encoding = GraphBertPositionalEncoding(d_model, max_nodes)
        
        self.layers = nn.ModuleList([
            ImprovedTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, edge_index_list, src_key_padding_mask):
        batch_size = src.size(0)
        max_seq_len = src.size(1)
        
        # Calculate positional encodings (same as before)
        pos_encodings = []
        for i in range(batch_size):
            cls_pos_encoding = torch.zeros(1, self.d_model).to(src.device)
            num_nodes = (~src_key_padding_mask[i][1:]).sum().item()
            
            if num_nodes > 0:
                graph_pos_encoding = self.positional_encoding(edge_index_list[i], num_nodes)
                padded_pos_encoding = F.pad(
                    graph_pos_encoding, 
                    (0, 0, 0, max_seq_len - num_nodes - 1), 
                    'constant', 0
                )
            else:
                padded_pos_encoding = torch.zeros(max_seq_len - 1, self.d_model).to(src.device)
            
            full_pos_encoding = torch.cat([cls_pos_encoding, padded_pos_encoding], dim=0)
            pos_encodings.append(full_pos_encoding)
        
        pos_encoding_batch = torch.stack(pos_encodings)
        
        # Add positional encoding
        src = src + pos_encoding_batch
        
        # Apply transformer layers
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        
        # Apply final layer normalization
        output = self.norm(src)
        
        return output


class ImprovedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(ImprovedTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Two-layer feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Using GELU instead of ReLU

    def forward(self, src, src_key_padding_mask):
        # Multi-head self-attention
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src
    
def perform_clustering(train_cls_outputs, random_seed, n_clusters):
    # train_cls_outputs가 이미 텐서이므로, 그대로 사용
    cls_outputs_tensor = train_cls_outputs  # [total_num_graphs, hidden_dim]
    cls_outputs_np = cls_outputs_tensor.detach().cpu().numpy()

    # K-Means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto").fit(cls_outputs_np)

    # 클러스터 중심 저장
    cluster_centers = kmeans.cluster_centers_

    return kmeans, cluster_centers


#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder= Encoder(num_features, hidden_dims, dropout_rate)
        self.transformer = ImprovedTransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=16,  # Increased from 8 to 16
            num_layers=6,  # Increased from 4 to 6
            dim_feedforward=hidden_dims[-1] * 8,  # Increased from 4 to 8
            max_nodes=max_nodes,
            dropout=dropout_rate
        )
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.edge_decoder = BilinearEdgeDecoder(max_nodes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
        
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, edge_index, batch, num_graphs):
        z_ = self.encoder(x, edge_index)
        z = self.dropout(z_)

        # 그래프별로 노드 임베딩과 엣지 인덱스를 분리
        z_list = [z[batch == i] for i in range(num_graphs)]
        edge_index_list = []
        start_idx = 0
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            mask = (batch == i)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes

        # 각 그래프에 대해 CLS 토큰 추가 및 패딩
        z_with_cls_list = []
        mask_list = []
        # 최대 노드 수 계산
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list)
        
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            cls_token = self.cls_token.repeat(1, 1, 1)  # [1, 1, hidden_dim]
            cls_token = cls_token.to(device)
            z_graph = z_list[i].unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            
            # 패딩
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)  # [max_nodes, 1, hidden_dim]
            
            # CLS 토큰 추가
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)  # [1, max_nodes+1, hidden_dim]
            z_with_cls_list.append(z_with_cls)

            # 마스크 생성 (True: 패딩된 위치, False: 유효한 위치)
            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)

        # 배치로 결합
        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)  # [batch_size, max_nodes+1, hidden_dim]
        mask = torch.stack(mask_list).to(z.device)  # [batch_size, max_nodes+1]

        encoded = self.transformer(z_with_cls_batch, edge_index_list, mask)

        # CLS 토큰과 노드 임베딩 분리
        cls_output = encoded[:, 0, :]       # [batch_size, hidden_dim]
        node_output = encoded[:, 1:, :]     # [batch_size, max_nodes, hidden_dim]

        # 패딩된 노드 무시하고 노드 임베딩 추출
        node_output_list = []
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            node_output_list.append(node_output[i, :num_nodes, :])

        # 노드 임베딩을 이어붙임
        u = torch.cat(node_output_list, dim=0)  # [total_num_nodes, hidden_dim]
        # node_output_concat = torch.cat(node_output_list, dim=0)

        # u에 MLP 적용하여 u' 생성
        u_prime = self.u_mlp(u)
        
        # 노드 특성 재구성: u'를 사용
        x_recon = self.feature_decoder(u_prime)
        # x_recon = self.feature_decoder(node_output_concat)
                
        # 인접행렬 재구성
        adj_recon_list = []
        idx = 0
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            z_graph = u_prime[idx:idx + num_nodes]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)
            idx += num_nodes
        
        # new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch).to(device)
        # z_tilde = self.encoder(x_recon, new_edge_index)
        
        return x_recon, adj_recon_list, cls_output, z_

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.zeros_(module.bias)

    def get_edge_index_from_adj_list(self, adj_recon_list, batch, threshold=0.5):
        edge_index_list = []
        start_idx = 0
        for i, adj in enumerate(adj_recon_list):
            num_nodes = (batch == i).sum().item()
            adj_binary = (adj[:num_nodes, :num_nodes] > threshold).float()
            edge_index = adj_binary.nonzero().t() + start_idx
            edge_index_list.append(edge_index)
            start_idx += num_nodes
        return torch.cat(edge_index_list, dim=1)
    

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


#%%
'''MODEL AND OPTIMIZER DEFINE'''
model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)


# %%
def run(dataset_name, random_seed, dataset_AN, split=None, device=device):
    all_results = []
    set_seed(random_seed)

    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']

    model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    train_loader = loaders['train']
    test_loader = loaders['test']

    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []

    for epoch in range(1, epochs+1):
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, optimizer, max_nodes, device)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            kmeans, cluster_centers = perform_clustering(train_cls_outputs, random_seed, n_clusters=n_cluster)
            
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = evaluate_model(model, test_loader, max_nodes, cluster_centers, device)
            
            scheduler.step(train_loss)
            
            all_results.append((auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly))
            print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation loss anomaly = {test_loss_anomaly:.4f}, Validation AUC = {auroc:.4f}, Validation AUPRC = {auprc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": test_loss,
                "val_loss_anomaly": test_loss_anomaly,
                "auroc": auroc,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            info_test = 'AD_AUC:{:.4f}, AD_AUPRC:{:.4f}, Test_Loss:{:.4f}, Test_Loss_Anomaly:{:.4f}'.format(auroc, auprc, test_loss, test_loss_anomaly)

            print(info_train + '   ' + info_test)

    return auroc


#%%
if __name__ == '__main__':
    ad_aucs = []
    splits = get_ad_split_TU(dataset_name, n_cross_val)    

    for trial in range(n_cross_val):
        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc = run(dataset_name, random_seed, dataset_AN, split=splits[trial])
        ad_aucs.append(ad_auc)

    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print(len(ad_aucs))

    print('[FINAL RESULTS] ' + results)
    
wandb.finish()


# %%
