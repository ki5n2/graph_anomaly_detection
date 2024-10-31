#%%
'''IMPORTS'''
import os
import re
import sys
import math
import time
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
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from functools import partial
from multiprocessing import Pool

from module.loss import loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, batch_nodes_subgraphs

from torch_geometric.utils import to_networkx, get_laplacian
from scipy.linalg import eigh
import networkx as nx

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score


#%%
'''TRAIN BERT'''
def train_bert_embedding(model, train_loader, bert_optimizer, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label
        node_label = torch.round(node_label).long()

        # 마스크 생성
        mask_indices = torch.rand(x.size(0), device=device) < 0.15  # 15% 노드 마스킹
        
        # BERT 인코딩 및 마스크 토큰 예측만 수행
        _, z, _, masked_outputs = model(
            x, edge_index, batch, num_graphs, mask_indices, training=True
        )
        
        # 마스크 예측 손실만 계산
        # mask_loss = F.mse_loss(masked_outputs, x[mask_indices])
        mask_loss = torch.norm(masked_outputs - x[mask_indices], p='fro')**2 / mask_indices.sum()
        # mask_loss_ = F.cross_entropy(masked_outputs_, node_label[mask_indices])
        loss = mask_loss
        print(f'mask_node_feature:{mask_loss}')
        # print(f'mask_label:{mask_loss_}')
        
        loss.backward()
        bert_optimizer.step()
        total_loss += loss.item()
        num_sample += num_graphs
    
    return total_loss / len(train_loader), num_sample, z.detach().cpu()


#%%
def train(model, train_loader, recon_optimizer, max_nodes, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label
        
        train_cls_outputs, z, z_ = model(x, edge_index, batch, num_graphs)
        
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(z[start_node:end_node] - z_[start_node:end_node], p='fro')**2 / num_nodes
            
            loss += node_loss

            start_node = end_node
        
        print(f'train_node_loss: {loss}')
        
        num_sample += num_graphs

        loss.backward()
        recon_optimizer.step()
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
            x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label

            e_cls_output, z, z_ = model(x, edge_index, batch, num_graphs)

            recon_errors = []
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                node_loss = torch.norm(z[start_node:end_node] - z_[start_node:end_node], p='fro')**2 / num_nodes
                
                # cls_vec = e_cls_output[i].cpu().numpy()  # [hidden_dim]
                cls_vec = e_cls_output[i].detach().cpu().numpy()  # [hidden_dim]
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')  # [1, n_clusters]
                min_distance = distances.min()

                recon_error = (node_loss.item() * alpha) + (min_distance.item() * gamma)              
                recon_errors.append(recon_error)
                
                print(f'test_node_loss: {node_loss.item() * alpha}')
                print(f'test_min_distance: {min_distance.item() * gamma}')

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


def cluster_center_compactness_loss(cluster_centers):
    """
    cluster_centers : tensor, shape (num_clusters, feature_dim)
        - num_clusters: 클러스터의 개수 (e.g., 100)
        - feature_dim: 각 클러스터 중심 벡터의 차원
    
    Returns:
    --------
    loss : tensor
        클러스터 중심들이 가까워지도록 하는 손실 값
    """
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32, requires_grad=True)
    num_clusters = cluster_centers.size(0)
    # num_clusters = len(cluster_centers).size(0)

    # 클러스터 중심 쌍의 유클리드 거리 계산

    distances = torch.cdist(cluster_centers, cluster_centers, p=2)  # shape (num_clusters, num_clusters)

    # 자기 자신과의 거리는 제외하고 모든 거리를 합산
    mask = torch.ones_like(distances, dtype=torch.bool)
    mask.fill_diagonal_(0)  # 대각선 원소를 0으로 설정하여 자기 자신과의 거리 제외
    compactness_loss = distances[mask].mean()  # 평균을 통해 최종 손실 산출
    
    return compactness_loss


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--n-head", type=int, default=2)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--BERT-epochs", type=int, default=100)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=3)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=10.0)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.1)
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
'''MODEL CONSTRUCTION'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, negative_slope=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, improved=True, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.negative_slope)
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
        x = self.activation(self.bn(x))
        x = self.dropout(x)
        
        return self.activation(x + residual)
    

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
    
    
#%%
class BertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, d_model, nhead, num_layers, max_nodes, num_node_classes, dropout_rate=0.1):
        super(BertEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dims[-1], nhead, hidden_dims[-1] * 4, dropout_rate, activation='gelu'
        ) 
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dims[-1]))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # self.predicter = nn.Linear(hidden_dims[-1], num_features)
        # self.predicter = nn.Sequential(
        #     nn.Linear(hidden_dims[-1], hidden_dims[-1]),
        #     nn.LeakyReLU(negative_slope=0.1),  # 음수 값도 처리 가능
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dims[-1], num_features)
        # )
        
        self.predicter = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.LayerNorm(hidden_dims[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_features)  # 마지막 레이어에는 활성화 함수 없음
        )
        
        # self.classifier = nn.Linear(num_features, num_node_classes)
        # Classifier - LeakyReLU 사용
        # self.classifier = nn.Sequential(
        #     nn.Linear(num_features, hidden_dims[-1]),
        #     nn.LeakyReLU(negative_slope=0.1),  # 음수 값도 처리 가능
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dims[-1], num_node_classes)
        # )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.d_model = d_model

    def forward(self, h, batch, mask_indices=None, training=True):
        batch_size = batch.max().item() + 1
        
        # 배치 내 모든 그래프를 동시에 처리하기 위한 패딩된 텐서 생성
        padded_tensor = torch.zeros(batch_size, self.max_nodes, h.size(-1), device=h.device)
        padding_mask = torch.ones(batch_size, self.max_nodes, dtype=torch.bool, device=h.device)
        
        # 마스크된 위치 추적을 위한 텐서
        if training and mask_indices is not None:
            mask_positions = torch.zeros_like(padding_mask)
        
        # 각 그래프의 노드를 패딩된 텐서에 할당
        for i in range(batch_size):
            # i번째 그래프
            mask = (batch == i)
            num_nodes = mask.sum().item()
            graph_h = h[mask]
            
            if training and mask_indices is not None:
                # i번째 그래프에 대한 마스크 유무
                graph_mask_indices = mask_indices[mask]
                # mask_token = mask_token.to(device)
                # i번째 그래프에 대한 마스크 적용
                graph_h[graph_mask_indices] = self.mask_token
                # 마스크된 위치 추적을 위한 텐서 값 변경
                mask_positions[i, :num_nodes][graph_mask_indices] = True
            
            padded_tensor[i, :num_nodes] = graph_h
            padding_mask[i, :num_nodes] = False
        
        # 전체 배치를 한 번에 트랜스포머에 통과
        transformed = self.transformer(
            padded_tensor.transpose(0, 1),  # [max_nodes, batch_size, hidden_dim]
            src_key_padding_mask=padding_mask
        ).transpose(0, 1)  # [batch_size, max_nodes, hidden_dim]
        
        # 결과 처리
        node_embeddings = []
        outputs = []
        # outputs_ = []
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            graph_encoded = transformed[i, :num_nodes]  # 패딩 제거
            node_embeddings.append(graph_encoded)
            
            if training and mask_indices is not None:
                # 마스크된 위치의 노드만 예측
                mask_pos = mask_positions[i, :num_nodes]
                if mask_pos.any():
                    masked_output = self.predicter(graph_encoded[mask_pos])
                    outputs.append(masked_output)
                    # masked_output_ = self.classifier(masked_output)
                    # outputs_.append(masked_output_)
        
        node_embeddings = torch.cat(node_embeddings, dim=0)
        
        if training and mask_indices is not None and outputs:
            outputs = torch.cat(outputs, dim=0)
            # outputs_ = torch.cat(outputs_, dim=0)
            return node_embeddings, outputs
        
        return node_embeddings
    

#%%
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu'
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
        
        src_ = src_.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        # src_key_padding_mask_ = src_key_padding_mask.transpose(0, 1)
        output_ = self.transformer_encoder(src_, src_key_padding_mask=src_key_padding_mask)
        output = output_.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]

        return output
    
    
#%%
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
    def __init__(self, num_features, hidden_dims, max_nodes, nhead, num_layers, num_node_labels, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.gcn_encoder = Encoder(num_features, hidden_dims, dropout_rate)
        # BERT 인코더로 변경
        self.encoder = BertEncoder(
            num_features=num_features,
            hidden_dims=hidden_dims,
            nhead=n_head,
            num_layers=n_layer,
            max_nodes=max_nodes,
            num_node_classes=num_node_labels,
            dropout_rate=dropout_rate
        )
        self.transformer = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=n_head,
            num_layers=n_layer,
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
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
        
        # 가중치 초기화
        self.apply(self._init_weights)


    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=True):
        h = self.gcn_encoder(x, edge_index)
        h = self.dropout(h)
        
        # BERT 인코딩
        if training and mask_indices is not None:
            z, masked_outputs = self.encoder(
                h, batch, mask_indices, training=True
            )
        else:
            z = self.encoder(
                h, batch, training=False
            )
        
        print(mask_indices)
        print(training)
        
        z_list = [z[batch == i] for i in range(num_graphs)] # 그래프 별 z 저장 (batch_size, num nodes, feature dim)
        edge_index_list = [] # 그래프 별 엣지 인덱스 저장 (batch_size), edge_index_list[0] = (2 x m), m is # of edges
        start_idx = 0
        
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            mask = (batch == i)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes
        
        z_with_cls_list = []
        mask_list = []
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list) # 배치 내 최대 노드 수
        
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            cls_token = self.cls_token.repeat(1, 1, 1)  # [1, 1, hidden_dim]
            cls_token = cls_token.to(device)
            z_graph = z_list[i].unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)  # [max_nodes, 1, hidden_dim] -> 나머지는 패딩
            
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)  # [1, max_nodes+1, hidden_dim] -> CLS 추가
            z_with_cls_list.append(z_with_cls)

            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)

        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)  # [batch_size, max_nodes+1, hidden_dim] -> 모든 그래프에 대한 CLS 추가
        mask = torch.stack(mask_list).to(z.device)  # [batch_size, max_nodes+1]

        encoded = self.transformer(z_with_cls_batch, edge_index_list, mask)

        cls_output = encoded[:, 0, :]       # [batch_size, hidden_dim]
        node_output = encoded[:, 1:, :]     # [batch_size, max_nodes, hidden_dim]
        
        node_output_list = []
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            node_output_list.append(node_output[i, :num_nodes, :])

        u = torch.cat(node_output_list, dim=0)  # [total_num_nodes, hidden_dim]

        u_prime = self.u_mlp(u)
        
        x_recon = self.feature_decoder(u_prime)
        
        h_ = self.gcn_encoder(x_recon, edge_index)
        h_ = self.dropout(h_)
        
        z_ = self.encoder(h_, batch, training=False)
        
        if training and mask_indices is not None:
            return cls_output, z, z_, masked_outputs
        
        return cls_output, z, z_

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


# %%
'''RUN'''
def run(dataset_name, random_seed, dataset_AN, trial, device=device):
    all_results = []
    set_seed(random_seed)
    split=splits[trial]
    
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    max_node_label = meta['max_node_label']
    
    # BERT 모델 저장 경로
    bert_save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/BERT_model/pretrained_bert_{dataset_name}_fold{trial}_nhead{n_head}_seed{random_seed}_BERT_epochs{BERT_epochs}_try0.pth'
    
    model = GRAPH_AUTOENCODER(
        num_features=num_features, 
        hidden_dims=hidden_dims, 
        max_nodes=max_nodes,
        nhead=n_head,
        num_layers=n_layer,
        num_node_labels=max_node_label,
        dropout_rate=dropout_rate
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []
    
    # 1단계: BERT 임베딩 학습
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        # BERT 인코더의 가중치만 로드
        model.encoder.load_state_dict(torch.load(bert_save_path))
    else:
        print("Training BERT from scratch...")
        # 1단계: BERT 임베딩 학습
        print("Stage 1: Training BERT embeddings...")

        pretrain_params = list(model.gcn_encoder.parameters()) + list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=learning_rate)
        # bert_scheduler = ReduceLROnPlateau(bert_optimizer, mode='min', factor=factor, patience=patience)
    
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample, node_embeddings = train_bert_embedding(
                model, train_loader, bert_optimizer, device
            )
            # bert_scheduler.step(train_loss)
            
            if epoch % log_interval == 0:
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}')
                
        # 학습된 BERT 저장
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
        
    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)

    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, recon_optimizer, max_nodes, device)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            
            # kmeans, cluster_centers = perform_clustering(train_cls_outputs, random_seed, n_clusters=n_cluster)
            # cluster_assignments, cluster_centers, cluster_sizes, n_clusters = analyze_clusters(train_cls_outputs)
            
            cluster_centers = train_cls_outputs.mean(dim=0)
            cluster_centers = cluster_centers.detach().cpu().numpy()
            cluster_centers = cluster_centers.reshape(-1, hidden_dims[-1])

            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = evaluate_model(model, test_loader, max_nodes, cluster_centers, device)
            # scheduler.step(auroc)
            
            all_results.append((auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly))
            print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation loss anomaly = {test_loss_anomaly:.4f}, Validation AUC = {auroc:.4f}, Validation AUPRC = {auprc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
            
            info_test = 'AD_AUC:{:.4f}, AD_AUPRC:{:.4f}, Test_Loss:{:.4f}, Test_Loss_Anomaly:{:.4f}'.format(auroc, auprc, test_loss, test_loss_anomaly)

            print(info_train + '   ' + info_test)

    return auroc


#%%
'''MAIN'''
if __name__ == '__main__':
    ad_aucs = []
    fold_times = []
    splits = get_ad_split_TU(dataset_name, n_cross_val)    

    start_time = time.time()  # 전체 실행 시작 시간

    for trial in range(n_cross_val):
        fold_start = time.time()  # 현재 폴드 시작 시간

        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc = run(dataset_name, random_seed, dataset_AN, trial)
        ad_aucs.append(ad_auc)
        
        fold_end = time.time()  # 현재 폴드 종료 시간
        fold_duration = fold_end - fold_start  # 현재 폴드 실행 시간
        fold_times.append(fold_duration)
        
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds.")
        
    total_time = time.time() - start_time  # 전체 실행 시간
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print(len(ad_aucs))
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")


# %%