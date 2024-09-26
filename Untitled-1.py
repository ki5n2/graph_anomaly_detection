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

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from functools import partial
from multiprocessing import Pool

from module.loss import info_nce_loss, Triplet_loss, loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, batch_nodes_subgraphs


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
        print(f'adj: {adj[0]}')
        x_recon, adj_recon_list, train_cls_outputs, z_ = model(x, edge_index, batch, num_graphs)
        print(f'adj: {adj_recon_list[0]}')
        
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(x_recon[start_node:end_node] - x[start_node:end_node], p='fro')**2 / num_nodes
            node_loss = node_loss / 10
            # print(f'train_node loss: {node_loss}')
            
            # Adjacency reconstruction loss
            adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / (num_nodes**2)
            # print(f'train_adj_loss: {adj_loss}')
            
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
    total_loss = 0
    total_loss_anomaly = 0

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

                # Node reconstruction error
                node_loss = (torch.norm(x_recon[start_node:end_node] - x[start_node:end_node], p='fro')**2) / num_nodes
                
                # Adjacency reconstruction error
                adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / (num_nodes**2)
                
                # 클러스터 중심과의 거리 계산
                cls_vec = e_cls_output[i].cpu().numpy()  # [hidden_dim]
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')  # [1, n_clusters]
                min_distance = distances.min()  # 가장 가까운 클러스터까지의 거리
                
                recon_error = node_loss.item() * 0.1 + adj_loss.item() * 1 + min_distance * 0.5
                recon_errors.append(recon_error.item())
                
                # print(f'test_node_loss: {node_loss.item() * 0.1 }')
                # print(f'test_adj_loss: {adj_loss.item() * 1}')
                # print(f'test_min_distance: {min_distance * 0.5 }')

                if data.y[i].item() == 0:
                    total_loss += recon_error
                else:
                    total_loss_anomaly += recon_error

                start_node = end_node

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

    return auroc, auprc, precision, recall, f1, total_loss / len(test_loader), total_loss_anomaly / len(test_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='AIDS')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset')

parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=2)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--dropout-rate", type=float, default=0.1)

parser.add_argument("--dataset-AN", action="store_false")

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
log_interval: int = args.log_interval
step_size: int = args.step_size
batch_size: int = args.batch_size
n_cross_val: int = args.n_cross_val
random_seed: int = args.random_seed
hidden_dims: list = args.hidden_dims
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
learning_rate: float = args.learning_rate
dropout_rate: float = args.dropout_rate

dataset_AN: bool = args.dataset_AN

set_seed(random_seed)

device = set_device()
# device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
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
        x = F.relu(self.bn(self.conv(x, edge_index)))
        x = self.dropout(x)
        return F.relu(x + residual)


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
    def __init__(self, embed_dim, num_features):
        super(FeatureDecoder, self).__init__()
        self.fc = nn.Linear(embed_dim, num_features)

    def forward(self, z):
        return self.fc(z)


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
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 포지셔널 인코딩 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 계산
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 포지셔널 인코딩 적용
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        pe = pe.unsqueeze(1)  # 크기를 [max_len, 1, d_model]로 변경
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
# 트랜스포머 인코더 클래스 수정
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask):
        # src: [seq_len, batch_size, d_model]
        src = self.pos_encoder(src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32).to(src.device)))
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output


def perform_clustering(train_cls_outputs, n_clusters=5):
    # train_cls_outputs가 이미 텐서이므로, 그대로 사용
    cls_outputs_tensor = train_cls_outputs  # [total_num_graphs, hidden_dim]
    cls_outputs_np = cls_outputs_tensor.detach().cpu().numpy()

    # K-Means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cls_outputs_np)

    # 클러스터 중심 저장
    cluster_centers = kmeans.cluster_centers_

    return kmeans, cluster_centers


#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.transformer_encoder = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=8,
            num_layers=2,
            dim_feedforward=hidden_dims[-1] * 4,
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

        # 그래프별로 노드 임베딩을 분리
        z_list = [z[batch == i] for i in range(num_graphs)]

        # 최대 노드 수 계산
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list)

        # 각 그래프에 대해 CLS 토큰 추가 및 패딩
        z_with_cls_list = []
        mask_list = []
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

        # 트랜스포머 인코딩
        z_with_cls_batch = z_with_cls_batch.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        encoded = self.transformer_encoder(z_with_cls_batch, src_key_padding_mask=mask)
        encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]

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
        
        # node_output_ = node_output.permute(0, 2, 1)
        # adj_recon = torch.matmul(node_output, node_output_) 
        # adj_recon = self.sigmoid(adj_recon)
        # adj_recon = adj_recon * (1 - torch.eye(max_nodes_in_batch, device=node_output.device))
        
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


#%%
'''DATASETS'''
splits = get_ad_split_TU(dataset_name, n_cross_val, random_seed)
loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, splits[0])
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
def run(dataset_name, random_seed, split=None, device=device):
    set_seed(random_seed)

    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']

    model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = loaders['train']
    test_loader = loaders['test']

    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []

    for epoch in range(1, epochs+1):
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, optimizer, max_nodes, device)

        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            kmeans, cluster_centers = perform_clustering(train_cls_outputs, n_clusters=5)
            
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = evaluate_model(model, test_loader, max_nodes, train_cls_outputs, device)

            info_test = 'AD_AUC:{:.4f}, AD_AUPRC:{:.4f}, Test_Loss:{:.4f}, Test_Loss_Anomaly:{:.4f}'.format(auroc, auprc, test_loss, test_loss_anomaly)

            print(info_train + '   ' + info_test)
    

    return auroc


#%%
if __name__ == '__main__':
    ad_aucs = []
    splits = get_ad_split_TU(dataset_name, n_cross_val, random_seed)
    key_auc_list = []

    for trial in range(n_cross_val):
        results = run(dataset_name, random_seed, split=splits[trial])
        ad_auc = results
        ad_aucs.append(ad_auc)

    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print(len(ad_aucs))

    print('[FINAL RESULTS] ' + results)
    
    
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        x = F.relu(self.bn(self.conv(x, edge_index)))
        x = self.dropout(x)
        return F.relu(x + residual)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers
        )

    def forward(self, tgt, memory):
        return self.transformer_decoder(tgt, memory)

class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.1, noise_std=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.transformer_decoder = TransformerDecoder(
            d_model=hidden_dims[-1],
            nhead=8,
            num_layers=3,
            dim_feedforward=hidden_dims[-1] * 4
        )
        self.node_decoder = nn.Linear(hidden_dims[-1], num_features)
        self.edge_decoder = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.dropout = nn.Dropout(dropout_rate)
        self.noise_std = noise_std
        
        # For global anomaly detection
        self.cls_cov = None

    def forward(self, x, edge_index, batch):
        # Encoding
        z = self.encoder(x, edge_index)
        z = self.dropout(z)

        # Add CLS token and prepare for transformer
        num_graphs = batch.max().item() + 1
        cls_tokens = self.cls_token.repeat(num_graphs, 1, 1)
        cls_tokens = cls_tokens.to(device)
        
        z_list = [z[batch == i] for i in range(num_graphs)]
        max_nodes = max(z_graph.size(0) for z_graph in z_list)
        
        z_padded_list = []
        for z_graph in z_list:
            padding = torch.zeros(max_nodes - z_graph.size(0), z_graph.size(1), device=z_graph.device)
            z_padded = torch.cat([z_graph, padding], dim=0)
            z_padded_list.append(z_padded.unsqueeze(0))
        
        z_padded = torch.cat(z_padded_list, dim=0)
        z_with_cls = torch.cat([cls_tokens, z_padded], dim=1)
        
        # Transformer decoding
        u = self.transformer_decoder(z_with_cls, z_with_cls)
        
        # Get CLS tokens
        cls_output = u[:, 0, :]
        
        # Remove CLS token
        u = u[:, 1:, :]
        
        # Unpad and reconstruct node features and adjacency matrices
        x_recon_list = []
        adj_recon_list = []
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            u_graph = u[i, :num_nodes]
            
            # Node feature reconstruction
            x_recon = self.node_decoder(u_graph)
            x_recon_list.append(x_recon)
            
            # Adjacency matrix reconstruction
            adj_recon = torch.sigmoid(u_graph @ u_graph.t())
            adj_recon_list.append(adj_recon)
        
        x_recon = torch.cat(x_recon_list, dim=0)
        
        return x_recon, adj_recon_list, z, u, cls_output

    def loss_function(self, x, x_recon, adj, adj_recon_list, batch):
        # Node feature reconstruction loss
        
        loss = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(x_recon[start_node:end_node] - x[start_node:end_node], p='fro')**2 / num_nodes
            node_loss = node_loss / 10
            
            # Adjacency reconstruction loss
            adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / (num_nodes**2)
            
            loss += node_loss + adj_loss

            start_node = end_node
            
        # node_loss = F.mse_loss(x_recon, x)
        
        # # Edge reconstruction loss
        # edge_loss = 0
        # for i, adj_recon in enumerate(adj_recon_list):
        #     edge_index_graph = edge_index[:, batch == i]
        #     adj_true = torch.zeros_like(adj_recon)
        #     adj_true[edge_index_graph[0], edge_index_graph[1]] = 1
        #     edge_loss += F.binary_cross_entropy(adj_recon, adj_true)
        # edge_loss /= len(adj_recon_list)
        
        # Total loss
        # total_loss = node_loss + edge_loss
        
        return loss, node_loss, adj_loss

    def fit_global_distribution(self, cls_outputs):
        # Fit a multivariate Gaussian to the CLS outputs
        self.cls_cov = EmpiricalCovariance().fit(cls_outputs.detach().cpu().numpy())

    def detect_anomalies(self, x, edge_index, batch, threshold_local=0.1, threshold_global=0.95):
        self.eval()
        with torch.no_grad():
            x_recon, adj_recon_list, _, _, cls_output = self(x, edge_index, batch)
            
            # Local anomaly detection
            reconstruction_error = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
            local_anomalies = reconstruction_error > threshold_local
            
            # Global anomaly detection
            cls_output_np = cls_output.cpu().numpy()
            mahalanobis_distances = self.cls_cov.mahalanobis(cls_output_np)
            global_anomalies = mahalanobis_distances > threshold_global
            
        return local_anomalies, global_anomalies, reconstruction_error, mahalanobis_distances

# Training function
def train(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, adj_recon_list, _, _, cls_output = model(batch.x, batch.edge_index, batch.batch)
            loss, node_loss, edge_loss = model.loss_function(batch.x, batch.edge_index, batch.batch, x_recon, adj_recon_list)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Fit global distribution after training
    model.eval()
    cls_outputs = []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            _, _, _, _, cls_output = model(batch.x, batch.edge_index, batch.batch)
            cls_outputs.append(cls_output)
    cls_outputs = torch.cat(cls_outputs, dim=0)
    model.fit_global_distribution(cls_outputs)

# Usage example
num_features = 32
hidden_dims = [256, 128]
model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, optimizer, device, num_epochs=100)

# Anomaly detection
def detect_anomalies_in_dataset(model, data_loader, device):
    model.eval()
    local_anomalies_all = []
    global_anomalies_all = []
    reconstruction_errors_all = []
    mahalanobis_distances_all = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            local_anomalies, global_anomalies, reconstruction_errors, mahalanobis_distances = model.detect_anomalies(batch.x, batch.edge_index, batch.batch)
            local_anomalies_all.extend(local_anomalies.cpu().numpy())
            global_anomalies_all.extend(global_anomalies)
            reconstruction_errors_all.extend(reconstruction_errors.cpu().numpy())
            mahalanobis_distances_all.extend(mahalanobis_distances)

    return local_anomalies_all, global_anomalies_all, reconstruction_errors_all, mahalanobis_distances_all

# Detect anomalies in test set
local_anomalies, global_anomalies, reconstruction_errors, mahalanobis_distances = detect_anomalies_in_dataset(model, test_loader, device)

# Print results
print(f"Number of local anomalies detected: {sum(local_anomalies)}")
print(f"Number of global anomalies detected: {sum(global_anomalies)}")