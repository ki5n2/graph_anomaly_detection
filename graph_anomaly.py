#%%
'''IMPORTS'''
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
from torch_geometric.utils import add_self_loops, k_hop_subgraph, to_dense_adj, subgraph, to_undirected, to_networkx
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, GAE, InnerProductDecoder

from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from module.loss import Triplet_loss, loss_cal, focal_loss
from module.utils import set_seed, set_device, add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon, visualize, EarlyStopping


#%%
'''TRAIN'''
def train(model, train_loader, optimizer, threshold=0.5):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z = model(data)
        
        loss = 0
        start_node = 0
        
        for i in range(data.num_graphs): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            graph_num_nodes = end_node - start_node        
            
            # node_loss_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
            # node_loss = node_loss_1 / 200

            adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
            l1_loss = adj_loss / 2
            
            # focal_loss_value = focal_loss(adj_recon_list[i], adj[i], gamma=2, alpha=0.25)
            # graph_edge_loss = focal_loss_value/graph_num_nodes
            
            # edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
            # edge_index = edges.t()
    
            # recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
            loss += l1_loss 
            
            start_node = end_node

        node_loss = torch.norm(x_recon - data.x, p='fro')**2
        node_loss = (node_loss/x_recon.size(0)) / 20
    
        recon_z_node_loss = torch.norm(z - z_tilde, p='fro')**2
        graph_z_node_loss = recon_z_node_loss / (z.size(1) * 2)
        
        z_g_dist = torch.pdist(z_g)
        z_tilde_g_dist = torch.pdist(z_tilde_g)
        w_distance = torch.tensor(wasserstein_distance(z_g_dist.detach().cpu().numpy(), z_tilde_g_dist.detach().cpu().numpy()), device='cuda')
        w_distance = w_distance * 50
        
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
        l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
        loss += node_loss + graph_z_node_loss + w_distance + triplet_loss + l2_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


#%%
'''EVALUATION'''
def evaluate_model(model, val_loader, threshold = 0.5):
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  
            adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z = model(data)
            
            recon_errors = []
            loss = 0
            start_node = 0
            
            for i in range(data.num_graphs): 
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() 
                end_node = start_node + num_nodes
                graph_num_nodes = end_node - start_node        
                
                node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                graph_node_loss = node_loss/graph_num_nodes
                node_recon_error = graph_node_loss / 100

                adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
                edge_recon_error = adj_loss / 16
                
                # edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                # edge_index = edges.t()
                
                recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                graph_z_node_loss = recon_z_node_loss/graph_num_nodes

                recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                graph_recon_loss = (graph_z_node_loss / 2) + (recon_z_graph_loss / 2)
            
                recon_error += node_recon_error + edge_recon_error + graph_recon_loss
                recon_errors.append(recon_error.item())
            
                loss += node_recon_error + edge_recon_error + graph_recon_loss

                start_node = end_node
            
            triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
            l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
            loss += triplet_loss + l2_loss
            total_loss += loss.item()
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    auroc = auc(fpr, tpr)
    
    pred_labels = (all_scores > optimal_threshold).astype(int)
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    
    return auroc, precision, recall, f1, total_loss / len(val_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='NCI1')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset/data')

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--test-batch-size", type=int, default=9999)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.1)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--dataset-AN", action="store_false")
parser.add_argument("--pretrained", action="store_false")

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
patience: list = args.patience
batch_size: int = args.batch_size
random_seed: int = args.random_seed
n_cross_val: int = args.n_cross_val
hidden_dims: list = args.hidden_dims
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
learning_rate: float = args.learning_rate

dataset_AN: bool = args.dataset_AN
pretrained: bool = args.pretrained

set_seed(random_seed)

device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#%%
# data_list = []
# label_list = []

# for data in graph_dataset:
#     data_list.append(data)
#     label_list.append(data.y.item())

# kfd = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

# splits = []
# for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
#     print(k)
#     print(train_index)
#     print(test_index)
#     splits.append((train_index, test_index))


# #%%
# Tox21_p53_training = TUDataset(root='./dataset', name='Tox21_p53_training').shuffle()
# Tox21_p53_testing = TUDataset(root='./dataset', name='Tox21_p53_testing').shuffle()

# # 훈련 데이터에서 정상 그래프만 선택
# normal_graphs = [data for data in Tox21_p53_training if data.y.item() == 0]

# #%%
# graph_dataset = TUDataset(root='./dataset', name='BZR').shuffle()

# dataset_normal = [data for data in graph_dataset if data.y.item() == 0]
# dataset_anomaly = [data for data in graph_dataset if data.y.item() == 1]

# print(f"Number of normal samples: {len(dataset_normal)}")
# print(f"Number of anomaly samples: {len(dataset_anomaly)}")

# train_normal_data, test_normal_data = train_test_split(dataset_normal, test_size=0.27, random_state=42)
# evaluation_data = test_normal_data + dataset_anomaly

# train_loader = DataLoader(train_normal_data, batch_size=200, shuffle=True)
# test_loader = DataLoader(evaluation_data, batch_size=128, shuffle=True)

# print(f"Number of samples in the evaluation dataset: {len(evaluation_data)}")
# print(f"Number of test normal data: {len(test_normal_data)}")
# print(f"Number of test anomaly samples: {len(dataset_anomaly)}")
# print(f"Ratio of test anomaly: {len(dataset_anomaly) / len(evaluation_data)}")


#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x, edge_index))
        x = self.dropout(x)
        return F.relu(x + residual)


#%%
class AdjacencyDecoder(nn.Module):
    def __init__(self, embed_dim):
        super(AdjacencyDecoder, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, z):
        # 내적을 통한 인접 행렬 재구성
        return torch.sigmoid(torch.matmul(z, z.t()))


class AdjacencyDecoder_(nn.Module):
    def __init__(self, embed_dim):
        super(AdjacencyDecoder_, self).__init__()
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z):
        return torch.sigmoid(torch.matmul(torch.matmul(z, self.W), z.t()))

    
class FeatureDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dims, num_features):
        super(FeatureDecoder, self).__init__()
        self.layers = nn.ModuleList()
        dims = [embed_dim] + list(hidden_dims) + [num_features]
        for i in range(len(dims) - 1):
            self.layers.append(GCNConv(dims[i], dims[i+1]))
        
    def forward(self, z, edge_index):
        for layer in self.layers[:-1]:
            z = F.relu(layer(z, edge_index))
        return self.layers[-1](z, edge_index)

    
class Feature_Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dims, num_features, device=device):
        super(Feature_Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList().to(device)
        current_dim = embed_dim
        for hidden_dim in reversed(hidden_dims[:-1]):
            self.decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            self.decoder_layers.append(nn.ReLU())
            self.decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        self.decoder_layers.append(nn.Linear(current_dim, num_features))

    def forward(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
        return z
    

class Adj_Decoder(nn.Module):
    def __init__(self, embed_dim):
        super(Adj_Decoder, self).__init__()
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, batch):
        # z: 노드 임베딩 (N x embed_dim)
        # batch: 각 노드가 속한 그래프를 나타내는 텐서 (N,)
        
        adj_recon_list = []
        
        # 가중치 행렬을 적용한 임베딩 계산
        weighted_z = torch.matmul(z, self.W)
        
        for batch_idx in torch.unique(batch):
            mask = (batch == batch_idx)
            z_graph = z[mask]
            weighted_z_graph = weighted_z[mask]
            
            # 개선된 인접 행렬 재구성
            adj_recon_graph = torch.sigmoid(torch.matmul(z_graph, weighted_z_graph.t()))
            adj_recon_list.append(adj_recon_graph)
        
        return adj_recon_list


class BilinearEdgeDecoder(nn.Module):
    def __init__(self, input_dim):
        super(BilinearEdgeDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, z):
        adj = torch.sigmoid(torch.mm(torch.mm(z, self.weight), z.t()))
        return adj
    
    
#%%    
class GRAPH_AUTOENCODER(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.2):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder_blocks = nn.ModuleList()        
        self.encoder_node_blocks = nn.ModuleList()        
        self.encoder_sub_blocks = nn.ModuleList()
        
        self.edge_decoder = BilinearEdgeDecoder(hidden_dims[-1])
        self.feature_decoder = Feature_Decoder(hidden_dims[-1], hidden_dims, num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_node_blocks.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            current_dim = hidden_dim  
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_sub_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim

        # 가중치 초기화
        self.apply(self._init_weights)
    
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # adjacency matrix
        adj = adj_original(edge_index, batch)
            
        # latent vector
        z = self.encode(x, edge_index)
        z = self.dropout(z)
        
        # perturbation
        z_prime = add_gaussian_perturbation(z)
        
        # adjacency matrix reconstruction
        adj_recon_list = []
        for i in range(data.num_graphs):
            mask = (batch == i)
            z_graph = z[mask]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)

        new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch)
        
        # node reconstruction
        x_recon = self.feature_decoder(z)

        # Graph classification
        z_g = global_max_pool(z, batch)  # Aggregate features for classification
        z_prime_g = global_max_pool(z_prime, batch) # (batch_size, embedded size)
        
        z_g_mlp = self.projection_head(z_g)
        z_prime_g_mlp = self.projection_head(z_prime_g) # (batch_size, embedded size)
        
        z_tilde = self.encode(x_recon, new_edge_index)
        z_tilde_g = global_max_pool(z_tilde, batch)
        
        # subgraph
        batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
        
        pos_x, pos_edge_index, pos_batch = batched_pos_subgraphs.x, batched_pos_subgraphs.edge_index, batched_pos_subgraphs.batch
        pos_sub_z, pos_new_edge_index = self.process_subgraphs(batched_pos_subgraphs)
        pos_sub_z = torch.cat(pos_sub_z) # (number of nodes, embedded size)
        
        unique_pos_batch, new_pos_batch = torch.unique(pos_batch, return_inverse=True)
        pos_sub_z_g = global_mean_pool(pos_sub_z, new_pos_batch)
        
        neg_x, neg_edge_index, neg_batch = batched_neg_subgraphs.x, batched_neg_subgraphs.edge_index, batched_neg_subgraphs.batch
        neg_sub_z, neg_new_edge_index = self.process_subgraphs(batched_neg_subgraphs)
        neg_sub_z = torch.cat(neg_sub_z)
        
        unique_neg_batch, new_neg_batch = torch.unique(neg_batch, return_inverse=True)
        neg_sub_z_g = global_mean_pool(neg_sub_z, new_neg_batch)
        
        target_z = self.encode_node(batched_target_node_features) # (batch_size, feature_size)
        
        return adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z
    
    
    def get_edge_index_from_adj_list(self, adj_recon_list, batch, threshold=0.5):
        edge_index_list = []
        start_idx = 0
        for i, adj in enumerate(adj_recon_list):
            num_nodes = (batch == i).sum().item()
            adj_binary = (adj > threshold).float()  # 임계값 적용
            edge_index = adj_binary.nonzero().t()
            edge_index += start_idx  # 전체 그래프에서의 인덱스로 조정
            edge_index_list.append(edge_index)
            start_idx += num_nodes
        return torch.cat(edge_index_list, dim=1)

    def encode(self, x, edge_index):
        for block in self.encoder_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)

    def encode_node(self, x):
        for encoder in self.encoder_node_blocks[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoder_node_blocks[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_subgraph(self, x, edge_index):
        for block in self.encoder_sub_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)
    
    def process_subgraphs(self, subgraphs):
        # 각 서브그래프에 대해 인코딩을 실행
        subgraph_embeddings = []
        for i in range(subgraphs.num_graphs):
            subgraph = subgraphs[i]
            x = subgraph.x
            edge_index = subgraph.edge_index

            # 로컬 인덱스로 edge_index 재조정
            unique_nodes, new_edge_index = torch.unique(edge_index, return_inverse=True)
            new_edge_index = new_edge_index.reshape(edge_index.shape)

            # 서브그래프 인코딩
            z = self.encode_subgraph(x, new_edge_index)
            subgraph_embeddings.append(z)

        return subgraph_embeddings, new_edge_index
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)            
                

#%%
'''DATASETS'''
data_name = 'COX2'
graph_dataset = TUDataset(root='./dataset', name=data_name).shuffle()

labels = np.array([data.y.item() for data in graph_dataset])

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of edge features: {graph_dataset.num_edge_features}')
print(f'labels: {labels}')

skf = StratifiedKFold(n_splits=n_cross_val, shuffle=True, random_state=random_seed)


#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = graph_dataset.num_features
hidden_dims=[256, 128]

model = GRAPH_AUTOENCODER(num_features, hidden_dims, dropout_rate=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
early_stopping = EarlyStopping(patience=30, verbose=True)


# %%
'''TRAIN PROCESS'''
torch.autograd.set_detect_anomaly(True)  

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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)
    
    print(f"  Training set size (normal only): {len(train_dataset)}")
    print(f"  Validation set size (normal + abnormal): {len(val_dataset)}")
    
    model = GRAPH_AUTOENCODER(num_features, hidden_dims, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
    early_stopping = EarlyStopping(patience=30, verbose=True)
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer)
        auroc, precision, recall, f1, test_loss = evaluate_model(model, val_loader)
        scheduler.step(test_loss)  # AUC 기반으로 학습률 조정
        early_stopping(test_loss, model)

        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation AUC = {auroc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')

    if early_stopping.early_stop:
        print("Early stopping")
        break

    print("\n")
    

# %%
