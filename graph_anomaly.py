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

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from module.loss import Triplet_loss, loss_cal, focal_loss
from module.utils import set_device, add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon, visualize


#%%
'''TRAIN'''
def train(model, train_loader, optimizer, scheduler, threshold=0.5):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, batch, x_recon, adj_recon_list, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z = model(data)
        # adj_list = to_dense_adj(data.edge_index, batch=data.batch)
        
        loss = 0
        start_node = 0
        
        for i in range(data.num_graphs): # 각 그래프별로 손실 계산
            num_nodes = (data.batch == i).sum().item() # 현재 그래프에 속하는 노드 수
            end_node = start_node + num_nodes
            graph_num_nodes = end_node - start_node        
            
            # node_loss_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
            # node_loss = node_loss_1 / 200
            
            edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
            graph_edge_loss = edge_loss/graph_num_nodes
            l1_loss = graph_edge_loss / 30
            
            # focal_loss_value = focal_loss(adj_recon_list[i], adj[i], gamma=2, alpha=0.25)
            # l1_loss = focal_loss_value
            # graph_edge_loss = focal_loss_value/graph_num_nodes
            # edge_loss_2 = F.binary_cross_entropy(adj_recon_prime_list[i], adj[i])
            # edge_loss = edge_loss_1 / 100
            
            edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
            edge_index = edges.t()
            
            z_tilde =  model.encode(x_recon, edge_index).to('cuda')
            z_tilde_g = global_max_pool(z_tilde, batch)
            
            recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
            graph_z_node_loss = recon_z_node_loss/graph_num_nodes
            
            recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
            l3_loss = (graph_z_node_loss / 10) + (recon_z_graph_loss / 10)
            loss += l1_loss + l3_loss
            
            start_node = end_node
        
        node_loss = torch.norm(x_recon - data.x, p='fro')**2
        node_loss = (node_loss/x_recon.size(0))
        
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
        l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
        loss += node_loss + triplet_loss + l2_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    return total_loss / len(train_loader)


#%%
'''EVALUATION'''
def evaluate_model(model, val_loader, threshold = 0.5):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  # Move data to the MPS device
            adj, z, z_g, batch, x_recon, adj_recon_list, _, _, _, _, _ = model(data)  # 모델 예측값
            
            recon_errors = []
            start_node = 0
            for i in range(data.num_graphs): # 각 그래프별로 손실 계산
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() # 현재 그래프에 속하는 노드 수
                end_node = start_node + num_nodes
                graph_num_nodes = end_node - start_node        
                
                node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                # node_recon_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
                graph_node_loss = node_loss/graph_num_nodes
                node_recon_error = graph_node_loss / 30

                # focal_loss_value = focal_loss(adj_recon_list[i], adj[i], gamma=2, alpha=0.25)
                # # edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
                # # edge_recon_2 = F.binary_cross_entropy(adj_recon_prime_list[i], adj[i])
                # graph_edge_loss = focal_loss_value/graph_num_nodes
                # edge_recon_error = graph_edge_loss * 100
                
                edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
                graph_edge_loss = edge_loss/graph_num_nodes
                edge_recon_error = graph_edge_loss / 30
                
                edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                edge_index = edges.t()
                
                z_tilde =  model.encode(x_recon, edge_index).to('cuda')
                z_tilde_g = global_max_pool(z_tilde, batch)
                
                recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                graph_z_node_loss = recon_z_node_loss/graph_num_nodes

                recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                graph_recon_loss = (graph_z_node_loss / 10) + (recon_z_graph_loss / 10)
            
                recon_error += node_recon_error + edge_recon_error + graph_recon_loss
                recon_errors.append(recon_error.item())

                start_node = end_node
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
        
    return auroc


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset/data')
parser.add_argument("--dataset-name", type=str, default='NCI1')
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])
parser.add_argument("--test-batch-size", type=int, default=9999)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--n-cross-val", type=int, default=5)

parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--fine-tune", type=bool, default=True)
parser.add_argument("--dataset-AN", action="store_false")
parser.add_argument("--pretrained", action="store_false")

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
'''OPTIONS'''
assets_root: str = args.assets_root
data_root: str = args.data_root
dataset_name: str = args.dataset_name

hidden_dims: list = args.hidden_dims
test_batch_size: int = args.test_batch_size
n_test_anomaly: int = args.n_test_anomaly
batch_size: int = args.batch_size
random_seed: int = args.random_seed
epochs: int = args.epochs
n_cross_val: int = args.n_cross_val

learning_rate: float = args.learning_rate
test_size: float = args.test_size
fine_tune: bool = args.fine_tune
dataset_AN: bool = args.dataset_AN
pretrained: bool = args.pretrained

# device = torch.device('cpu')
device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# wandb.init(project="graph anomaly detection", entity="ki5n2")

# wandb.config.update(args)

# wandb.config = {
#   "random_seed": random_seed,
#   "learning_rate": 0.0001,
#   "epochs": 100
# }


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
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x, edge_index))
        return F.relu(x + residual)
    
    
# %%
class GRAPH_AUTOENCODER(torch.nn.Module):
    def __init__(self, num_features, hidden_dims):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder_blocks = torch.nn.ModuleList()
        self.decoder_blocks = torch.nn.ModuleList()
        self.encoders_node = torch.nn.ModuleList()
        self.encoder_sub_blocks = torch.nn.ModuleList()
        self.classifiers = torch.nn.ModuleList()
        self.act = nn.ReLU()
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(ResidualBlock(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        for hidden_dim in reversed(hidden_dims[:-1]):
            self.decoder_blocks.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.decoder_blocks.append(nn.Linear(current_dim, num_features))

        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoders_node.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim  
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_sub_blocks.append(ResidualBlock(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)                        
            
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # adjacency matrix
        adj = adj_original(edge_index, batch)
            
        # latent vector
        z = self.encode(x, edge_index)
        
        # perturbation
        z_prime = add_gaussian_perturbation(z)
        
        # adjacency matrix reconstruction
        adj_recon_list, adj_recon_prime_list = adj_recon(z, z_prime, batch)
        
        # node reconstruction
        x_recon = self.decode(z)
        
        # Graph classification
        z_g = global_max_pool(z, batch)  # Aggregate features for classification
        z_prime_g = global_max_pool(z_prime, batch) # (batch_size, embedded size)
        
        z_g_mlp = self.projection_head(z_g)
        z_prime_g_mlp = self.projection_head(z_prime_g) # (batch_size, embedded size)
        
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
        
        return adj, z, z_g, batch, x_recon, adj_recon_list, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z
    
    def encode(self, x, edge_index):
        for block in self.encoder_blocks:
            x = block(x, edge_index)
        return F.normalize(x, p=2, dim=1)

    def decode(self, z):
        x = z
        for decoder in self.decoder_blocks[:-1]:
            x = self.act(decoder(x))
        return torch.sigmoid(self.decoder_blocks[-1](x))
    
    def encode_node(self, x):
        for encoder in self.encoders_node[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoders_node[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_subgraph(self, x, edge_index):
        for block in self.encoder_sub_blocks:
            x = block(x, edge_index)
        return F.normalize(x, p=2, dim=1)
    
    def process_subgraphs(self, subgraphs):
        # 각 서브그래프에 대해 인코딩을 실행
        subgraph_embeddings = []
        for i in range(len(subgraphs)):
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


#%%
'''DATASETS'''
graph_dataset = TUDataset(root='./dataset', name='BZR').shuffle()
labels = np.array([data.y.item() for data in graph_dataset])

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of edge features: {graph_dataset.num_edge_features}')
print(f'labels: {labels}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = graph_dataset.num_features
hidden_dims=[256, 128]

model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# %%
'''TRAIN PROCESS'''
torch.autograd.set_detect_anomaly(True)  

epochs = 100
for fold, (train_idx, val_idx) in enumerate(skf.split(graph_dataset, labels)):
    print(f"Fold {fold + 1}")
    
    train_normal_idx = [idx for idx in train_idx if labels[idx] == 0]
    print(len(train_idx))
    print(len(train_normal_idx))
    
    val_normal_idx = [idx for idx in val_idx if labels[idx] == 0]
    val_anormal_idx = [idx for idx in val_idx if labels[idx] == 1]
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
    val_loader = DataLoader(val_dataset, batch_size=9999, shuffle=True)
    
    print(f"  Training set size (normal only): {len(train_dataset)}")
    print(f"  Validation set size (normal + abnormal): {len(val_dataset)}")
    
    model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, scheduler)
        auroc = evaluate_model(model, val_loader)
        scheduler.step(auroc)  # AUC 기반으로 학습률 조정
        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation AUC = {auroc:.4f}')
        # wandb.log({"epoch": epoch, "train loss": train_loss, "test AUC": auroc})

    print("\n")
    
# wandb.finish()

#%%

