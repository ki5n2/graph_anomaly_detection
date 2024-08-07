#%%
import os
import torch
import random
import numpy as np
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import init
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import DataLoader, Dataset, Data, Batch
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph, to_undirected, to_networkx
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, GAE, InnerProductDecoder

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from module.loss import Triplet_loss, loss_cal
from module.utils import set_device, add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon, visualize

device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#%%
'''DATASETS'''
Tox21_p53_training = TUDataset(root='./dataset', name='Tox21_p53_training').shuffle()
print(f'Number of Tox21_p53_training: {len(Tox21_p53_training)}')
print(f'Number of features: {Tox21_p53_training.num_features}')

Tox21_p53_evaluation = TUDataset(root='./dataset', name='Tox21_p53_evaluation').shuffle()
print(f'Number of Tox21_p53_evaluation: {len(Tox21_p53_evaluation)}')
print(f'Number of features: {Tox21_p53_evaluation.num_features}')

Tox21_p53_testing = TUDataset(root='./dataset', name='Tox21_p53_testing').shuffle()
labels = np.array([data.y.item() for data in Tox21_p53_testing])

print(f'Number of Tox21_p53_testing: {len(Tox21_p53_testing)}')
print(f'Number of features: {Tox21_p53_testing.num_features}')

print(f'Number of edge features: {Tox21_p53_testing.num_edge_features}')
print(f'labels: {labels}')

# 5-fold cross-validation 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#%%
# 데이터 로드
Tox21_p53_training = TUDataset(root='./dataset', name='Tox21_p53_training').shuffle()
Tox21_p53_testing = TUDataset(root='./dataset', name='Tox21_p53_testing').shuffle()

# 훈련 데이터에서 정상 그래프만 선택
normal_graphs = [data for data in Tox21_p53_training if data.y.item() == 0]

#%%
dataset_normal = [data for data in graph_dataset if data.y.item() == 0]
dataset_anomaly = [data for data in graph_dataset if data.y.item() == 1]

print(f"Number of normal samples: {len(dataset_normal)}")
print(f"Number of anomaly samples: {len(dataset_anomaly)}")

train_normal_data, test_normal_data = train_test_split(dataset_normal, test_size=0.25, random_state=42)
evaluation_data = test_normal_data + dataset_anomaly[:200]

train_loader = DataLoader(train_normal_data, batch_size=128, shuffle=True)
test_loader = DataLoader(evaluation_data, batch_size=128, shuffle=True)

print(f"Number of samples in the evaluation dataset: {len(evaluation_data)}")
print(f"Number of test normal data: {len(test_normal_data)}")
print(f"Number of test anomaly samples: {len(dataset_anomaly[:200])}")
print(f"Ratio of test anomaly: {len(dataset_anomaly[:200]) / len(evaluation_data)}")


# %%
class GRAPH_AUTOENCODER(torch.nn.Module):
    def __init__(self, num_features, hidden_dims):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.encoders_node = torch.nn.ModuleList()
        self.classifiers = torch.nn.ModuleList()
        self.encoders_subgraphs = torch.nn.ModuleList()
        self.act = nn.ReLU()
        self.projection_head = nn.Sequential(nn.Linear(hidden_dims[-1], hidden_dims[-1]), nn.ReLU(), nn.Linear(hidden_dims[-1], hidden_dims[-1]))

        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoders.append(GCNConv(current_dim, hidden_dim, add_self_loops=True))
            current_dim = hidden_dim
        
        for hidden_dim in reversed(hidden_dims[:-1]):
            self.decoders.append(nn.Linear(current_dim, hidden_dim, bias=False))
            current_dim = hidden_dim
        self.decoders.append(nn.Linear(current_dim, num_features, bias=False))
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoders_node.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim  
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoders_subgraphs.append(GCNConv(current_dim, hidden_dim, add_self_loops=True))
            current_dim = hidden_dim
            
        # self.classifiers.append(nn.Linear(hidden_dims[-1], 2))  # Assume last encoder output dimension for simplicity        
        self.classifiers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[1]),  # Reduce dimension to a smaller space
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 16),  # Reduce dimension to a smaller space
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
            )    
        
        # Node-level anomaly detector
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),  # Reduce dimension to a smaller space
            torch.nn.ReLU(),
            nn.Linear(32, 1),
            torch.nn.Sigmoid()  # Output a probability of being anomalous
        )
        
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
        for encoder in self.encoders[:-1]:
            x = self.act(encoder(x, edge_index))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoders[-1](x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x

    def decode(self, x):
        for decoder in self.decoders[:-1]:
            x = self.act(decoder(x))            
        x = torch.sigmoid(self.decoders[-1](x))
        return x
    
    def encode_node(self, x):
        for encoder in self.encoders_node[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoders_node[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def classify(self, x):
        for classifier in self.classifiers:
            x = classifier(x)
        return x

    def encode_subgraph(self, x, edge_index):
        for encoder in self.encoders_subgraphs[:-1]:
            x = self.act(encoder(x, edge_index))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoders[-1](x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x
   
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
num_features = graph_dataset.num_features
hidden_dims=[256, 128]

model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
def train(model, train_loader, optimizer, threshold=0.5):
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
            
            node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
            # node_loss_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
            # node_loss = node_loss_1 / 200
            
            edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
            # edge_loss_2 = F.binary_cross_entropy(adj_recon_prime_list[i], adj[i])
            # edge_loss = edge_loss_1 / 100
            
            l1_loss = (node_loss / 200) + (edge_loss / 100)
            
            edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
            edge_index = edges.t()
            
            z_tilde =  model.encode(x_recon, edge_index).to('cuda')
            z_tilde_g = global_max_pool(z_tilde, batch)
            
            recon_z_node_loss = torch.norm(z[i] - z_tilde[i], p='fro')**2
            recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
            l3_loss = (recon_z_node_loss / 5) + (recon_z_graph_loss / 5)
                        
            loss += l1_loss + l3_loss
            
            start_node = end_node
        
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 50
        l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) / 10
        loss += triplet_loss + l2_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


#%%
def evaluate_model(model, test_loader, threshold = 0.5):
    model.eval()
    max_AUC = 0.0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # Move data to the MPS device
            adj, z, z_g, batch, x_recon, adj_recon_list, _, _, _, _, _ = model(data)  # 모델 예측값
            
            label_y=[]
            label_pred = []

            start_node = 0
            for i in range(data.num_graphs): # 각 그래프별로 손실 계산
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() # 현재 그래프에 속하는 노드 수
                end_node = start_node + num_nodes

                node_recon_1 = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                # node_recon_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
                node_recon_error = node_recon_1 / 200
             
                edge_recon_1 = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
                # edge_recon_2 = F.binary_cross_entropy(adj_recon_prime_list[i], adj[i])
                edge_recon_error = edge_recon_1 / 100
                
                edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                edge_index = edges.t()
                
                z_tilde =  model.encode(x_recon, edge_index).to('cuda')
                z_tilde_g = global_max_pool(z_tilde, batch)
                
                recon_z_node_loss = torch.norm(z[i] - z_tilde[i], p='fro')**2
                recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                graph_recon_loss = (recon_z_node_loss / 5) + (recon_z_graph_loss / 5)
            
                recon_error += node_recon_error + edge_recon_error + graph_recon_loss
                label_pred.append(recon_error.item())
                
                start_node = end_node
            
            label_pred = np.array(label_pred)
            
            label_y.extend(data.y.cpu().numpy())
            label_y = np.array(label_y)
            label_y = 1.0 - label_y
                   
            fpr_ab, tpr_ab, _ = roc_curve(label_y, label_pred)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            
            # print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            if test_roc_ab > max_AUC:
                max_AUC=test_roc_ab
        
        auroc_final = max_AUC
        
    return auroc_final


# %%
torch.autograd.set_detect_anomaly(True)  

epochs = 100
for fold, (train_idx, val_idx) in enumerate(skf.split(graph_dataset, labels)):
    print(f"Fold {fold + 1}")
    
    # 훈련 데이터에서 정상 데이터만 선택
    train_normal_idx = [idx for idx in train_idx if labels[idx] == 1]
    
    # 훈련 및 검증 데이터셋 생성
    train_dataset = [graph_dataset[i] for i in train_normal_idx]
    val_dataset = [graph_dataset[i] for i in val_idx]
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"  Training set size (normal only): {len(train_dataset)}")
    print(f"  Validation set size (normal + abnormal): {len(val_dataset)}")
    
    model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer)
        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}')

        # 에포크마다 평가 수행
        auroc_final = evaluate_model(model, test_loader)
        print(f'Epoch {epoch+1}: Validation AUC = {auroc_final:.4f}')

    print("\n")
    
    