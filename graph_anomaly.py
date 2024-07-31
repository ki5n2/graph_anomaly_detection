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
from torch_geometric.data import DataLoader, Dataset, Data, Batch
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, GAE, InnerProductDecoder
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph, to_undirected, to_networkx
from torch_geometric.transforms import BaseTransform

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from module.utils import set_device, add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon


#%%
device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#%%
'''DATASETS'''
dataset_AIDS = TUDataset(root='./dataset', name='AIDS')
dataset_AIDS = dataset_AIDS.shuffle()

print(f'Number of graphs: {len(dataset_AIDS)}')
print(f'Number of features: {dataset_AIDS.num_features}')
print(f'Number of edge features: {dataset_AIDS.num_edge_features}')


#%%
dataset_normal = [data for data in dataset_AIDS if data.y.item() == 1]
dataset_anomaly = [data for data in dataset_AIDS if data.y.item() == 0]
train_data, test_data = train_test_split(dataset_normal, test_size=0.125, random_state=42)
evaluation_data = test_data + dataset_anomaly

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(evaluation_data, batch_size=128, shuffle=False)

print(f"Number of positive samples: {len(dataset_normal)}")
print(f"Number of negative samples: {len(dataset_anomaly)}")
print(f"Number of samples in the evaluation dataset: {len(evaluation_data)}")


#%%  
act = nn.ReLU()

def encode(x, edge_index):
    for encoder in encoders[:-1]:
        x = act(encoder(x, edge_index))
        bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
        x = bn_module(x)
    x = encoders[-1](x, edge_index)
    return x
    
def decode(x):
    for decoder in decoders[:-1]:
        x = act(decoder(x))                
    x = torch.sigmoid(decoders[-1](x))
    return x

def encode_node(x):
    for encoder in encoders_node[:-1]:
        x = act(encoder(x))
        bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
        x = bn_module(x)
    x = encoders_node[-1](x)
    return x

def classify(x):
    for classifier in classifiers:
        x = classifier(x)
    return x

def encode_subgraph(x, edge_index):
    for encoder in encoders_subgraphs:
        x = act(encoder[:-1](x, edge_index))
        bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
        x = bn_module(x)
    x = encoders[-1](x, edge_index)
    return x

def process_subgraphs(subgraphs):
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
        z = encode_subgraph(x, new_edge_index)
        subgraph_embeddings.append(z)

    return subgraph_embeddings, new_edge_index

#%%
def visualize(graph, color='skyblue', edge_color='blue'):
    G = to_networkx(graph, to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                 node_color=color, edge_color=edge_color)


#%%
dataset_AIDS[0]
for i in range(10, 15):
    visualize(dataset_AIDS[16])
    
    
#%%
class Triplet_Loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Triplet_Loss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 거리 계산
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss 계산
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses

Triplet_loss = Triplet_Loss(margin=1.0)


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
        # adj = to_dense_adj(edge_index, batch, max_num_nodes=x.shape[0])[0]
        
        adj = adj_original(edge_index, batch)
            
        # latent vector
        z = self.encode(x, edge_index)
        
        # perturbation
        z_prime = add_gaussian_perturbation(z)
        
        # reconstruction adjacency matrix
        adj_recon_list, adj_recon_prime_list = adj_recon(z, z_prime, batch)
        
        # node reconstruction
        x_recon = self.decode(z)
        # x_recon_prime = self.decode(z_prime)
        
        # Graph classification
        z_pool = global_mean_pool(z, batch)  # Aggregate features for classification
        # (batch_size, 2)
        # graph_pred = self.classify(z_pool)
        
        # subgraph
        batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
        
        # target_z = classifiers_(batched_target_node_features)
        # (batch_size, feature_size)
        target_z = self.encode_node(batched_target_node_features)
        
        pos_x, pos_edge_index, pos_batch = batched_pos_subgraphs.x, batched_pos_subgraphs.edge_index, batched_pos_subgraphs.batch
        pos_sub_z, pos_new_edge_index = self.process_subgraphs(batched_pos_subgraphs)
        pos_sub_z = torch.cat(pos_sub_z) # (number of nodes, embedded size)
        
        unique_pos_batch, new_pos_batch = torch.unique(pos_batch, return_inverse=True)
        pos_sub_z_pool = global_mean_pool(pos_sub_z, new_pos_batch)
        # pos_sub_graph_pred = self.classify(pos_sub_z_pool)
        
        neg_x, neg_edge_index, neg_batch = batched_neg_subgraphs.x, batched_neg_subgraphs.edge_index, batched_neg_subgraphs.batch
        neg_sub_z, neg_new_edge_index = self.process_subgraphs(batched_neg_subgraphs)
        neg_sub_z = torch.cat(neg_sub_z)
        
        unique_neg_batch, new_neg_batch = torch.unique(neg_batch, return_inverse=True)
        neg_sub_z_pool = global_mean_pool(neg_sub_z, new_neg_batch)
        neg_sub_graph_pred = self.classify(neg_sub_z_pool)
        
        # pos_adj = to_dense_adj(pos_new_edge_index, new_pos_batch, max_num_nodes=pos_x.shape[0])[0]
        # neg_adj = to_dense_adj(neg_new_edge_index, new_neg_batch, max_num_nodes=neg_x.shape[0])[0]

        return adj, x_recon, adj_recon_list, target_z, pos_sub_z_pool, neg_sub_z_pool
    
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
num_features = dataset_AIDS.num_features
hidden_dims=[256, 128, 64]

model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion_node = torch.nn.L1Loss()
criterion_label = nn.BCELoss()
    
def train(model, train_loader, optimizer, criterion_node, criterion_label):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, x_recon, adj_recon_list, target_z, pos_sub_z_pool, neg_sub_z_pool = model(data)
        # adj_list = to_dense_adj(data.edge_index, batch=data.batch)
        
        loss = 0
        start_node = 0
        
        for i in range(len(data)): # 각 그래프별로 손실 계산
            num_nodes = (data.batch == i).sum().item() # 현재 그래프에 속하는 노드 수
            end_node = start_node + num_nodes
            
            node_loss_1 = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
            # node_loss_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
            node_loss = node_loss_1 / 100
            
            edge_loss_1 = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
            # edge_loss_2 = F.binary_cross_entropy(adj_recon_prime_list[i], adj[i])
            edge_loss = edge_loss_1 / 50
            
            # graph_pred.shape -> (batch_size, 2)
            # data.y maybe (batch_size)
            # class_loss_graph = criterion_label(graph_pred[i][0], data.y[i].float())
            # class_loss_pos_sub = criterion_label(pos_sub_graph_pred[i][0], data.y[i].float())
            # class_loss = (class_loss_graph + class_loss_pos_sub) / 10
            
            loss += node_loss + edge_loss
            
            start_node = end_node
        
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_pool, neg_sub_z_pool))
        loss += triplet_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


#%%
def evaluate_model(model, test_loader):
    model.eval()
    true_labels = []
    pred_probs = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # Move data to the MPS device
            adj, x_recon, adj_recon_list, target_z, pos_sub_z_pool, neg_sub_z_pool = model(data)  # 모델 예측값
            
            recon_error_list = []
            start_node = 0
        
            for i in range(len(data)): # 각 그래프별로 손실 계산
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() # 현재 그래프에 속하는 노드 수
                end_node = start_node + num_nodes

                node_recon_1 = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                # node_recon_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
                node_recon_error = node_recon_1 / 100
             
                edge_recon_1 = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
                # edge_recon_2 = F.binary_cross_entropy(adj_recon_prime_list[i], adj[i])
                edge_recon_error = edge_recon_1 / 50
            
                # graph_pred.shape -> (batch_size, 2)
                # data.y maybe (batch_size)
                # class_loss_graph = criterion_label(graph_pred[i][0], data.y[i].float())
                # class_loss_pos_sub = criterion_label(pos_sub_graph_pred[i][0], data.y[i].float())
                # class_loss = (class_loss_graph + class_loss_pos_sub) / 5
            
                recon_error += node_recon_error + edge_recon_error
                recon_error_list.append(recon_error)
                
                start_node = end_node

            recon_error_rank = torch.stack(recon_error_list)
            _, indices = torch.topk(recon_error_rank, (len(recon_error_list)*2) // 10)    
            y_pred = torch.ones_like(recon_error_rank)
            y_pred[indices] = 0
            
            # y_pred = torch.sigmoid(graph_pred)
            # y_pred = torch.argsort(torch.argsort(y_pred, dim=1, descending=True), dim=1)[:, 0]
            
            # 예측 확률과 실제 레이블을 리스트에 추가
            true_labels.extend(data.y.cpu().numpy())
            pred_probs.extend(y_pred.cpu().numpy())

    # AUC 점수 계산
    auc_score = roc_auc_score(true_labels, pred_probs)
    return auc_score


#%%
epochs = 100
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion_node, criterion_label)
    print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}')

    # 에포크마다 평가 수행
    auc_score = evaluate_model(model, test_loader)
    print(f'Epoch {epoch+1}: Validation AUC = {auc_score:.4f}')
    

# %%
