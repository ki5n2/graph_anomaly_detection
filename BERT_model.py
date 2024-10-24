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
from torch.utils.data import Dataset, DataLoader
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
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        
        # Query, Key, Value 변환을 위한 선형 레이어
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        
    def transpose_for_scores(self, x):
        # 입력 shape: (batch_size, num_nodes, hidden_size)
        batch_size, num_nodes = x.size()[:2]
        # (batch_size, num_nodes, hidden_size) ->
        # (batch_size, num_nodes, num_attention_heads, attention_head_size)
        
        x = x.view(batch_size, num_nodes, self.num_attention_heads, self.attention_head_size)
        # 최종 shape: (batch_size, num_attention_heads, num_nodes, attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # hidden_states shape: (batch_size, num_nodes, hidden_size)
        
        # Query, Key, Value 변환
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 어텐션 스코어 계산
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        # 마스크 적용
        if attention_mask is not None:
            # attention_mask shape: (batch_size, num_nodes)를 어텐션 shape에 맞게 확장
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, self.num_attention_heads, -1, -1)
            attention_mask = (1.0 - attention_mask) * -10000.0  # 마스킹된 위치에 큰 음수값
            attention_scores = attention_scores + attention_mask
        
        # 어텐션 확률 계산
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 최종 출력 계산
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.all_head_size)
        
        return context_layer
    

class MolecularGraphBERTLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # 자기 주의 메커니즘
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layernorm1(attention_output + hidden_states)
        
        # Feed-forward 네트워크
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(layer_output + attention_output)
        
        return layer_output


def create_node_mask(num_nodes, mask_prob=0.15):
    """
    노드 마스킹을 위한 마스크 생성
    mask_prob: 마스킹할 노드의 비율
    """
    mask = torch.ones(num_nodes)
    mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_prob)]
    mask[mask_indices] = 0
    return mask


#%%
num_node_features = 32
hidden_size = 256
num_attention_heads = 8
num_classes = 10 
batch_size = 32
num_epochs = 100
num_nodes = 50
    
layer = MolecularGraphBERTLayer(hidden_size, num_attention_heads)

hidden_states = torch.randn(batch_size, num_nodes, hidden_size)
attention_mask = torch.stack([create_node_mask(num_nodes) for _ in range(batch_size)]) # 0 마스크

# 순전파
output = layer(hidden_states, attention_mask)


#%%
# class MolecularGraphBERT(nn.Module):
#     def __init__(self, num_node_features, hidden_size, num_attention_heads, num_classes, num_layers=6, dropout_prob=0.1):
#         super().__init__()
        
#         self.node_embedding = nn.Linear(num_node_features, hidden_size)
        
#         # 여러 층의 트랜스포머 레이어
#         self.layers = nn.ModuleList([
#             MolecularGraphBERTLayer(hidden_size, num_attention_heads, dropout_prob)
#             for _ in range(num_layers)
#         ])
        
#         # 마스킹된 노드의 클래스를 예측하기 위한 분류기
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, num_classes)
#         )
        
#     def forward(self, node_features, attention_mask=None):
#         # 노드 임베딩
#         hidden_states = self.node_embedding(node_features)
        
#         # 트랜스포머 레이어 통과
#         for layer in self.layers:
#             hidden_states = layer(hidden_states, attention_mask)
            
#         # 분류기를 통한 예측
#         logits = self.classifier(hidden_states)
        
#         return logits


# #%%
# class PretrainingDataset(Dataset):
#     def __init__(self, node_features, node_labels, mask_prob=0.15):
#         self.node_features = node_features  # shape: (num_graphs, num_nodes, num_features)
#         self.node_labels = node_labels      # shape: (num_graphs, num_nodes)
#         self.mask_prob = mask_prob
        
#     def __len__(self):
#         return len(self.node_features)
    
#     def __getitem__(self, idx):
#         features = self.node_features[idx]
#         labels = self.node_labels[idx]
        
#         # 마스크 생성
#         mask = torch.ones_like(labels, dtype=torch.float)
#         mask_indices = torch.randperm(len(labels))[:int(len(labels) * self.mask_prob)]
#         mask[mask_indices] = 0
        
#         # 마스킹된 특성 생성 ([MASK] 토큰을 0으로 표현)
#         masked_features = features.clone()
#         masked_features[mask_indices] = 0
        
#         return {
#             'features': masked_features,
#             'mask': mask,
#             'labels': labels,
#             'mask_indices': mask_indices
#         }

# def pretrain_model(model, train_loader, num_epochs, device, learning_rate=1e-4):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()
    
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         correct_predictions = 0
#         total_masked_nodes = 0
        
#         for batch in train_loader:
#             # 데이터를 디바이스로 이동
#             features = batch['features'].to(device)
#             mask = batch['mask'].to(device)
#             labels = batch['labels'].to(device)
#             mask_indices = batch['mask_indices']
            
#             # 순전파
#             logits = model(features, mask)
            
#             # 마스킹된 노드의 예측값과 라벨만 사용하여 손실 계산
#             batch_size = features.size(0)
#             masked_logits = []
#             masked_labels = []
            
#             for i in range(batch_size):
#                 masked_logits.append(logits[i][mask_indices[i]])
#                 masked_labels.append(labels[i][mask_indices[i]])
            
#             masked_logits = torch.cat(masked_logits)
#             masked_labels = torch.cat(masked_labels)
            
#             loss = criterion(masked_logits, masked_labels)
            
#             # 역전파 및 옵티마이저 스텝
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # 통계 계산
#             total_loss += loss.item()
#             pred = masked_logits.argmax(dim=-1)
#             correct_predictions += (pred == masked_labels).sum().item()
#             total_masked_nodes += len(masked_labels)
        
#         # 에폭 결과 출력
#         avg_loss = total_loss / len(train_loader)
#         accuracy = correct_predictions / total_masked_nodes
#         print(f'Epoch {epoch+1}/{num_epochs}:')
#         print(f'Average Loss: {avg_loss:.4f}')
#         print(f'Accuracy: {accuracy:.4f}')
#         print('-------------------------')
    
#     return model


# def main():
#     num_node_features = 32
#     hidden_size = 256
#     num_attention_heads = 8
#     num_classes = 10 
#     batch_size = 32
#     num_epochs = 100
    
#     # 예시 데이터 생성 (실제로는 실제 분자 데이터를 사용)
#     num_graphs = 1000
#     num_nodes = 50
#     node_features = torch.randn(num_graphs, num_nodes, num_node_features)
#     node_labels = torch.randint(0, num_classes, (num_graphs, num_nodes))
    
#     # 데이터셋과 데이터로더 생성
#     dataset = PretrainingDataset(node_features, node_labels)
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # 디바이스 설정
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 모델 초기화 및 학습
#     model = MolecularGraphBERT(
#         num_node_features=num_node_features,
#         hidden_size=hidden_size,
#         num_attention_heads=num_attention_heads,
#         num_classes=num_classes
#     ).to(device)
    
#     # 사전학습 수행
#     pretrained_model = pretrain_model(model, train_loader, num_epochs, device)
    
#     # 학습된 모델 저장
#     torch.save(pretrained_model.state_dict(), 'pretrained_molecular_graph_bert.pth')

# if __name__ == '__main__':
#     main()
    
    
#%%
dataset_name = 'COX2'
n_cross_val = 5

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from copy import deepcopy


#%%
class MaskedNodePredictor(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_attention_heads, num_classes, dropout_prob=0.1):
        super().__init__()
        self.node_embedding = nn.Linear(num_node_features, hidden_size)
        self.layers = nn.ModuleList([
            MolecularGraphBERTLayer(hidden_size, num_attention_heads, dropout_prob)
            for _ in range(6)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )
        self.hidden_size = hidden_size
    
    def get_node_embeddings(self, node_features, attention_mask=None):
        hidden_states = self.node_embedding(node_features)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
    
    def forward(self, node_features, attention_mask=None):
        embeddings = self.get_node_embeddings(node_features, attention_mask)
        predictions = self.classifier(embeddings)
        return predictions, embeddings


# class GraphReconstructor(nn.Module):
#     def __init__(self, hidden_size, num_node_features):
#         super().__init__()
#         # 특성 재구성을 위한 디코더
#         self.feature_decoder = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, num_node_features)
#         )
        
#         # 인접행렬 재구성을 위한 디코더
#         self.adjacency_decoder = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, node_embeddings):
#         # 노드 특성 재구성
#         reconstructed_features = self.feature_decoder(node_embeddings)
        
#         # 인접행렬 재구성
#         num_nodes = node_embeddings.size(1)
#         row_embeddings = node_embeddings.unsqueeze(2).repeat(1, 1, num_nodes, 1)
#         col_embeddings = node_embeddings.unsqueeze(1).repeat(1, num_nodes, 1, 1)
#         pair_embeddings = torch.cat([row_embeddings, col_embeddings], dim=-1)
#         pair_embeddings = pair_embeddings.view(-1, pair_embeddings.size(-1))
#         adjacency_logits = self.adjacency_decoder(pair_embeddings)
#         reconstructed_adjacency = adjacency_logits.view(-1, num_nodes, num_nodes)
        
#         return reconstructed_features, reconstructed_adjacency


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
    

def train_node_predictor(model, train_loader, device, num_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            mask_indices = batch['mask_indices']
            
            # 순전파
            predictions, _ = model(features, mask)
            
            # 마스크된 노드의 예측값과 라벨만 사용
            batch_size = features.size(0)
            masked_predictions = []
            masked_labels = []
            
            for i in range(batch_size):
                masked_predictions.append(predictions[i][mask_indices[i]])
                masked_labels.append(labels[i][mask_indices[i]])
            
            masked_predictions = torch.cat(masked_predictions)
            masked_labels = torch.cat(masked_labels)
            
            # 손실 계산 및 역전파
            loss = criterion(masked_predictions, masked_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = masked_predictions.max(1)
            total += masked_labels.size(0)
            correct += predicted.eq(masked_labels).sum().item()
        
        if (epoch + 1) % 10 == 0:
            accuracy = 100. * correct / total
            print(f'Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, '
                  f'Accuracy = {accuracy:.2f}%')
    
    return model


def train_graph_reconstructor(node_predictor, reconstructor, train_loader, device, 
                            num_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=lr)
    
    reconstructor.train()
    node_predictor.eval()  # 학습된 노드 임베딩을 고정
    
    for epoch in range(num_epochs):
        total_loss = 0
        feature_losses = 0
        adjacency_losses = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            adjacency = batch['adjacency'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            # 노드 임베딩 추출 (기울기 계산 없이)
            with torch.no_grad():
                _, node_embeddings = node_predictor(features, mask)
            
            # 그래프 재구성
            reconstructed_features, reconstructed_adjacency = reconstructor(node_embeddings)
            
            # 손실 계산
            feature_loss = F.mse_loss(reconstructed_features, features)
            adjacency_loss = F.binary_cross_entropy(reconstructed_adjacency, adjacency)
            loss = feature_loss + adjacency_loss
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            feature_losses += feature_loss.item()
            adjacency_losses += adjacency_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Total Loss: {total_loss/len(train_loader):.4f}')
            print(f'Feature Loss: {feature_losses/len(train_loader):.4f}')
            print(f'Adjacency Loss: {adjacency_losses/len(train_loader):.4f}')
    
    return reconstructor


def run_sequential_training(dataset, node_predictor, reconstructor, device, 
                          batch_size=32, num_epochs_step1=100, num_epochs_step2=100):
    # 훈련 데이터 로더 생성
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Step 1: Training Node Predictor...")
    trained_predictor = train_node_predictor(
        node_predictor, 
        train_loader, 
        device, 
        num_epochs=num_epochs_step1
    )
    
    print("\nStep 2: Training Graph Reconstructor...")
    trained_reconstructor = train_graph_reconstructor(
        trained_predictor,
        reconstructor,
        train_loader,
        device,
        num_epochs=num_epochs_step2
    )
    
    return trained_predictor, trained_reconstructor

# 사용 예시
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 파라미터
    num_node_features = 32
    hidden_size = 256
    num_attention_heads = 8
    num_classes = 10
    
    # 모델 초기화
    node_predictor = MaskedNodePredictor(
        num_node_features=num_node_features,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_classes=num_classes
    ).to(device)
    
    reconstructor = GraphReconstructor(
        hidden_size=hidden_size,
        num_node_features=num_node_features
    ).to(device)
    
    # 데이터셋 준비
    dataset = YourDataset(...)
    
    # 순차적 학습 실행
    trained_predictor, trained_reconstructor = run_sequential_training(
        dataset=dataset,
        node_predictor=node_predictor,
        reconstructor=reconstructor,
        device=device
    )
    
    # 모델 저장
    torch.save(trained_predictor.state_dict(), 'node_predictor.pth')
    torch.save(trained_reconstructor.state_dict(), 'reconstructor.pth')

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

            adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / num_nodes
            adj_loss = adj_loss * adj_theta
            
            loss += adj_loss
            
            start_node = end_node
        
        print(f'train_adj_loss: {loss}')
     
        node_loss = (torch.norm(x_recon - x, p='fro')**2) / max_nodes
        node_loss = node_loss * node_theta
        print(f'train_node loss: {node_loss}')
        
        # z_node_loss = torch.norm(z_tilde - z_, p='fro')**2 / num_nodes
        # z_node_loss = z_node_loss * 0.3
        # print(f'train_z_node loss: {z_node_loss}z')
        
        # dissimmilar = calculate_dissimilarity(cluster_centers).item()
        # dissimmilar = dissimmilar * 10
        # print(f'dissimmilar: {dissimmilar}')
        
        loss += node_loss
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
            x, edge_index, batch, num_graphs, y_ = data.x, data.edge_index, data.batch, data.num_graphs, data.y

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
                min_distance = distances.min()

                # recon_error = node_loss.item() * 0.1 + adj_loss.item() * 1 + min_distance * 0.5
                recon_error = node_loss.item() * alpha + adj_loss.item() * beta + min_distance.item() * gamma
                recon_errors.append(recon_error)
                
                print(f'test_node_loss: {node_loss.item() * alpha }')
                print(f'test_adj_loss: {adj_loss.item() * beta }')
                print(f'test_min_distance: {min_distance.item() * gamma }')
                print(f'test_label: {y_[i].item() }')

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
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.001)

parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.025)
parser.add_argument("--gamma", type=float, default=0.25)
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
        

#%%
class GraphBertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        
        # WSP와 LE 각각에 d_model/2 차원을 할당
        # self.wsp_encoder = nn.Linear(max_nodes, d_model // 2)
        # self.le_encoder = nn.Linear(max_nodes, d_model // 2)
        self.wsp_encoder = nn.Linear(max_nodes, max_nodes)
        self.le_encoder = nn.Linear(max_nodes, max_nodes)
        # 비선형 활성화 함수와 추가 MLP 레이어 정의
        self.mlp = nn.Sequential(
            nn.Linear(max_nodes * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
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
    
    # def get_wsp_encoding_(self, edge_index, num_nodes):
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
        
        # MLP를 통한 비선형 변환 적용
        pos_encoding = self.mlp(pos_encoding)
        
        return pos_encoding
    

class HybridTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(HybridTransformerLayer, self).__init__()
        self.node_attn = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.graph_attn = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Node attention (within graphs)
        node_out = self.node_attn(src, src_mask, src_key_padding_mask)
        
        # Graph attention (across graphs for each node position)
        graph_in = src.transpose(0, 1)
        graph_mask = src_key_padding_mask.transpose(0, 1)
        graph_out = self.graph_attn(graph_in, src_mask, graph_mask)
        graph_out = graph_out.transpose(0, 1)
        graph_out[:,0,:] = 0.0
        # Combine outputs
        combined = self.norm(node_out + graph_out)
        return combined
    
    
class HybridTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(HybridTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            HybridTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask, src_key_padding_mask)
        return output
    
    
class TransformerEncoder_(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder_, self).__init__()
        self.positional_encoding = GraphBertPositionalEncoding(d_model, max_nodes)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True
        # )
        self.d_model = d_model
        self.transformer = HybridTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

        # self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer, num_layers)
        # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers)
        
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
        
        output = self.transformer(src_, src_key_padding_mask=src_key_padding_mask)
        
        # # 트랜스포머 인코딩
        # output = self.transformer_encoder1(src_, src_key_padding_mask=src_key_padding_mask)
        # src_T = src_.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        # src_key_padding_mask_T = src_key_padding_mask.transpose(0, 1)
        # output_T = self.transformer_encoder2(src_T, src_key_padding_mask=src_key_padding_mask_T)
        # output_T = output_T.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = GraphBertPositionalEncoding(d_model, max_nodes)
        # encoder_layer_n = nn.TransformerEncoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first=True
        # )
        encoder_layer_g = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu'
        )
        # self.transformer_encoder_n = nn.TransformerEncoder(encoder_layer_n, num_layers)
        self.transformer_encoder_g = nn.TransformerEncoder(encoder_layer_g, num_layers)
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
                padded_pos_encoding = torch.zeros(max_seq_len - 1, d_model).to(src.device)
            
            # CLS 토큰 인코딩과 노드 인코딩 결합
            full_pos_encoding = torch.cat([cls_pos_encoding, padded_pos_encoding], dim=0)
            pos_encodings.append(full_pos_encoding)
        
        # 모든 배치의 포지셔널 인코딩 결합
        pos_encoding_batch = torch.stack(pos_encodings)
        
        # 포지셔널 인코딩 추가
        src_ = src + pos_encoding_batch
        print(src_.shape)
        
        src_ = src_.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        print(src_.shape)
        print(src_key_padding_mask.shape)
        # src_key_padding_mask_ = src_key_padding_mask.transpose(0, 1)
        # print(src_key_padding_mask_.shape)
        output_ = self.transformer_encoder_g(src_, src_key_padding_mask=src_key_padding_mask)
        output = output_.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # output = self.transformer_encoder_n(output, src_key_padding_mask=src_key_padding_mask)
            
        return output


def perform_clustering(train_cls_outputs, random_seed, n_clusters):
    # train_cls_outputs가 이미 텐서이므로, 그대로 사용
    cls_outputs_tensor = train_cls_outputs  # [total_num_graphs, hidden_dim]
    cls_outputs_np = cls_outputs_tensor.detach().cpu().numpy()
    
    # K-Means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto").fit(cls_outputs_np)

    # 클러스터 중심 저장
    cluster_centers = kmeans.cluster_centers_

    return kmeans, cluster_centers


def mean_euclidean_distance_loss(outputs):
    # outputs: [batch_size, feature_dim] 형태의 텐서
    n = outputs.size(0)
    
    # 모든 쌍 사이의 차이를 계산
    diffs = outputs.unsqueeze(1) - outputs.unsqueeze(0)
    
    # 유클리디안 거리 계산
    distances = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-9)  # 1e-9는 0으로 나누는 것을 방지
    
    # 행렬의 상삼각 부분만을 이용해 중복 계산 방지
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    
    # 평균 거리 계산
    mean_distance = torch.mean(distances[mask])
    
    return mean_distance


def calculate_dissimilarity(matrix):
    num_points = matrix.shape[0]
    dissimilarity = 0.0
    
    # Loop through all pairs of rows in the matrix
    for i in range(num_points):
        for j in range(i + 1, num_points):
            # Calculate Euclidean distance between rows i and j
            distance = np.linalg.norm(matrix[i] - matrix[j])
            # Accumulate the distance
            dissimilarity += distance
    
    return dissimilarity


#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.transformer = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=8,
            num_layers=2,
            dim_feedforward=hidden_dims[-1] * 4,
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

        # cls_output = encoded_T[:, 0, :]       # [batch_size, hidden_dim]
        # node_output_T = encoded_T[:, 1:, :]     # [batch_size, max_nodes, hidden_dim]
        
        # node_output = node_output_ + node_output_T
        
        node_output_list = []
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            node_output_list.append(node_output[i, :num_nodes, :])

        u = torch.cat(node_output_list, dim=0)  # [total_num_nodes, hidden_dim]
        # node_output_concat = torch.cat(node_output_list, dim=0)

        u_prime = self.u_mlp(u)
        
        x_recon = self.feature_decoder(u_prime)
        # x_recon = self.feature_decoder(node_output_concat)
                
        adj_recon_list = []
        idx = 0
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            z_graph = u_prime[idx:idx + num_nodes]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)
            idx += num_nodes
        
        new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch).to(device)
        z_tilde = self.encoder(x_recon, new_edge_index)
        
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
'''RUN'''
def run(dataset_name, random_seed, dataset_AN, split=None, device=device):
    all_results = []
    set_seed(random_seed)

    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']

    model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)

    train_loader = loaders['train']
    test_loader = loaders['test']

    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []

    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, optimizer, max_nodes, device)
        
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

    for trial in range(2):
        fold_start = time.time()  # 현재 폴드 시작 시간

        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc = run(dataset_name, random_seed, dataset_AN, split=splits[trial])
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
