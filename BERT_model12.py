#%%
'''PRESENT'''
print('이번 BERT 모델 19은 AIDS, BZR, COX2, DHFR에 대한 실험 파일입니다. 마스크 토큰 재구성을 통한 프리-트레인이 이루어집니다. 이후 기존과 같이 노드 특성을 재구성하는 모델을 통해 이상을 탐지합니다. 기존 파일과 다른 점은 성능 평가 결과 비교를 코드 내에서 수행하고자 하였으며, 해당 파일만 실행하면 결과까지 한 번에 볼 수 있도록 하였습니다. 또한, 재구성 오류에 대한 정규화가 이루어져 있습니다. 추가로 훈련 데이터에 대한 산점도와 로그 스케일이 적용되어 있습니다. 그리고 2D density estimation이 적용되어 있습니다. 그리고 TopER 과정이 반영되어 있습니다. 밀도 기반 이상 스코어. 프리트레인 과정이 변경되었습니다.')

#%%
'''IMPORTS'''
import os
import re
import sys
import json
import math
import time
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch

from scipy.linalg import eigh
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph, KernelDensity
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, silhouette_score, silhouette_samples

from functools import partial
from multiprocessing import Pool

from module.loss import loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, compute_persistence_and_betti, process_batch_graphs, loocv_bandwidth_selection

import networkx as nx


#%%
'''TRAIN BERT'''
def train_bert_embedding(model, train_loader, bert_optimizer, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        # 마스크 생성
        mask_indices = torch.rand(x.size(0), device=device) < 0.15  # 15% 노드 마스킹
        
        # BERT 인코딩 및 마스크 토큰 예측
        node_embeddings, masked_outputs = model(
            x, edge_index, batch, num_graphs,
            mask_indices=mask_indices,
            is_pretrain=True
        )
        
        # 마스크된 노드의 특징 재구성 손실 계산
        mask_loss = torch.norm(masked_outputs - x[mask_indices], p='fro')**2 / mask_indices.sum()
        
        mask_loss.backward()
        bert_optimizer.step()
        total_loss += mask_loss.item()
        num_sample += num_graphs
        
        print(f'mask_node_feature:{mask_loss.item()}')
    
    return total_loss / len(train_loader), num_sample


#%%
def train_bert_edge_reconstruction(model, train_loader, bert_optimizer, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        # Edge reconstruction 수행
        node_embeddings, adj_recon_list = model(
            x, edge_index, batch, num_graphs,
            is_pretrain=True,
            edge_training=True
        )
        
        start_idx = 0
        edge_loss = 0
        
        for i in range(num_graphs):
            # 현재 그래프의 노드 수 계산
            mask = (batch == i)
            num_nodes = mask.sum().item()
            end_idx = start_idx + num_nodes
            
            # 현재 그래프의 edge_index 추출 및 조정
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)]
            graph_edges = graph_edges - start_idx
            
            # 실제 adjacency matrix 생성 (max_nodes 크기로)
            true_adj = torch.zeros((model.encoder.edge_decoder.max_nodes, 
                                  model.encoder.edge_decoder.max_nodes), 
                                 device=device)
            true_adj[graph_edges[0], graph_edges[1]] = 1
            true_adj = true_adj + true_adj.t()
            true_adj = (true_adj > 0).float()
            
            # 손실 계산 (실제 노드가 있는 부분만)
            adj_recon = adj_recon_list[i]  # 이미 max_nodes 크기로 패딩된 상태
            
            # 실제 노드가 있는 부분의 마스크 생성
            node_mask = torch.zeros_like(adj_recon, dtype=torch.bool)
            node_mask[:num_nodes, :num_nodes] = True
            
            # MSE 손실 계산 (실제 노드 영역만)
            mse_loss = torch.sum((adj_recon[node_mask] - true_adj[node_mask]) ** 2) / (node_mask.sum())
            edge_loss += mse_loss
            
            start_idx = end_idx
        
        edge_loss = edge_loss / num_graphs
        
        edge_loss.backward()
        bert_optimizer.step()
        
        total_loss += edge_loss.item()
        num_sample += num_graphs
        
        print(f'edge_reconstruction_mse_loss: {edge_loss.item()}')
    
    return total_loss / len(train_loader), num_sample


#%%
# def train(model, train_loader, recon_optimizer, device, epoch, dataset_name, cluster_centers=None):
#     model.train()
#     total_loss = 0
#     num_sample = 0
#     reconstruction_errors = []
    
#     for data in train_loader:
#         data = process_batch_graphs(data)
#         data = data.to(device)
#         recon_optimizer.zero_grad()
#         x, edge_index, batch, num_graphs, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.true_stats
        
#         # Forward pass
#         train_cls_outputs, x_recon, stats_pred, adj_recon_list = model(
#             x, edge_index, batch, num_graphs, is_pretrain=False
#         )
        
#         # 손실 계산 및 역전파
#         loss = 0
#         node_loss = 0
#         edge_loss = 0
#         stats_loss = 0
#         start_node = 0
        
#         for i in range(num_graphs):
#             num_nodes = (batch == i).sum().item()
#             end_node = start_node + num_nodes

#             if dataset_name == 'AIDS':
#                 node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
#                 stats_loss_ = persistence_stats_loss(stats_pred, true_stats)
#             else:
#                 node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
#                 stats_loss_ = persistence_stats_loss(stats_pred, true_stats) / num_nodes
                
#             node_loss += node_loss_
#             stats_loss += stats_loss_ 
            
#             # 엣지 재구성 손실 (이상 탐지 모델의 decoder 사용)
#             graph_edges = edge_index[:, (edge_index[0] >= start_node) & (edge_index[0] < end_node)]
#             graph_edges = graph_edges - start_node
#             true_adj = torch.zeros((model.edge_recon.max_nodes, 
#                                   model.edge_recon.max_nodes), 
#                                  device=device)
#             true_adj[graph_edges[0], graph_edges[1]] = 1
#             true_adj = true_adj + true_adj.t()
#             true_adj = (true_adj > 0).float()
            
#             node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
#             node_mask[:num_nodes, :num_nodes] = True
#             edge_loss_ = torch.sum(
#                 (adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2
#             ) / node_mask.sum()
            
#             edge_loss += edge_loss_
            
#             start_node = end_node
        
#         beta_ = 0.5   # 엣지 재구성 가중치
#         edge_loss = beta_ * edge_loss
        
#         gamma_ = 1.0
#         stats_loss = gamma_ * stats_loss
        
#         loss = node_loss + edge_loss + stats_loss
        
#         print(f'node_loss: {node_loss}')
#         print(f'edge_loss: {edge_loss}')
#         print(f'stats_loss: {stats_loss}')
        
#         num_sample += num_graphs
#         loss.backward()
#         recon_optimizer.step()
#         total_loss += loss.item()
        
#     # 모델 업데이트 완료 후 산점도 데이터 수집
#     model.eval()  # 평가 모드로 전환
#     with torch.no_grad():
#         for data in train_loader:
#             data = process_batch_graphs(data)
#             data = data.to(device)
#             train_cls_outputs, x_recon, stats_pred, adj_recon_list = model(
#                 x, edge_index, batch, num_graphs, is_pretrain=False
#             )
            
#             # 산점도 데이터 수집
#             if epoch % 5 == 0:
#                 start_node = 0
#                 for i in range(num_graphs):
#                     num_nodes = (batch == i).sum().item()
#                     end_node = start_node + num_nodes
                    
#                     if dataset_name == 'AIDS':
#                         node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
#                         stats_loss_ = persistence_stats_loss(stats_pred, true_stats)
#                     else:
#                         node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
#                         stats_loss_ = persistence_stats_loss(stats_pred, true_stats) / num_nodes
#                     # 엣지 재구성 손실 (이상 탐지 모델의 decoder 사용)
            
#                     graph_edges = edge_index[:, (edge_index[0] >= start_node) & (edge_index[0] < end_node)]
#                     graph_edges = graph_edges - start_node
#                     true_adj = torch.zeros((model.edge_recon.max_nodes, 
#                                         model.edge_recon.max_nodes), 
#                                         device=device)
#                     true_adj[graph_edges[0], graph_edges[1]] = 1
#                     true_adj = true_adj + true_adj.t()
#                     true_adj = (true_adj > 0).float()
                    
#                     node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
#                     node_mask[:num_nodes, :num_nodes] = True
#                     edge_loss_ = torch.sum(
#                         (adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2
#                     ) / node_mask.sum()
                    
#                     node_loss_scaled = node_loss_.item() * alpha
                    
#                     edge_loss__ = edge_loss_.item() * beta    

#                     stats_loss_scaled = stats_loss_.item() * gamma
                    
#                     reconstruction_errors.append({
#                         'reconstruction': node_loss_scaled + edge_loss__,
#                         'topology': stats_loss_scaled,
#                         'type': 'train_normal'
#                     })
#                     start_node = end_node
                
#     model.train()  # 다시 훈련 모드로 전환
    
#     if epoch % 5 == 0:
#         # Create figure with two subplots
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
#         recon_errors = [point['reconstruction'] for point in reconstruction_errors]
#         cluster_errors = [point['topology'] for point in reconstruction_errors]
        
#         # Normal scale plot
#         ax1.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
#         ax1.set_xlabel('Reconstruction Error')
#         ax1.set_ylabel('Topology')
#         ax1.set_title(f'Training Error Distribution (Epoch {epoch})')
#         ax1.grid(True)

#         # Log scale plot
#         ax2.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
#         ax2.set_xlabel('Reconstruction Error')
#         ax2.set_ylabel('Topology')
#         ax2.set_title(f'Training Error Distribution - Log Scale (Epoch {epoch})')
#         ax2.set_xscale('log')
#         ax2.set_yscale('log')
#         ax2.grid(True)

#         plt.tight_layout()
#         save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection//error_distribution_plot/plot/{dataset_name}_time_{current_time}/train_error_distribution_epoch_{epoch}.png'
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path)
#         plt.close()
        
#     return total_loss / len(train_loader), num_sample, cluster_centers, reconstruction_errors


#%%
def train(model, train_loader, recon_optimizer, device, epoch, dataset_name, cluster_centers=None):
    model.train()
    total_loss = 0
    num_sample = 0
    reconstruction_errors = []
    
    # 학습 단계
    for data in train_loader:
        data = process_batch_graphs(data)
        data = data.to(device)
        recon_optimizer.zero_grad()
        x, edge_index, batch, num_graphs, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.true_stats
        
        # Forward pass
        train_cls_outputs, x_recon, stats_pred, adj_recon_list = model(
            x, edge_index, batch, num_graphs, is_pretrain=False
        )
        
        if epoch % 5 == 0:
            cls_outputs_np = train_cls_outputs.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_cluster, random_state=random_seed)
            kmeans.fit(cls_outputs_np)
            cluster_centers = kmeans.cluster_centers_
        
        loss = 0
        node_loss = 0
        edge_loss = 0
        stats_loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            if dataset_name == 'AIDS':
                node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
            else:
                node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            
            stats_loss_ = persistence_stats_loss(stats_pred[i], true_stats[i])

            node_loss += node_loss_
            stats_loss += stats_loss_ 

            # 엣지 재구성 손실
            graph_edges = edge_index[:, (edge_index[0] >= start_node) & (edge_index[0] < end_node)]
            graph_edges = graph_edges - start_node
            true_adj = torch.zeros((model.edge_recon.max_nodes, 
                                  model.edge_recon.max_nodes), 
                                 device=device)
            true_adj[graph_edges[0], graph_edges[1]] = 1
            true_adj = true_adj + true_adj.t()
            true_adj = (true_adj > 0).float()
            
            node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
            node_mask[:num_nodes, :num_nodes] = True
            edge_loss_ = torch.sum(
                (adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2
            ) / node_mask.sum()
            
            edge_loss += edge_loss_
            start_node = end_node
                    
        beta_ = 0.5   # 엣지 재구성 가중치
        edge_loss = beta_ * edge_loss
        
        gamma_ = 10.0
        stats_loss = gamma_ * stats_loss
        
        loss = node_loss + edge_loss + stats_loss
        
        print(f'node_loss: {node_loss}')
        print(f'edge_loss: {edge_loss}')
        print(f'stats_loss: {stats_loss}')
        
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    # 모델 업데이트 완료 후 산점도 데이터 수집
    model.eval()  # 평가 모드로 전환
    with torch.no_grad():
        for data in train_loader:
            data = process_batch_graphs(data)
            data = data.to(device)
            x, edge_index, batch, num_graphs, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.true_stats
            
            # Forward pass
            train_cls_outputs, x_recon, stats_pred, adj_recon_list = model(
                x, edge_index, batch, num_graphs, is_pretrain=False
            )
            
            # 산점도 데이터 수집
            if epoch % 5 == 0:
                start_node = 0
                for i in range(num_graphs):
                    num_nodes = (batch == i).sum().item()
                    end_node = start_node + num_nodes

                    if dataset_name == 'AIDS':
                        node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
                    else:
                        node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                    
                    stats_loss_ = persistence_stats_loss(stats_pred[i], true_stats[i])
                    
                    node_loss_scaled = node_loss_.item() * alpha
                    
                    graph_edges = edge_index[:, (edge_index[0] >= start_node) & (edge_index[0] < end_node)]
                    graph_edges = graph_edges - start_node
                    true_adj = torch.zeros((model.edge_recon.max_nodes, 
                                          model.edge_recon.max_nodes), 
                                         device=device)
                    true_adj[graph_edges[0], graph_edges[1]] = 1
                    true_adj = true_adj + true_adj.t()
                    true_adj = (true_adj > 0).float()
                    
                    node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
                    node_mask[:num_nodes, :num_nodes] = True
                    edge_loss_ = torch.sum(
                        (adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2
                    ) / node_mask.sum()
                    edge_loss__ = edge_loss_.item() * beta

                    stats_loss_scaled = stats_loss_.item() * gamma
                    
                    reconstruction_errors.append({
                        'reconstruction': node_loss_scaled + edge_loss__,
                        'topology': stats_loss_scaled,
                        'type': 'train_normal'
                    })
                    
                    start_node = end_node

    if epoch % 5 == 0:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        recon_errors = [point['reconstruction'] for point in reconstruction_errors]
        cluster_errors = [point['topology'] for point in reconstruction_errors]
        
        # Normal scale plot
        ax1.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Topology')
        ax1.set_title(f'Training Error Distribution (Epoch {epoch})')
        ax1.grid(True)

        # Log scale plot
        ax2.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Topology')
        ax2.set_title(f'Training Error Distribution - Log Scale (Epoch {epoch})')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True)

        plt.tight_layout()
        save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/train_error_distribution_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    
    model.train()  # 다시 훈련 모드로 전환
    return total_loss / len(train_loader), num_sample, cluster_centers, reconstruction_errors


#%%
def evaluate_model(model, test_loader, cluster_centers, n_clusters, gamma_clusters, random_seed, reconstruction_errors, epoch, dataset_name, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors_test = []  # 새로 추가
    
    with torch.no_grad():
        for data in test_loader:
            data = process_batch_graphs(data)
            data = data.to(device)
            x, edge_index, batch, num_graphs, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.true_stats
            # Forward pass - evaluation mode
            e_cls_output, x_recon, stats_pred, adj_recon_list = model(
                x, edge_index, batch, num_graphs, is_pretrain=False
            )
            
            e_cls_outputs_np = e_cls_output.detach().cpu().numpy()  # [num_graphs, hidden_dim]
            
            recon_errors = []
            start_idx = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_idx = start_idx + num_nodes
                
                # Reconstruction error 계산
                if dataset_name == 'AIDS':
                    node_loss = torch.norm(x[start_idx:end_idx] - x_recon[start_idx:end_idx], p='fro')**2
                else:
                    node_loss = torch.norm(x[start_idx:end_idx] - x_recon[start_idx:end_idx], p='fro')**2 / num_nodes
                    
                stats_loss = persistence_stats_loss(stats_pred[i], true_stats[i]) 
                    
                node_loss = node_loss.item() * alpha
                stats_loss = stats_loss.item() * gamma
                    
                # 2. 엣지 재구성 오류
                graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)]
                graph_edges = graph_edges - start_idx
                true_adj = torch.zeros((model.edge_recon.max_nodes, 
                                      model.edge_recon.max_nodes), 
                                     device=device)
                true_adj[graph_edges[0], graph_edges[1]] = 1
                true_adj = true_adj + true_adj.t()
                true_adj = (true_adj > 0).float()
                
                node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
                node_mask[:num_nodes, :num_nodes] = True
                edge_loss = torch.sum(
                    (adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2
                ) / node_mask.sum()
                edge_loss = edge_loss.item() * beta
                
                # cls_vec = e_cls_outputs_np[i].reshape(1, -1)
                # distances = cdist(cls_vec, cluster_centers, metric='euclidean')
                # min_distance = distances.min().item() * gamma

                # 전체 에러는 변환된 값들의 평균으로 계산
                recon_error = node_loss + edge_loss + stats_loss          
                
                # 변환된 값들 저장
                reconstruction_errors_test.append({
                    'reconstruction': node_loss + edge_loss,
                    # 'edge_reconstruction': edge_loss,
                    'topology': stats_loss,
                    'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
                })
                
                print(f'test_node_loss: {node_loss}')
                print(f'test_edge_loss: {edge_loss}')
                print(f'test_stats_loss: {stats_loss}')
                
                recon_errors.append(recon_error)
                
                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error
                    
                start_idx = end_idx
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())
    
    # 시각화를 위한 데이터 변환
    visualization_data = {
        'normal': [
            {'reconstruction': error['reconstruction'], 
             'topology': error['topology']}
            for error in reconstruction_errors_test if error['type'] == 'test_normal'
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'topology': error['topology']}
            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'
        ]
    }

    # 데이터 분리 및 특징 벡터 구성
    train_data = np.array([[error['reconstruction'], error['topology']] 
                          for error in reconstruction_errors if error['type'] == 'train_normal'])
    test_normal = np.array([[error['reconstruction'], error['topology']] 
                           for error in reconstruction_errors_test if error['type'] == 'test_normal'])
    test_anomaly = np.array([[error['reconstruction'], error['topology']] 
                            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'])
    
    # 밀도 기반 스코어링 적용
    # # Scott의 규칙 적용
    # bandwidth = scott_rule_bandwidth(train_data)
    bandwidth, _ = loocv_bandwidth_selection(train_data)
    print(f'bandwidth : {bandwidth}')
    # LOOCV 적용
    density_scorer = DensityBasedScoring(bandwidth=bandwidth)
    density_scorer.fit(train_data)
    
    # 이상 스코어 계산
    normal_scores = density_scorer.score_samples(test_normal)
    anomaly_scores = density_scorer.score_samples(test_anomaly)
    
    # 전체 스코어 및 라벨 구성
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    
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
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 원본 산점도와 밀도 등고선
    contour = density_scorer.plot_density_contours(train_data, test_normal, test_anomaly, ax1)
    fig.colorbar(contour, ax=ax1)
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Topology Error')
    ax1.set_title('Density-based Anomaly Detection')
    ax1.legend()
    
    # 이상 스코어 분포
    ax2.hist(normal_scores, bins=50, alpha=0.5, density=True, label='Normal', color='green')
    ax2.hist(anomaly_scores, bins=50, alpha=0.5, density=True, label='Anomaly', color='red')
    ax2.set_xlabel('Anomaly Score (-log density)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Anomaly Scores')
    ax2.legend()
    
    plt.tight_layout()
    save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/density_analysis/{dataset_name}_time_{current_time}/epoch_{epoch}_fold_{trial}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # 시각화를 위한 데이터 변환
    visualization_data = {
        'normal': [
            {'reconstruction': error['reconstruction'], 
             'topology': error['topology']}
            for error in reconstruction_errors_test if error['type'] == 'test_normal'
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'topology': error['topology']}
            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'
        ]
    }
    
    total_loss_mean = total_loss_ / sum(all_labels == 0)
    total_loss_anomaly_mean = total_loss_anomaly_ / sum(all_labels == 1)
    
    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, all_scores, all_labels, reconstruction_errors_test, visualization_data


def persistence_stats_loss(pred_stats, true_stats):
    continuous_loss = F.mse_loss(pred_stats, true_stats)
    
    return continuous_loss


#%%
def plot_error_distribution(train_errors, test_errors, epoch, trial, dataset_name, current_time):
    # 데이터 분리
    train_normal_recon = [e['reconstruction'] for e in train_errors if e['type'] == 'train_normal']
    train_normal_cluster = [e['topology'] for e in train_errors if e['type'] == 'train_normal']
    
    test_normal_recon = [e['reconstruction'] for e in test_errors if e['type'] == 'test_normal']
    test_normal_cluster = [e['topology'] for e in test_errors if e['type'] == 'test_normal']
    
    test_anomaly_recon = [e['reconstruction'] for e in test_errors if e['type'] == 'test_anomaly']
    test_anomaly_cluster = [e['topology'] for e in test_errors if e['type'] == 'test_anomaly']

    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 일반 스케일 플롯
    ax1.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax1.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax1.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Topology Error')
    ax1.set_title(f'Error Distribution (Epoch {epoch})')
    ax1.legend()
    ax1.grid(True)

    # 로그 스케일 플롯
    ax2.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax2.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax2.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Topology Error')
    ax2.set_title(f'Error Distribution - Log Scale (Epoch {epoch})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/combined_error_distribution_epoch_{epoch}_fold_{trial}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # JSON으로 데이터 저장
    error_data = {
        'train_normal': [{'reconstruction': r, 'topology': c} 
                        for r, c in zip(train_normal_recon, train_normal_cluster)],
        'test_normal': [{'reconstruction': r, 'topology': c} 
                       for r, c in zip(test_normal_recon, test_normal_cluster)],
        'test_anomaly': [{'reconstruction': r, 'topology': c} 
                        for r, c in zip(test_anomaly_recon, test_anomaly_cluster)]
    }
    
    json_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/combined_error_distribution_epoch_{epoch}_fold_{trial}.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(error_data, f)


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
    
    def plot_density_contours(self, X_train, X_test_normal, X_test_anomaly, ax, num_points=100):
        """
        밀도 등고선과 데이터 포인트를 시각화
        Args:
            X_train: 학습 데이터 (정상)
            X_test_normal: 테스트 데이터 (정상)
            X_test_anomaly: 테스트 데이터 (이상)
            ax: matplotlib axis
            num_points: 그리드 포인트 수
        """
        # 그리드 생성
        x_min = min(X_train[:, 0].min(), X_test_normal[:, 0].min(), X_test_anomaly[:, 0].min()) - 0.1
        x_max = max(X_train[:, 0].max(), X_test_normal[:, 0].max(), X_test_anomaly[:, 0].max()) + 0.1
        y_min = min(X_train[:, 1].min(), X_test_normal[:, 1].min(), X_test_anomaly[:, 1].min()) - 0.1
        y_max = max(X_train[:, 1].max(), X_test_normal[:, 1].max(), X_test_anomaly[:, 1].max()) + 0.1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_points),
                            np.linspace(y_min, y_max, num_points))
        
        # 그리드 포인트에서 밀도 계산
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.score_samples(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 등고선 플롯
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = ax.contourf(xx, yy, Z, levels=levels, cmap='RdYlBu_r', alpha=0.7)
        
        # 데이터 포인트 플롯
        scatter_params = {
            'alpha': 0.7,
            'edgecolor': 'white',
            's': 80,
            'linewidth': 1.5
        }
        
        ax.scatter(X_train[:, 0], X_train[:, 1],
                c='dodgerblue', label='Train (Normal)',
                marker='o', **scatter_params)
        ax.scatter(X_test_normal[:, 0], X_test_normal[:, 1],
                c='mediumseagreen', label='Test (Normal)',
                marker='o', **scatter_params)
        ax.scatter(X_test_anomaly[:, 0], X_test_anomaly[:, 1],
                c='crimson', label='Test (Anomaly)',
                marker='o', **scatter_params)
        
        # 축 레이블과 그리드 설정
        ax.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax.set_ylabel('Topology Error', fontsize=12, fontweight='bold')
        
        # 그리드 스타일 개선
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)  # 그리드를 데이터 포인트 아래로
        
        # 범례 스타일 개선
        ax.legend(fontsize=10, frameon=True, facecolor='white', 
                edgecolor='gray', loc='upper right',
                bbox_to_anchor=(1.0, 1.0))
        
        # 축 스타일 개선
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
        
        return contour


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
parser.add_argument("--BERT-epochs", type=int, default=30)
parser.add_argument("--epochs", type=int, default=300)
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
parser.add_argument("--beta", type=float, default=5.0)
parser.add_argument("--gamma", type=float, default=1000.0)
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
class BertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, d_model, nhead, num_layers, max_nodes, dropout_rate=0.1):
        super().__init__()
        self.gcn_encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.positional_encoding = GraphBertPositionalEncoding(hidden_dims[-1], max_nodes)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dims[-1], nhead, hidden_dims[-1] * 4, dropout_rate, activation='gelu', batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dims[-1]))
        self.predicter = nn.Linear(hidden_dims[-1], num_features)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.edge_decoder = BilinearEdgeDecoder(max_nodes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.d_model = d_model
        
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=False, edge_training=False):
        h = self.gcn_encoder(x, edge_index)
        
        # 배치 처리
        z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(h, edge_index, batch)
        
        # 3. 각 그래프에 대해 포지셔널 인코딩 계산
        pos_encoded_list = []
        for i, (z_graph, edge_idx) in enumerate(zip(z_list, edge_index_list)):
            # 포지셔널 인코딩 계산
            pos_encoding = self.positional_encoding(edge_idx, z_graph.size(0))
            # 인코딩 적용
            z_graph_with_pos = z_graph + pos_encoding
            pos_encoded_list.append(z_graph_with_pos)
            
        z_with_cls_batch, padding_mask = BatchUtils.add_cls_token(
            pos_encoded_list, self.cls_token, max_nodes_in_batch, x.device
        )
                
        # 5. 마스킹 적용
        if training and mask_indices is not None:
            mask_positions = torch.zeros_like(padding_mask)
            start_idx = 0
            for i in range(len(z_list)):
                num_nodes = z_list[i].size(0)
                graph_mask_indices = mask_indices[start_idx:start_idx + num_nodes]
                mask_positions[i, 1:num_nodes+1] = graph_mask_indices
                node_indices = mask_positions[i].nonzero().squeeze(-1)
                # mask_token = mask_token.to(device)
                z_with_cls_batch[i, node_indices] = self.mask_token
                padding_mask[i, num_nodes+1:] = True
                start_idx += num_nodes
                
        # Transformer 처리
        transformed = self.transformer(
            z_with_cls_batch,
            src_key_padding_mask=padding_mask
        )
        
        if edge_training == False:
            node_embeddings, masked_outputs = self._process_outputs(
                transformed, batch, mask_positions if training and mask_indices is not None else None
                )
        else:
            # 결과 추출
            node_embeddings, _ = self._process_outputs(
                transformed, batch, mask_positions=None
            )
        
        if training and edge_training:
            adj_recon_list = []
            idx = 0
            for i in range(num_graphs):
                num_nodes = z_list[i].size(0)
                z_graph = node_embeddings[idx:idx + num_nodes]
                adj_recon = self.edge_decoder(z_graph)
                adj_recon_list.append(adj_recon)
                idx += num_nodes
            
        if training:
            if edge_training:
                return node_embeddings, adj_recon_list
            else:
                return node_embeddings, masked_outputs

        return node_embeddings

    def _apply_masking(self, z_with_cls_batch, padding_mask, batch, mask_indices):
        batch_size = z_with_cls_batch.size(0)
        mask_positions = torch.zeros_like(padding_mask)
        start_idx = 0
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            graph_mask_indices = mask_indices[start_idx:start_idx + num_nodes]
            mask_positions[i, 1:num_nodes+1] = graph_mask_indices
            node_indices = mask_positions[i].nonzero().squeeze(-1)
            z_with_cls_batch[i, node_indices] = self.mask_token
            padding_mask[i, num_nodes+1:] = True
            start_idx += num_nodes
            
        return mask_positions

    def _process_outputs(self, transformed, batch, mask_positions=None):
        node_embeddings = []
        masked_outputs = []
        batch_size = transformed.size(0)
        start_idx = 0
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            # CLS 토큰을 제외한 노드 임베딩 추출
            graph_encoded = transformed[i, 1:num_nodes+1]
            node_embeddings.append(graph_encoded)
            
            # 전체 노드에 대해 예측 수행
            all_predictions = self.predicter(graph_encoded)
            
            # 마스크된 위치의 예측값만 선택
            if mask_positions is not None:
                current_mask_positions = mask_positions[i, 1:num_nodes+1]
                if current_mask_positions.any():
                    masked_predictions = all_predictions[current_mask_positions]
                    masked_outputs.append(masked_predictions)
            
            start_idx += num_nodes
        
        node_embeddings = torch.cat(node_embeddings, dim=0)
        
        # 전체 예측값과 마스크된 위치의 예측값 반환
        if mask_positions is not None and masked_outputs:
            return node_embeddings, torch.cat(masked_outputs, dim=0)
        return node_embeddings, None

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


class GraphBertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        
        self.wsp_encoder = nn.Linear(max_nodes, d_model // 2)
        self.le_encoder = nn.Linear(max_nodes, d_model // 2)
        
    def get_wsp_encoding(self, edge_index, num_nodes):
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(edge_index_np[0], edge_index_np[1]))
        
        spl_matrix = torch.zeros((num_nodes, self.max_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(G, source=i, target=j)
                    except nx.NetworkXNoPath:
                        path_length = self.max_nodes
                    if j < self.max_nodes:
                        spl_matrix[i, j] = path_length
                        
        return spl_matrix.to(edge_index.device)
    
    def get_laplacian_encoding(self, edge_index, num_nodes):
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to_dense()
        
        L_np = L.cpu().numpy()
        _, eigenvecs = eigh(L_np)
        le_matrix = torch.from_numpy(eigenvecs).float().to(edge_index.device)
        
        padded_le = torch.zeros((num_nodes, self.max_nodes), device=edge_index.device)
        padded_le[:, :num_nodes] = le_matrix
        
        return padded_le
    
    def forward(self, edge_index, num_nodes):
        wsp_matrix = self.get_wsp_encoding(edge_index, num_nodes)
        wsp_encoding = self.wsp_encoder(wsp_matrix)
        
        le_matrix = self.get_laplacian_encoding(edge_index, num_nodes)
        le_encoding = self.le_encoder(le_matrix)
        
        return torch.cat([wsp_encoding, le_encoding], dim=-1)
    

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
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT, nhead, 
                 num_layers_BERT, num_layers, dropout_rate=0.1):
        super().__init__()
        self.encoder = BertEncoder(
            num_features=num_features,
            hidden_dims=hidden_dims,
            d_model=hidden_dims[-1],
            nhead=nhead_BERT,
            num_layers=num_layers_BERT,
            max_nodes=max_nodes,
            dropout_rate=dropout_rate
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
        self.edge_recon = BilinearEdgeDecoder(max_nodes)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, is_pretrain=False, edge_training=False):
        if is_pretrain:
            if edge_training:
                # Edge reconstruction pretraining
                node_embeddings, adj_recon_list = self.encoder(
                    x, edge_index, batch, num_graphs,
                    training=True,
                    edge_training=True
                )
                return node_embeddings, adj_recon_list
            else:
                # Mask token reconstruction pretraining
                node_embeddings, masked_outputs = self.encoder(
                    x, edge_index, batch, num_graphs,
                    mask_indices=mask_indices,
                    training=True,
                    edge_training=False
                )
                return node_embeddings, masked_outputs
        
        else:
            # Fine-tuning phase (이상 탐지)
            node_embeddings = self.encoder(
                x, edge_index, batch, num_graphs,
                training=False,
                edge_training=False
            )
            
            # 배치 처리
            z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(
                node_embeddings, edge_index, batch, num_graphs
            )
            
            # CLS 토큰 추가
            z_with_cls_batch, mask = BatchUtils.add_cls_token(
                z_list, self.cls_token, max_nodes_in_batch, x.device
            )
            
            # Transformer 처리
            encoded = self.transformer_d(z_with_cls_batch, mask)
            
            # 출력 처리
            cls_output = encoded[:, 0, :]  # [batch_size, hidden_dim]
            node_outputs = [encoded[i, 1:z_list[i].size(0)+1, :] for i in range(num_graphs)]
            u = torch.cat(node_outputs, dim=0)
            
            # 통계량 예측 및 특징 재구성
            stats_pred = self.stats_predictor(cls_output)
            u_prime = self.u_mlp(u)
            x_recon = self.feature_decoder(u_prime)
            
            # 엣지 재구성
            adj_recon_list = []
            start_idx = 0
            for i in range(num_graphs):
                mask = (batch == i)
                num_nodes = mask.sum().item()
                end_idx = start_idx + num_nodes
                
                # 현재 그래프의 노드 임베딩 추출
                current_embeddings = u_prime[start_idx:end_idx]
                # 엣지 재구성
                adj_recon = self.edge_recon(current_embeddings)
                adj_recon_list.append(adj_recon)
                
                start_idx = end_idx
            
            return cls_output, x_recon, stats_pred, adj_recon_list
    

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

# %%
'''RUN'''
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
    bert_save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/BERT_model/Class/all_pretrained_bert_{dataset_name}_fold{trial}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_try21.pth'

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
    
    # 1단계: BERT 임베딩 학습
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        # BERT 인코더의 가중치만 로드
        model.encoder.load_state_dict(torch.load(bert_save_path))
    else:
        print("Training BERT from scratch...")
        # 1단계: BERT 임베딩 학습
        print("Stage 1: Training BERT embeddings...")

        pretrain_params = list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=learning_rate)
        
        print("Stage 1-1: Training BERT with mask token reconstruction...")
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample = train_bert_embedding(
                model, train_loader, bert_optimizer, device
            )
            
            if epoch % log_interval == 0:
                print(f'BERT Mask Training Epoch {epoch}: Loss = {train_loss:.4f}')
        

        print("\nStage 1-2: Training BERT with edge reconstruction...")
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample = train_bert_edge_reconstruction(
                model, train_loader, bert_optimizer, device
            )
            
            if epoch % log_interval == 0:
                print(f'BERT Edge Training Epoch {epoch}: Loss = {train_loss:.4f}')
                
        # 학습된 BERT 저장
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
        
    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cluster_centers, train_errors = train(
            model, train_loader, recon_optimizer, device, epoch, dataset_name
        )
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
                        
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, all_scores, all_labels, test_errors, visualization_data = evaluate_model(model, test_loader, train_cluster_centers, n_cluster, gamma_cluster, random_seed, train_errors, epoch, dataset_name, device)
                                                                                                                                                    
            plot_error_distribution(train_errors, test_errors, epoch, trial, dataset_name, current_time)

            # 결과 저장 디렉토리 생성
            for path_type in ['json', 'plot']:
                save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/{path_type}/{dataset_name}_time_{current_time}/'
                os.makedirs(save_path, exist_ok=True)
            
            # 시각화 데이터 저장
            json_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/error_distribution_epoch_{epoch}_fold_{trial}.json'
            with open(json_path, 'w') as f:
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
    results_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/cross_val_results/all_pretrained_bert_{dataset_name}_time_{current_time}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_try20.json'
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
