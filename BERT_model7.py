#%%
'''PRESENT'''
print('이번 BERT 모델 7은 AIDS, BZR, COX2, DHFR에 대한 실험 파일입니다. 마스크 토큰 재구성을 통한 프리-트레인이 이루어지고 엣지 재구성은 학습된 노드 임베딩을 바탕으로 수행됩니다. 이 때 마스크는 더 이상 사용되지 않습니다. 이후 기존과 같이 노드 특성을 재구성하는 모델을 통해 이상을 탐지합니다.')

#%%
'''IMPORTS'''
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
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx, get_laplacian
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from matplotlib.colors import Normalize

from scipy.linalg import eigh
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from functools import partial
from multiprocessing import Pool

from module.loss import loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, batch_nodes_subgraphs, visualize

import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


#%%
'''TRAIN BERT'''
def train_bert_embedding(model, train_loader, bert_optimizer, bert_scheduler, device, mask_ratio=0.15):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for batch_idx, data in enumerate(train_loader):
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        # 마스크 생성 - 각 그래프별로 독립적으로 마스킹
        mask_indices = []
        start_idx = 0
        for i in range(num_graphs):
            graph_mask = (batch == i)
            num_nodes = graph_mask.sum().item()
            # 각 그래프마다 최소 1개 노드는 마스킹
            num_masks = max(1, int(num_nodes * mask_ratio))
            graph_indices = torch.randperm(num_nodes)[:num_masks]
            graph_mask_indices = torch.zeros(num_nodes, device=device, dtype=torch.bool)
            graph_mask_indices[graph_indices] = True
            mask_indices.append(graph_mask_indices)
            start_idx += num_nodes
        
        mask_indices = torch.cat(mask_indices)

        # BERT 인코딩 및 마스크 토큰 예측만 수행
        _, _, masked_outputs = model(
            x, edge_index, batch, num_graphs, mask_indices, training=True, edge_training=False
        )
        
        masked_features = x[mask_indices]
        mask_loss = torch.norm(masked_outputs - masked_features, p='fro')**2 / mask_indices.sum()
        # mask_loss = F.mse_loss(masked_outputs, masked_features)
        
        # Gradient clipping 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss = mask_loss
        print(f'mask_node_feature:{mask_loss}')
        
        loss.backward()
        bert_optimizer.step()
        # 배치마다 스케줄러 step
        bert_scheduler.step()
        
        total_loss += loss.item()
        num_sample += num_graphs

        if batch_idx % 10 == 0:  # 10배치마다 출력
            print(f'Batch {batch_idx}: Loss = {loss.item():.4f}, '
                  f'Masked nodes: {mask_indices.sum().item()}, '
                  f'LR: {bert_optimizer.param_groups[0]["lr"]:.6f}')
            
    return total_loss / len(train_loader), num_sample


#%%
# def train_bert_embedding_(model, train_loader, bert_optimizer, device):
#     model.train()
#     total_loss = 0
#     num_sample = 0
    
#     for data in train_loader:
#         bert_optimizer.zero_grad()
#         data = data.to(device)
#         x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
#         adj = adj_original(edge_index, batch, max_nodes)
        
#         # BERT 인코딩 및 마스크 토큰 예측만 수행
#         _, _, adj_recon_list = model(
#             x, edge_index, batch, num_graphs, training=True, edge_training=True
#         )

#         loss = 0
#         start_node = 0
#         for i in range(num_graphs):
#             num_nodes = (batch == i).sum().item()
#             end_node = start_node + num_nodes
            
#             adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / num_nodes
#             adj_loss = adj_loss / 20
#             loss += adj_loss
            
#             start_node = end_node
            
#         print(f'edge_loss:{loss}')
#         # print(f'mask_label:{mask_loss_}')
        
#         loss.backward()
#         bert_optimizer.step()
#         total_loss += loss.item()
#         num_sample += num_graphs
    
#     return total_loss / len(train_loader), num_sample


# '''EVALUATION'''
# def evaluate_model(model, test_loader, max_nodes, cluster_centers, device):
#     model.eval()
#     total_loss_ = 0
#     total_loss_anomaly_ = 0
#     total_loss_mean = 0
#     total_loss_anomaly_mean = 0

#     all_labels = []
#     all_scores = []

#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

#             e_cls_output, x_recon = model(x, edge_index, batch, num_graphs)

#             recon_errors = []
#             start_node = 0
#             for i in range(num_graphs):
#                 num_nodes = (batch == i).sum().item()
#                 end_node = start_node + num_nodes
                
#                 node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                
#                 # cls_vec = e_cls_output[i].cpu().numpy()  # [hidden_dim]
#                 cls_vec = e_cls_output[i].detach().cpu().numpy()  # [hidden_dim]
#                 distances = cdist([cls_vec], cluster_centers, metric='euclidean')  # [1, n_clusters]
#                 min_distance = distances.min()

#                 recon_error = (node_loss.item() * alpha) + (min_distance.item() * gamma)              
#                 recon_errors.append(recon_error)
                
#                 print(f'test_node_loss: {node_loss.item() * alpha}')
#                 print(f'test_min_distance: {min_distance.item() * gamma}')

#                 if data.y[i].item() == 0:
#                     total_loss_ += recon_error
#                 else:
#                     total_loss_anomaly_ += recon_error

#                 start_node = end_node
            
#             total_loss = total_loss_ / sum(data.y == 0)
#             total_loss_anomaly = total_loss_anomaly_ / sum(data.y == 1)
            
#             total_loss_mean += total_loss
#             total_loss_anomaly_mean += total_loss_anomaly
            
#             all_scores.extend(recon_errors)
#             all_labels.extend(data.y.cpu().numpy())

#     all_labels = np.array(all_labels)
#     all_scores = np.array(all_scores)

#     # Compute metrics
#     fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
#     auroc = auc(fpr, tpr)
#     precision, recall, _ = precision_recall_curve(all_labels, all_scores)
#     auprc = auc(recall, precision)

#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
#     pred_labels = (all_scores > optimal_threshold).astype(int)

#     precision = precision_score(all_labels, pred_labels)
#     recall = recall_score(all_labels, pred_labels)
#     f1 = f1_score(all_labels, pred_labels)

#     return auroc, auprc, precision, recall, f1, total_loss_mean / len(test_loader), total_loss_anomaly_mean / len(test_loader)


#%%
def train(model, train_loader, recon_optimizer, max_nodes, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_cls_loss = 0
    num_sample = 0
    
    for data in train_loader:
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        train_cls_outputs, x_recon = model(x, edge_index, batch, num_graphs)
        
        recon_loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            
            recon_loss += node_loss

            start_node = end_node
                
        # CLS Similarity Loss
        if num_graphs > 1:  # 배치에 2개 이상의 그래프가 있을 때만 계산
            # 1. Pairwise Distance Matrix 계산
            cls_distances = torch.cdist(train_cls_outputs, train_cls_outputs, p=2)  # [num_graphs, num_graphs]
            
            # 2. 대각선 요소 제외 (자기 자신과의 거리)
            mask = ~torch.eye(num_graphs, dtype=bool, device=device)
            cls_distances = cls_distances[mask]
            
            # 3. 평균 거리 계산
            cls_loss = cls_distances.mean()
        else:
            cls_loss = torch.tensor(0.0, device=device)
                
        # Total Loss (alpha는 CLS 손실의 가중치)
        alpha_ = 10.0  # 이 값은 조정 가능
        cls_loss = alpha_ * cls_loss
        
        loss = recon_loss + cls_loss
            
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_cls_loss += cls_loss.item()
        
        print(f'train_recon_loss: {recon_loss.item():.4f}, train_cls_loss: {cls_loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    
    print(f'Average reconstruction loss: {avg_recon_loss:.4f}')
    print(f'Average CLS similarity loss: {avg_cls_loss:.4f}')        
        
    return avg_loss, num_sample, train_cls_outputs.detach().cpu()


def train(model, train_loader, recon_optimizer, max_nodes, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label
        
        train_cls_outputs, x_recon = model(x, edge_index, batch, num_graphs)
        
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            
            loss += node_loss

            start_node = end_node
        
        print(f'train_node_loss: {loss}')
        
        num_sample += num_graphs

        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), num_sample, train_cls_outputs.detach().cpu()


#%%
def evaluate_model(model, test_loader, cluster_centers, n_clusters, gamma_clusters, random_seed, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors = []  # 새로 추가
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            e_cls_output, x_recon = model(x, edge_index, batch, num_graphs)
            
            e_cls_outputs_np = e_cls_output.detach().cpu().numpy()  # [num_graphs, hidden_dim]
            spectral_embedder_test = SpectralEmbedding(
                n_components=n_clusters, 
                affinity='rbf', 
                gamma=gamma_clusters,
                random_state=random_seed
            )

            spectral_embeddings_test = spectral_embedder_test.fit_transform(e_cls_outputs_np)  # [num_graphs, k]
            
            recon_errors = []
            start_node = 0
            
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # Reconstruction error 계산
                node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                node_loss = node_loss.item() * alpha
                transformed_cls_vec = spectral_embeddings_test[i]  # [k]
                transformed_cls_vec = np.reshape(transformed_cls_vec, (1, -1))  # 또는 transformed_cls_vec[None, :]
                distances = cdist(transformed_cls_vec, cluster_centers, metric='euclidean')
                # distances = cdist([transformed_cls_vec], cluster_centers, metric='euclidean')  # [1, n_clusters]
                min_distance = distances.min().item() * gamma
                
                # 변환된 값들 저장
                reconstruction_errors.append({
                    'reconstruction': node_loss,
                    'clustering': min_distance,
                    'is_anomaly': data.y[i].item() == 1
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
            for error in reconstruction_errors if not error['is_anomaly']
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'clustering': error['clustering']}
            for error in reconstruction_errors if error['is_anomaly']
        ]
    }
    
    # 메트릭 계산
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
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

    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, visualization_data


#%%
def analyze_cls_embeddings(model, test_loader, device, n_components=2):
    """
    CLS token embeddings를 추출하고 PCA를 적용하여 더 자세한 분석 수행
    
    Args:
        model: 학습된 모델
        test_loader: 테스트 데이터 로더
        device: 연산 장치
        n_components: PCA 차원 수
    """
    model.eval()
    cls_embeddings = []
    labels = []
    
    # CLS embeddings 추출
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            
            cls_output, _ = model(x, edge_index, batch, num_graphs)
            cls_embeddings.append(cls_output.cpu())
            labels.extend(data.y.cpu().numpy())
    
    # 텐서를 numpy 배열로 변환
    cls_embeddings = torch.cat(cls_embeddings, dim=0).numpy()
    labels = np.array(labels)
    
    # PCA 적용
    pca = PCA()
    cls_embeddings_pca = pca.fit_transform(cls_embeddings)
    
    # 분산 설명률 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    eigenvalues = pca.explained_variance_
    
    # 시각화
    plt.figure(figsize=(20, 15))
    
    # 1. PCA 산점도 (첫 2개 주성분)
    plt.subplot(221)
    scatter = plt.scatter(cls_embeddings_pca[:, 0], cls_embeddings_pca[:, 1], 
                         c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Label (0: Normal, 1: Anomaly)')
    plt.xlabel(f'PC1 (variance ratio: {explained_variance_ratio[0]:.3f})')
    plt.ylabel(f'PC2 (variance ratio: {explained_variance_ratio[1]:.3f})')
    plt.title('PCA of CLS Token Embeddings')
    
    # 2. 누적 분산 설명률
    plt.subplot(222)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    
    # 3. 스크린 플롯
    plt.subplot(223)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.grid(True)
    
    # 4. 로그 스케일 스크린 플롯
    plt.subplot(224)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.yscale('log')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Scree Plot (Log Scale)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 분석 결과 저장
    save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/pca_analysis/cls_pca_analysis_{dataset_name}_{current_time}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # 주요 지표 계산
    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    # 분석 결과 출력
    print("\nPCA Analysis Results:")
    print(f"Number of features in original space: {cls_embeddings.shape[1]}")
    print(f"Number of components needed for 90% variance: {n_components_90}")
    print(f"Number of components needed for 95% variance: {n_components_95}")
    print("\nTop 5 Principal Components:")
    for i in range(5):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} "
              f"(cumulative: {cumulative_variance_ratio[i]:.4f})")
    
    # Class separation analysis
    normal_indices = labels == 0
    anomaly_indices = labels == 1
    
    pc1_normal = cls_embeddings_pca[normal_indices, 0]
    pc1_anomaly = cls_embeddings_pca[anomaly_indices, 0]
    pc2_normal = cls_embeddings_pca[normal_indices, 1]
    pc2_anomaly = cls_embeddings_pca[anomaly_indices, 1]
    
    print("\nClass Separation Analysis:")
    print("PC1 - Normal vs Anomaly:")
    print(f"Normal mean: {np.mean(pc1_normal):.4f}, std: {np.std(pc1_normal):.4f}")
    print(f"Anomaly mean: {np.mean(pc1_anomaly):.4f}, std: {np.std(pc1_anomaly):.4f}")
    print("\nPC2 - Normal vs Anomaly:")
    print(f"Normal mean: {np.mean(pc2_normal):.4f}, std: {np.std(pc2_normal):.4f}")
    print(f"Anomaly mean: {np.mean(pc2_anomaly):.4f}, std: {np.std(pc2_anomaly):.4f}")
    
    return cls_embeddings_pca, explained_variance_ratio, eigenvalues


#%%
def perform_clustering(train_cls_outputs, random_seed, n_clusters, gamma_clusters):
    """
    스펙트럴 클러스터링을 수행하는 함수
    
    Args:
        train_cls_outputs: 학습 데이터의 CLS 토큰 출력값
        random_seed: 랜덤 시드
        n_clusters: 클러스터 수
    
    Returns:
        spectral: 학습된 스펙트럴 클러스터링 모델
        cluster_centers: 각 클러스터의 중심점
    """
    # CPU로 이동 및 Numpy 배열로 변환
    cls_outputs_np = train_cls_outputs.detach().cpu().numpy()
    
    # 고유벡터 기반의 k차원 임베딩 생성
    spectral_embedder = SpectralEmbedding(
        n_components=n_clusters, 
        affinity='rbf', 
        gamma=gamma_clusters,
        random_state=random_seed
    )
    spectral_embeddings = spectral_embedder.fit_transform(cls_outputs_np)  # [num_samples, k]
    
    # k-평균 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(spectral_embeddings)
    
    cluster_centers = kmeans.cluster_centers_
    
    return spectral_embedder, cluster_centers


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--n-head-BERT", type=int, default=2)
parser.add_argument("--n-layer-BERT", type=int, default=2)
parser.add_argument("--n-head", type=int, default=8)
parser.add_argument("--n-layer", type=int, default=8)
parser.add_argument("--BERT-epochs", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=21)
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
parser.add_argument("--learning-rate", type=float, default=0.00001)

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.001)
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
    

# def perform_clustering(train_cls_outputs, random_seed, n_clusters):
#     cls_outputs_np = train_cls_outputs.detach().cpu().numpy()
#     kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto").fit(cls_outputs_np)
#     return kmeans, kmeans.cluster_centers_
            

#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT, nhead, num_layers_BERT, num_layers, dropout_rate=0.1):
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

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=False, edge_training=False):
        # BERT 인코딩
        if training:
            if edge_training:
                z, adj_recon_list = self.encoder(x, edge_index, batch, num_graphs, mask_indices, training=True, edge_training=True)
            else:
                z, masked_outputs = self.encoder(x, edge_index, batch, num_graphs, mask_indices, training=True, edge_training=False)
        else:
            z = self.encoder(x, edge_index, batch, num_graphs, training=False, edge_training=False)
        
        # 배치 처리
        z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(z, edge_index, batch, num_graphs)
        z_with_cls_batch, mask = BatchUtils.add_cls_token(
            z_list, self.cls_token, max_nodes_in_batch, x.device
        )
        
        # Transformer 처리 - 수정된 부분
        encoded = self.transformer_d(z_with_cls_batch, mask)  # edge_index_list 제거
        
        # 출력 처리
        cls_output = encoded[:, 0, :]
        node_outputs = [encoded[i, 1:z_list[i].size(0)+1, :] for i in range(num_graphs)]
        u = torch.cat(node_outputs, dim=0)
        
        # 디코딩
        u_prime = self.u_mlp(u)
        x_recon = self.feature_decoder(u_prime)
        
        if training:
            if edge_training:
                return cls_output, x_recon, adj_recon_list
            else:
                return cls_output, x_recon, masked_outputs
        return cls_output, x_recon
    

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
    bert_save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/BERT_model/Class/pretrained_bert_{dataset_name}_fold{trial}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_gcn{hidden_dims[-1]}_edge_train_try7.pth'
    
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
        model.encoder.load_state_dict(torch.load(bert_save_path, weights_only=True))
    else:
        print("Training BERT from scratch...")
        # 1단계: BERT 임베딩 학습
        print("Stage 1: Training BERT embeddings...")

        pretrain_params = list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=learning_rate)
        bert_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(bert_optimizer, mode='min', factor=0.5, patience=1000, verbose=True)

        # OneCycleLR 스케줄러 사용 (warm-up과 유사한 효과)
        bert_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            bert_optimizer,
            max_lr=learning_rate,
            epochs=BERT_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% 구간을 warm-up에 사용
            anneal_strategy='cos'
        )
        # bert_scheduler = ReduceLROnPlateau(bert_optimizer, mode='min', factor=factor, patience=patience)
    
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample = train_bert_embedding(
                model, train_loader, bert_optimizer, bert_scheduler, device
            )
            bert_scheduler.step(train_loss)
            
            if epoch % log_interval == 0:
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}')
        
                current_lr = bert_optimizer.param_groups[0]['lr']
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}, LR = {current_lr:.6f}')
        
        
        # for epoch in range(1, BERT_epochs+1):
        #     train_adj_loss, num_sample_ = train_bert_embedding_(
        #         model, train_loader, bert_optimizer, device
        #     )
        #     # bert_scheduler.step(train_loss)
            
        #     if epoch % log_interval == 0:
        #         print(f'BERT Edge Training Epoch {epoch}: Loss = {train_adj_loss:.4f}')
                      
        # 학습된 BERT 저장
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
        
    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, recon_optimizer, max_nodes, device)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            # kmeans, cluster_centers = perform_clustering(train_cls_outputs, random_seed, n_clusters=n_cluster)
            # cluster_centers = cluster_centers.reshape(-1, hidden_dims[-1])

            spectral_embedder, cluster_centers = perform_clustering(train_cls_outputs, random_seed, n_clusters=n_cluster, gamma_clusters=gamma_cluster)
            # cluster_centers = cluster_centers.reshape(-1, hidden_dims[-1])
            
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, visualization_data = evaluate_model(model, test_loader, cluster_centers, n_cluster, gamma_cluster, random_seed, device)
            
            save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}/error_distribution_epoch_{epoch}_fold_{trial}.json'
            # save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}2/error_distribution_epoch_{epoch}_fold_{trial}.json'
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
            
            # run() 함수 내에서 평가 시점에 다음 코드 추가
            cls_embeddings_pca, explained_variance_ratio, eigenvalues = analyze_cls_embeddings(model, test_loader, device)
            
    return auroc, epoch_results, cls_embeddings_pca, explained_variance_ratio, eigenvalues


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
        ad_auc, epoch_results, cls_embeddings_pca, explained_variance_ratio, eigenvalues = run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=epoch_results)
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


#%%
# base_dir = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}/'
base_dir = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_rand_18034064/'

trial = 0
for epoch in range(1, epochs + 1):    
    if epoch % log_interval == 0:
        with open(base_dir + f'error_distribution_epoch_{epoch}_fold_{trial}.json', 'r') as f:
            data = json.load(f)

        # 데이터 추출
        normal_recon = [point['reconstruction'] for point in data['normal']]
        normal_cluster = [point['clustering'] for point in data['normal']]
        anomaly_recon = [point['reconstruction'] for point in data['anomaly']]
        anomaly_cluster = [point['clustering'] for point in data['anomaly']]

        # 산점도 그리기
        plt.figure(figsize=(10, 6))
        plt.scatter(normal_recon, normal_cluster, c='blue', label='Normal', alpha=0.6)
        plt.scatter(anomaly_recon, anomaly_cluster, c='red', label='Anomaly', alpha=0.6)

        # x축과 y축 범위 설정
        plt.xlim(4, 10)  # x축 범위를 0~10으로 설정
        plt.ylim(2, 5)   # y축 범위를 0~5로 설정
        
        plt.xlabel('Reconstruction Error (node_loss * alpha)')
        plt.ylabel('Clustering Distance (min_distance * gamma)')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True)

        # 저장하거나 보여주기
        plt.savefig(f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_rand_18034064/error_distribution_plot_epoch_{epoch}_fold_{trial}.png')  # 파일로 저장
        # plt.savefig(f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}2/error_distribution_plot_epoch_{epoch}_fold_{trial}.png')  # 파일로 저장
        plt.show()  # 직접 보기
        

# %%
# current_time_ = time.localtime()
# current_time = time.strftime("%Y_%m_%d_%H_%M", current_time_)
# print(f'random number saving: {current_time}')

# save_path = f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/'
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph

# 1. 데이터 생성
n_samples = 200
X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# 2. 유사도 행렬 계산 (k-nearest neighbors 방식)
k = 10
connectivity = kneighbors_graph(X, n_neighbors=k, include_self=False)
affinity_matrix = 0.5 * (connectivity + connectivity.T)

# 3. Spectral Clustering 수행
n_clusters = 2
clustering = SpectralClustering(
    n_clusters=n_clusters,
    affinity='precomputed',
    random_state=42
)
labels = clustering.fit_predict(affinity_matrix.toarray())

# 4. 시각화
plt.figure(figsize=(12, 4))

# 4.1 원본 데이터
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
plt.title('Original Data')

# 4.2 클러스터링 결과
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Clustering Results')

# 결과 표시
plt.tight_layout()
plt.show()

# 5. 클러스터별 통계
for i in range(n_clusters):
    cluster_points = X[labels == i]
    print(f"\nCluster {i} Statistics:")
    print(f"Number of points: {len(cluster_points)}")
    print(f"Centroid: {cluster_points.mean(axis=0)}")
    print(f"Standard deviation: {cluster_points.std(axis=0)}")
    
    
#%%
def visualize_molecule_graph(x, edge_index, node_label, batch=None, node_idx=1, title="Molecular Graph Structure"):
    """
    노드 라벨에 따라 다른 색상으로 분자 구조 스타일의 그래프를 시각화하는 함수
    
    Parameters:
    -----------
    x : torch.Tensor
        노드 특성을 포함하는 텐서 [num_nodes, num_features]
    edge_index : torch.Tensor
        엣지 정보를 포함하는 텐서 [2, num_edges]
    node_label : torch.Tensor
        노드 라벨 정보를 포함하는 텐서 [num_nodes]
    batch : torch.Tensor, optional
        배치 정보를 포함하는 텐서
    node_idx : int
        시각화할 그래프의 인덱스 (배치가 있는 경우)
    title : str
        그래프 제목
    """
    # CUDA 텐서를 CPU로 이동
    x = x.cpu()
    edge_index = edge_index.cpu()
    node_label = node_label.cpu()
    if batch is not None:
        batch = batch.cpu()
    
    # 배치가 있는 경우 특정 그래프만 선택
    if batch is not None:
        mask = batch == node_idx
        sub_x = x[mask]
        sub_label = node_label[mask]
        
        # 노드 매핑 생성 (전체 인덱스 → 서브그래프 인덱스)
        node_idx_list = torch.where(mask)[0]
        node_mapper = {int(idx): i for i, idx in enumerate(node_idx_list)}
        
        # edge_index 필터링
        row, col = edge_index
        mask_edge = (batch[row] == node_idx) & (batch[col] == node_idx)
        
        # edge_index의 노드 번호를 새로운 인덱스로 매핑
        filtered_row = row[mask_edge]
        filtered_col = col[mask_edge]
        
        mapped_row = torch.tensor([node_mapper[int(idx)] for idx in filtered_row])
        mapped_col = torch.tensor([node_mapper[int(idx)] for idx in filtered_col])
        
        sub_edge_index = torch.stack([mapped_row, mapped_col])
    else:
        sub_x = x
        sub_label = node_label
        sub_edge_index = edge_index

    # NetworkX 그래프 생성
    G = nx.Graph()
    
    # 노드 추가 (라벨 정보 포함)
    for i in range(len(sub_x)):
        G.add_node(i, label=int(sub_label[i]))
    
    # 엣지 추가
    edges = sub_edge_index.t().detach().numpy()
    G.add_edges_from(edges)

    # 그래프 레이아웃 설정
    pos = nx.kamada_kawai_layout(G)
    
    # Figure 설정
    plt.figure(figsize=(10, 10), facecolor='white')
    
    # 실제 라벨 값 가져오기
    unique_labels = sorted(list(set([G.nodes[node]['label'] for node in G.nodes()])))
    
    # 라벨별 색상 매핑 설정
    color_map = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
    label_color_dict = {label: color_map[i] for i, label in enumerate(unique_labels)}
    
    # 각 노드의 라벨에 따른 색상 리스트 생성
    node_colors = [label_color_dict[G.nodes[node]['label']] for node in G.nodes()]
    
    # 노드 그리기 (원자 스타일)
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=1500,
                          edgecolors='black',
                          linewidths=2)
    
    # 엣지 그리기 (결합 스타일)
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=2,
                          alpha=0.8)
    
    # 실제 라벨 값으로 범례 추가
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=label_color_dict[label], 
                                 markersize=15,
                                 label=str(label))
                      for label in unique_labels]
    plt.legend(handles=legend_elements, title="Node Labels", 
              loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(f"{title} (Graph {node_idx})", fontsize=16, pad=20)
    plt.axis('off')
    
    # 여백 조정 - 범례가 잘리지 않도록 오른쪽 여백 확보
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    return plt.gcf()


# %%
fig = visualize_molecule_graph(x, edge_index, node_label, batch, node_idx=3)
plt.show()

# %%
def visualize_2d(x, edge_index, batch=None, node_label=None, node_idx=1, color='skyblue', edge_color='blue', title="Graph Visualization"):
    """
    PyTorch Geometric 데이터를 시각화하는 함수
    
    Parameters:
    -----------
    x : torch.Tensor
        노드 특성을 포함하는 텐서 [num_nodes, num_features]
    edge_index : torch.Tensor
        엣지 정보를 포함하는 텐서 [2, num_edges]
    batch : torch.Tensor, optional
        배치 정보를 포함하는 텐서
    node_label : torch.Tensor, optional
        노드 라벨 정보를 포함하는 텐서 [num_nodes]
    node_idx : int
        시각화할 그래프의 인덱스 (배치가 있는 경우)
    color : str
        노드 색상
    edge_color : str
        엣지 색상
    title : str
        그래프 제목
    """
    # CUDA 텐서를 CPU로 이동
    x = x.cpu()
    edge_index = edge_index.cpu()
    if batch is not None:
        batch = batch.cpu()
    if node_label is not None:
        node_label = node_label.cpu()
    
    # 배치가 있는 경우 특정 그래프만 선택
    if batch is not None:
        mask = batch == node_idx
        sub_x = x[mask]
        if node_label is not None:
            sub_label = node_label[mask]
        
        # 노드 매핑 생성
        node_idx_list = torch.where(mask)[0]
        node_mapper = {int(idx): i for i, idx in enumerate(node_idx_list)}
        
        # edge_index 필터링
        row, col = edge_index
        mask_edge = (batch[row] == node_idx) & (batch[col] == node_idx)
        filtered_row = row[mask_edge]
        filtered_col = col[mask_edge]
        
        # 새로운 인덱스로 매핑
        mapped_row = torch.tensor([node_mapper[int(idx)] for idx in filtered_row])
        mapped_col = torch.tensor([node_mapper[int(idx)] for idx in filtered_col])
        sub_edge_index = torch.stack([mapped_row, mapped_col])
    else:
        sub_x = x
        sub_edge_index = edge_index
        sub_label = node_label if node_label is not None else None
    
    # PyTorch Geometric Data 객체 생성
    data = Data(x=sub_x, edge_index=sub_edge_index)
    
    # networkx 그래프로 변환
    G = to_networkx(data, to_undirected=True)
    
    # Figure 설정
    plt.figure(figsize=(10, 10), facecolor='white')
    
    # 노드 색상 설정 (node_label이 있는 경우 라벨에 따라 다른 색상 사용)
    if node_label is not None:
        unique_labels = sorted(list(set(sub_label.tolist())))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
        node_colors = [color_dict[int(sub_label[i])] for i in range(len(sub_x))]
    else:
        node_colors = color
    
    # 그래프 그리기
    nx.draw_networkx(G, 
                     pos=nx.spring_layout(G, seed=42),
                     with_labels=True,
                     node_color=node_colors,
                     edge_color=edge_color,
                     node_size=500,
                     font_size=10,
                     font_weight='bold',
                     width=2,
                     alpha=0.9)
    
    plt.title(f"{title} (Graph {node_idx})", fontsize=16, pad=20)
    plt.axis('off')
    
    # 노드 라벨이 있는 경우 범례 추가 (정수형으로 표시)
    if node_label is not None:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color_dict[label],
                                    markersize=10,
                                    label=int(label))  # 정수형으로 변환
                          for label in unique_labels]
        plt.legend(handles=legend_elements,
                  title="Node Labels",
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.85)
    
    return plt.gcf()


# %%
fig = visualize(x, edge_index, batch, node_idx=1)
plt.show()

# 노드 라벨을 사용한 시각화
fig = visualize(x, edge_index, batch, node_label, node_idx=1)
plt.show()

# 커스텀 색상을 사용한 시각화
fig = visualize(x, edge_index, batch, color='lightgreen', edge_color='darkgreen', node_idx=1)
plt.show()

#%%

def visualize_3d(x, edge_index, batch=None, node_label=None, node_idx=1, color='skyblue', 
                edge_color='blue', title="3D Graph Visualization"):
    """
    PyTorch Geometric 데이터를 3D로 시각화하는 함수
    
    Parameters:
    -----------
    x : torch.Tensor
        노드 특성을 포함하는 텐서 [num_nodes, num_features]
    edge_index : torch.Tensor
        엣지 정보를 포함하는 텐서 [2, num_edges]
    batch : torch.Tensor, optional
        배치 정보를 포함하는 텐서
    node_label : torch.Tensor, optional
        노드 라벨 정보를 포함하는 텐서 [num_nodes]
    node_idx : int
        시각화할 그래프의 인덱스 (배치가 있는 경우)
    """
    # CUDA 텐서를 CPU로 이동
    x = x.cpu()
    edge_index = edge_index.cpu()
    if batch is not None:
        batch = batch.cpu()
    if node_label is not None:
        node_label = node_label.cpu()
    
    # 배치가 있는 경우 특정 그래프만 선택
    if batch is not None:
        mask = batch == node_idx
        sub_x = x[mask]
        if node_label is not None:
            sub_label = node_label[mask]
        
        node_idx_list = torch.where(mask)[0]
        node_mapper = {int(idx): i for i, idx in enumerate(node_idx_list)}
        
        row, col = edge_index
        mask_edge = (batch[row] == node_idx) & (batch[col] == node_idx)
        filtered_row = row[mask_edge]
        filtered_col = col[mask_edge]
        
        mapped_row = torch.tensor([node_mapper[int(idx)] for idx in filtered_row])
        mapped_col = torch.tensor([node_mapper[int(idx)] for idx in filtered_col])
        sub_edge_index = torch.stack([mapped_row, mapped_col])
    else:
        sub_x = x
        sub_edge_index = edge_index
        sub_label = node_label if node_label is not None else None
    
    # PyTorch Geometric Data 객체 생성
    data = Data(x=sub_x, edge_index=sub_edge_index)
    
    # networkx 그래프로 변환
    G = to_networkx(data, to_undirected=True)
    
    # 3D 레이아웃 생성
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Figure 설정
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # 노드 색상 설정
    if node_label is not None:
        unique_labels = sorted(list(set(sub_label.tolist())))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
        node_colors = [color_dict[int(sub_label[i])] for i in range(len(sub_x))]
    else:
        node_colors = [color] * G.number_of_nodes()
    
    # 노드 위치 추출
    node_xyz = np.array([pos[v] for v in sorted(G.nodes())])
    
    # 엣지 위치 추출
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # 노드 그리기
    ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], 
              c=node_colors, s=200, alpha=0.9)

    # 엣지 그리기
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color=edge_color, alpha=0.5)

    # 노드 번호 표시
    for i, (x, y, z) in enumerate(node_xyz):
        ax.text(x, y, z, str(i), fontsize=8, fontweight='bold')
    
    # 범례 추가 (노드 라벨이 있는 경우)
    if node_label is not None:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color_dict[label],
                                    markersize=10,
                                    label=int(label))
                          for label in unique_labels]
        ax.legend(handles=legend_elements,
                 title="Node Labels",
                 loc='center left',
                 bbox_to_anchor=(1.1, 0.5))
    
    # 축 레이블과 제목 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"{title} (Graph {node_idx})", fontsize=16, pad=20)
    
    # 보기 각도 설정
    ax.view_init(elev=20, azim=45)
    
    # 여백 조정
    plt.tight_layout()
    
    return fig

# 시각화 방법을 전환할 수 있는 함수
def visualize(x, edge_index, batch=None, node_label=None, node_idx=1, 
             color='skyblue', edge_color='blue', title="Graph Visualization", 
             mode='2d'):
    """
    2D 또는 3D 시각화를 선택할 수 있는 함수
    
    Parameters:
    -----------
    mode : str
        '2d' 또는 '3d' 선택
    """
    if mode == '3d':
        return visualize_3d(x, edge_index, batch, node_label, node_idx, 
                          color, edge_color, title)
    else:
        # 기존의 2D 시각화 코드 사용
        return visualize_2d(x, edge_index, batch, node_label, node_idx, 
                          color, edge_color, title)
# %%
fig = visualize(x, edge_index, batch, node_label, node_idx=10, mode='3d')
plt.show()

# %%
import plotly.graph_objects as go

def visualize_3d_interactive(x, edge_index, batch=None, node_label=None, node_idx=1, title="Interactive 3D Graph"):
    """
    PyTorch Geometric 데이터를 인터랙티브 3D로 시각화하는 함수
    마우스로 자유롭게 회전, 확대/축소, 이동이 가능합니다.
    
    Parameters:
    -----------
    x : torch.Tensor
        노드 특성을 포함하는 텐서 [num_nodes, num_features]
    edge_index : torch.Tensor
        엣지 정보를 포함하는 텐서 [2, num_edges]
    batch : torch.Tensor, optional
        배치 정보를 포함하는 텐서
    node_label : torch.Tensor, optional
        노드 라벨 정보를 포함하는 텐서 [num_nodes]
    node_idx : int
        시각화할 그래프의 인덱스 (배치가 있는 경우)
    title : str
        그래프 제목
    """
    # CUDA 텐서를 CPU로 이동
    x = x.cpu()
    edge_index = edge_index.cpu()
    if batch is not None:
        batch = batch.cpu()
    if node_label is not None:
        node_label = node_label.cpu()
    
    # 배치 처리
    if batch is not None:
        mask = batch == node_idx
        sub_x = x[mask]
        if node_label is not None:
            sub_label = node_label[mask]
        
        node_idx_list = torch.where(mask)[0]
        node_mapper = {int(idx): i for i, idx in enumerate(node_idx_list)}
        
        row, col = edge_index
        mask_edge = (batch[row] == node_idx) & (batch[col] == node_idx)
        filtered_row = row[mask_edge]
        filtered_col = col[mask_edge]
        
        mapped_row = torch.tensor([node_mapper[int(idx)] for idx in filtered_row])
        mapped_col = torch.tensor([node_mapper[int(idx)] for idx in filtered_col])
        sub_edge_index = torch.stack([mapped_row, mapped_col])
    else:
        sub_x = x
        sub_edge_index = edge_index
        sub_label = node_label if node_label is not None else None

    # NetworkX 그래프로 변환
    G = to_networkx(Data(x=sub_x, edge_index=sub_edge_index), to_undirected=True)
    
    # 3D 레이아웃 생성
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # 노드 위치 추출
    Xn = [pos[k][0] for k in G.nodes()]
    Yn = [pos[k][1] for k in G.nodes()]
    Zn = [pos[k][2] for k in G.nodes()]

    # 엣지 위치 추출 (각 엣지를 선으로 표현하기 위해 None을 추가)
    Xe, Ye, Ze = [], [], []
    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
        Ze.extend([pos[e[0]][2], pos[e[1]][2], None])
    
    # 노드 색상 설정
    if node_label is not None:
        unique_labels = sorted(list(set(sub_label.tolist())))
        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                 for r, g, b, _ in plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))]
        node_colors = [colors[unique_labels.index(int(sub_label[i]))] for i in range(len(sub_x))]
    else:
        node_colors = ['skyblue'] * G.number_of_nodes()

    # 트레이스 생성
    edge_trace = go.Scatter3d(
        x=Xe, y=Ye, z=Ze,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter3d(
        x=Xn, y=Yn, z=Zn,
        mode='markers+text',
        marker=dict(
            size=15,
            color=node_colors,
            line=dict(color='black', width=1)
        ),
        text=[str(i) for i in range(len(Xn))],  # 노드 번호
        hoverinfo='text'
    )
    
    # 레이아웃 설정
    layout = go.Layout(
        title=f'{title} (Graph {node_idx})',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False)
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='closest'
    )
    
    # Figure 생성
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    
    # 노드 라벨이 있는 경우 범례 추가
    if node_label is not None:
        # 범례를 위한 더미 트레이스 추가
        for label, color in zip(unique_labels, colors):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                showlegend=True,
                name=f'Label {int(label)}'
            ))
    
    return fig

# %%
fig = visualize_3d_interactive(x, edge_index, batch, node_label)
fig.show()  # 브라우저에서 열림

# %%
