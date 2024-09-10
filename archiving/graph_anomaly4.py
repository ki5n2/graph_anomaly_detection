#%%
'''IMPORTS'''
import os
import re
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

from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from module.loss import info_nce_loss, Triplet_loss, loss_cal
from util import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU
        
        
#%%
'''TARIN'''
def train(model, noise_model_encoder, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z, aug_z_g = model(data)
        print(len(adj))
        print(len(adj_recon_list))
        print(adj[0].shape)
        print(adj_recon_list[0].shape)
        print(adj[0])
        print(adj_recon_list[0])
        
        z, z_g = model.encoder(h0, adj)
        z_hat, z_hat_g = gen_ran_output(h0, adj, model.encoder, noise_model_encoder)

        loss = 0
        start_node = 0
        
        loss1 = 0
        loss2 = 0
        
        for i in range(data.num_graphs): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            # graph_num_nodes = end_node - start_node        
            
            adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
            l1_loss = adj_loss / 400
            
            z_dist = torch.pdist(z[start_node:end_node])
            z_tilde_dist = torch.pdist(z_tilde[start_node:end_node])
            w_distance = torch.tensor(wasserstein_distance(z_dist.detach().cpu().numpy(), z_tilde_dist.detach().cpu().numpy()), device='cpu') / 2
            
            loss += l1_loss + w_distance

            loss1 += l1_loss
            loss2 += w_distance            
            
            start_node = end_node
            
        print(f'Train adj_loss : {loss1}')
        print(f'Train w_distance loss :{loss2}')
        
        node_loss = torch.norm(x_recon - data.x, p='fro')**2 / x_recon.size(0)
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) * 3
        l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
        
        print(f'Train node loss: {node_loss}')
        print(f'Train triplet_loss :{triplet_loss}')
        print(f'Train l2_loss :{l2_loss}')
        
        loss += node_loss + triplet_loss + l2_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # recon_z_node_loss = torch.norm(z - z_tilde, p='fro')**2
        # graph_z_node_loss = recon_z_node_loss / (z.size(1) * 4)
        # print(f'Train graph z_node loss:{graph_z_node_loss}')
            
    return total_loss / len(train_loader)


#%%
'''EVALUATION'''
def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_loss_anomaly = 0
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  
            adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z, aug_z_g = model(data)
            
            recon_errors = []
            start_node = 0
            
            for i in range(data.num_graphs): 
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() 
                end_node = start_node + num_nodes
                
                node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2 / num_nodes
                node_recon_error = node_loss * 2
                
                adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
                edge_recon_error = adj_loss / 50
                    
                # edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                # edge_index = edges.t()
                
                # recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                # graph_z_node_loss = recon_z_node_loss/graph_num_nodes

                # recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                # graph_recon_loss = (graph_z_node_loss) + (recon_z_graph_loss)
                # print(f'graph_recon_loss: {graph_recon_loss}')
                
                z_dist = torch.pdist(z[start_node:end_node])
                z_tilde_dist = torch.pdist(z_tilde[start_node:end_node])
                w_distance = torch.tensor(wasserstein_distance(z_dist.detach().cpu().numpy(), z_tilde_dist.detach().cpu().numpy()), device='cpu') 
                
                print(f'node_recon_error: {node_recon_error}')
                print(f'edge_recon_error: {edge_recon_error}')
                print(f'w_distance: {w_distance}')
                
                recon_error += node_recon_error + edge_recon_error + w_distance
                recon_errors.append(recon_error.item())

                # test loss
                l1_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i]) / 400
                
                # recon_z_node_loss_ = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                # graph_z_node_loss_ = recon_z_node_loss_/graph_num_nodes
                
                # recon_z_graph_loss_ = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                # l3_loss = (graph_z_node_loss_) + (recon_z_graph_loss_)
                
                if data[i].y.item() == 0:
                    total_loss += l1_loss + node_loss + w_distance / 2
                else:
                    total_loss_anomaly += l1_loss + node_loss + w_distance / 2
                
                start_node = end_node
            
            # node_loss = torch.norm(x_recon - data.x, p='fro')**2
            # node_loss = (node_loss/x_recon.size(0))
            # triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
            # l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
            # loss += node_loss + triplet_loss + l2_loss
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
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
    
    return auroc, auprc, precision, recall, f1, total_loss / len(val_loader), total_loss_anomaly / len(val_loader)


#%%
def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, early_stopping, epochs, device):
    best_auroc = 0
    best_model = None
    
    # wandb 실험 초기화
    # wandb.init(project="graph anomaly detection", name=f"fold-{fold}", reinit=True)
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        auroc, auprc, precision, recall, f1, val_loss, val_loss_anomaly = evaluate_model(model, val_loader, device)
        
        scheduler.step(auroc)
        early_stopping(val_loss, model)
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_model = model.state_dict()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Loss Anomaly: {val_loss_anomaly:.4f}')
        print(f'AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        # # wandb 로깅
        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        #     "val_loss_anomaly": val_loss_anomaly,
        #     "auroc": auroc,
        #     "auprc": auprc,
        #     "precision": precision,
        #     "recall": recall,
        #     "f1": f1,
        #     "learning_rate": optimizer.param_groups[0]['lr']
        # })
        
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break
    
    model.load_state_dict(best_model)
    # wandb.finish()  # wandb 실험 종료
    return model, best_auroc


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset')

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
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
patience: list = args.patience
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
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv2 = GCNConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        # GCNConv 층에 대한 특별한 초기화
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.conv1.lin.weight, gain=gain)
        # nn.init.xavier_uniform_(self.conv2.lin.weight, gain=gain)
        nn.init.zeros_(self.conv1.bias)
        # nn.init.zeros_(self.conv2.bias)

        # BatchNorm 층 초기화
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        # nn.init.constant_(self.bn2.weight, 1)
        # nn.init.constant_(self.bn2.bias, 0)

        # Shortcut 층 초기화 (Linear인 경우)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight, gain=1.0)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        # x = self.bn2(self.conv2(x, edge_index))
        # x = self.dropout(x)
        return F.relu(x + residual)
    
    
class Feature_Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dims, num_features):
        super(Feature_Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList()
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
    

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, input_dim, max_nodes, threshold=0.5):
        super(BilinearEdgeDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.threshold = threshold
        self.max_nodes = max_nodes
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        # z의 실제 노드 수
        actual_nodes = z.size(0)
        
        # 원래 크기의 인접 행렬 생성
        adj = torch.sigmoid(torch.mm(torch.mm(z, self.weight), z.t()))
        adj_binary = (adj > self.threshold).float()
        
        # 최대 크기에 맞춰 패딩
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj_binary
        
        return padded_adj


#%%    
class GRAPH_AUTOENCODER(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.2):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder_blocks = nn.ModuleList()        
        self.encoder_node_blocks = nn.ModuleList()        
        self.encoder_sub_blocks = nn.ModuleList()
        
        self.edge_decoder = BilinearEdgeDecoder(hidden_dims[-1], max_nodes, threshold=0.5)
        self.feature_decoder = Feature_Decoder(hidden_dims[-1], hidden_dims, num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        
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

        # aug_data = self.conservative_augment_molecular_graph(data)
        # aug_x, aug_edge_index, aug_batch = aug_data.x, aug_data.edge_index, aug_data.batch
        
        # # latent vector
        # aug_z = self.encode(aug_x, aug_edge_index)
        # aug_z = self.dropout(aug_z)
        # aug_z_g = global_max_pool(aug_z, aug_batch)  # Aggregate features for classification
        
        # adjacency matrix
        # adj = adj_original(edge_index, batch, self.max_nodes)
        
        # latent vector
        # z = self.encode(x, edge_index)
        # z = self.dropout(z)
        
        # perturbation
        # z_prime = add_gaussian_perturbation(z)
        
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
        z_g = global_max_pool(z, batch) # Aggregate features for classification
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
        
        return adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z, aug_z_g
    
    
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
    
    def conservative_augment_molecular_graph(self, graph, node_attr_noise_std=0.01, edge_mask_prob=0.03):
        augmented_graph = graph.clone()
        
        # 1. 노드 특성에 미세한 가우시안 노이즈 추가
        if graph.x is not None:
            noise = torch.randn_like(graph.x) * node_attr_noise_std
            augmented_graph.x = graph.x + noise
        
        # 2. 매우 낮은 확률로 일부 엣지 마스킹 (완전히 제거하지 않음)
        if random.random() < edge_mask_prob:
            edge_index = augmented_graph.edge_index
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > 0.1  # 10%의 엣지만 마스킹
            masked_edge_index = edge_index[:, mask]
            
            # 마스킹된 엣지의 정보를 별도로 저장
            augmented_graph.masked_edges = edge_index[:, ~mask]
            augmented_graph.edge_index = masked_edge_index
        
        return augmented_graph
         
    def _init_weights(self, module):
        if isinstance(module, ResidualBlock):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, BilinearEdgeDecoder):
            nn.init.xavier_uniform_(module.weight, gain=0.01)  
            
            
#%%
'''DATASETS'''
splits = get_ad_split_TU(dataset_name, n_cross_val, random_seed)
loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, splits[0])

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset[0].x.shape[1]}')
print(f'Number of edge features: {graph_dataset_.num_edge_features}')


#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = meta['num_feat']
max_nodes = meta['max_nodes']

model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)


#%%
if dataset_name == 'COX2':
    dataset_AN = False
    print(dataset_AN)
    
    
#%%
'''TRAIN PROCESS'''
torch.autograd.set_detect_anomaly(True)  

all_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(graph_dataset, labels)):
    print(f"Fold {fold + 1}/{n_cross_val}")
    
    if dataset_AN:
        train_normal_idx = [idx for idx in train_idx if labels[idx] == 1]
        train_dataset = [graph_dataset[i] for i in train_normal_idx]
        val_dataset = [graph_dataset[i] for i in val_idx]

        for idx, data in enumerate(train_dataset):
            data.y = 0
            data['idx'] = idx

        for data in val_dataset:
            data.y = 1 if data.y == 0 else 0      
            
    else:
        train_normal_idx = [idx for idx in train_idx if labels[idx] == 0]
        train_dataset = [graph_dataset[i] for i in train_normal_idx]
        val_dataset = [graph_dataset[i] for i in val_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    
    print(f"  Training set size (normal only): {len(train_dataset)}")
    print(f"  Validation set size (normal + abnormal): {len(val_dataset)}")
    
    model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
    early_stopping = EarlyStopping(patience=777, verbose=True)
    
    # for epoch in range(epochs):
    #     train_loss = train(model, train_loader, optimizer)
    #     auroc, auprc, precision, recall, f1, max_AUC, test_loss, test_loss_anomaly = evaluate_model(model, val_loader)
            
    #     scheduler.step(auroc)  # AUC 기반으로 학습률 조정
    #     early_stopping(test_loss, model)

    best_model, best_auroc = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, early_stopping, epochs, device)

    final_auroc, final_auprc, final_precision, final_recall, final_f1, _, _ = evaluate_model(best_model, val_loader, device)
    all_results.append((final_auroc, final_auprc, final_precision, final_recall, final_f1))

mean_results = np.mean(all_results, axis=0)
std_results = np.std(all_results, axis=0)

print("\nOverall 5-Fold Cross-Validation Results:")
print(f"AUROC: {mean_results[0]:.4f} ± {std_results[0]:.4f}")
print(f"AUPRC: {mean_results[1]:.4f} ± {std_results[1]:.4f}")
print(f"Precision: {mean_results[2]:.4f} ± {std_results[2]:.4f}")
print(f"Recall: {mean_results[3]:.4f} ± {std_results[3]:.4f}")
print(f"F1 Score: {mean_results[4]:.4f} ± {std_results[4]:.4f}")

# # wandb에 최종 결과 로깅
# wandb.init(project="graph anomaly detection", name="final-results", reinit=True)
# wandb.log({
#     "mean_auroc": mean_results[0],
#     "std_auroc": std_results[0],
#     "mean_auprc": mean_results[1],
#     "std_auprc": std_results[1],
#     "mean_precision": mean_results[2],
#     "std_precision": std_results[2],
#     "mean_recall": mean_results[3],
#     "std_recall": std_results[3],
#     "mean_f1": mean_results[4],
#     "std_f1": std_results[4]
# })

# wandb.finish()

# %%
splits = get_ad_split_TU(dataset_name, n_cross_val, random_seed)

def run(random_seed, split=None, device=device):
    set_seed(random_seed)
    
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split)
    
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']

    model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 

    train_loader = loaders['train']
    test_loader = loaders['test']

    for epoch in range(1, epochs+1):

        model.train()
        loss_all = 0
        num_sample = 0
        for data in train_loader:
            print(data)
            optimizer.zero_grad()
            data = data.to(device)
            y, y_hyper, node_imp, edge_imp = model(data)
            loss = model.loss_nce(y, y_hyper).mean()
            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
            loss.backward()
            optimizer.step()

        info_train = 'Epoch {:3d}, Loss CL {:.4f}'.format(epoch, loss_all / num_sample)

        if epoch % args.log_interval == 0:
            model.eval()

            # anomaly detection
            all_ad_true = []
            all_ad_score = []
            for data in test_loader:
                all_ad_true.append(data.y.cpu())
                data = data.to(device)
                with torch.no_grad():
                    y, y_hyper, _, _ = model(data)
                    ano_score = model.loss_nce(y, y_hyper)
                all_ad_score.append(ano_score.cpu())

            ad_true = torch.cat(all_ad_true)
            ad_score = torch.cat(all_ad_score)
            ad_auc = roc_auc_score(ad_true, ad_score)

            info_test = 'AD_AUC:{:.4f}'.format(ad_auc)

            # explanation
            if is_xgad:
                all_node_explain_true = []
                all_node_explain_score = []
                all_edge_explain_true = []
                all_edge_explain_score = []
                for data in explain_loader:
                    data = data.to(device)
                    with torch.no_grad():
                        node_score = model.explainer(data.x, data.edge_index, data.batch)
                        edge_score = model.lift_node_score_to_edge_score(node_score, data.edge_index)
                    all_node_explain_true.append(data.node_label.cpu())
                    all_node_explain_score.append(node_score.cpu())
                    all_edge_explain_true.append(data.edge_label.cpu())
                    all_edge_explain_score.append(edge_score.cpu())

                x_node_true = torch.cat(all_node_explain_true)
                x_node_score = torch.cat(all_node_explain_score)
                x_node_auc = roc_auc_score(x_node_true, x_node_score)

                x_edge_true = torch.cat(all_edge_explain_true)
                x_edge_score = torch.cat(all_edge_explain_score)
                x_edge_auc = roc_auc_score(x_edge_true, x_edge_score)

                info_test += '| X AUC(node):{:.4f} | X AUC(edge):{:.4f}'.format(x_node_auc, x_edge_auc)

            print(info_train + '   ' + info_test)

    if is_xgad:
        return ad_auc, x_node_auc, x_edge_auc
    else:
        return ad_auc

if __name__ == '__main__':
    args = arg_parse()
    ad_aucs = []
    if args.dataset in explainable_datasets:
        x_node_aucs = []
        x_edge_aucs = []
        splits = [None] * args.num_trials
    else:
        splits = get_ad_split_TU(args, fold=5)
        key_auc_list = []

    for trial in range(args.num_trials):
        results = run(args, seed=trial, split=splits[trial])
        if args.dataset in explainable_datasets:
            ad_auc, x_node_auc, x_edge_auc = results
            ad_aucs.append(ad_auc)
            x_node_aucs.append(x_node_auc)
            x_edge_aucs.append(x_edge_auc)
        else:
            ad_auc = results
            ad_aucs.append(ad_auc)

    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    if args.dataset in explainable_datasets:
        results += ' | X AUC (node): {:.2f}+-{:.2f}'.format(np.mean(x_node_aucs) * 100, np.std(x_node_aucs) * 100)
        results += ' | X AUC (edge): {:.2f}+-{:.2f}'.format(np.mean(x_edge_aucs) * 100, np.std(x_edge_aucs) * 100)

    print('[FINAL RESULTS] ' + results)