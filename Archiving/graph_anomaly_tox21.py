#%%
import os
import wandb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import init
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_max_pool

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from module.model import GRAPH_AUTOENCODER
from module.loss import Triplet_loss, loss_cal
from module.utils import set_device
    

#%%
def train(model, train_loader, optimizer, threshold=0.5):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, batch, x_recon, adj_recon_list, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z = model(data)
        
        loss = 0
        start_node = 0
        
        for i in range(len(data)): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            
            node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
            
            edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
            
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
def evaluation(model, test_loader, threshold = 0.5):
    model.eval()
    max_AUC = 0.0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  
            adj, z, z_g, batch, x_recon, adj_recon_list, _, _, _, _, _ = model(data)  
            
            label_y=[]
            label_pred = []

            start_node = 0
            for i in range(data.num_graphs): 
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() 
                end_node = start_node + num_nodes

                node_recon_1 = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                node_recon_error = node_recon_1 / 200
             
                edge_recon_1 = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
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
            
            if test_roc_ab > max_AUC:
                max_AUC=test_roc_ab
        
        auroc_final = max_AUC
        
    return auroc_final


#%%
'''ARGPARSER'''

parser = argparse.ArgumentParser()

parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset/data')
parser.add_argument("--dataset-name", type=str, default='NCI1')

parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])
parser.add_argument("--n-test-anomaly", type=int, default=200)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--learning-rate", type=float, default=0.00001)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--fine-tune", type=bool, default=True)
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
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size
batch_size: int = args.batch_size
random_seed: int = args.random_seed
epochs: int = args.epochs

learning_rate: float = args.learning_rate
test_size: float = args.test_size
fine_tune: bool = args.fine_tune
pretrained: bool = args.pretrained

# device = torch.device('cpu')
device = set_device()

wandb.init(project="graph anomaly detection", entity="ki5n2")

wandb.config.update(args)

wandb.config = {
  "random_seed": random_seed,
  "n_test_anomaly": n_test_anomaly,
  "learning_rate": 0.0001,
  "epochs": 100
}

device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    
#%%
'''DATASETS'''
graph_dataset = TUDataset(root='./dataset/data', name='Tox21_HSE_training')
graph_dataset = graph_dataset.shuffle()

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of edge features: {graph_dataset.num_edge_features}')

dataset_normal = [data for data in graph_dataset if data.y.item() == 0]
dataset_anomaly = [data for data in graph_dataset if data.y.item() == 1]

print(f"Number of normal samples: {len(dataset_normal)}")
print(f"Number of anomaly samples: {len(dataset_anomaly)}")

train_normal_data, test_normal_data = train_test_split(dataset_normal, test_size=test_size, random_state=random_seed)
evaluation_data = test_normal_data + dataset_anomaly[:n_test_anomaly]

train_loader = DataLoader(train_normal_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(evaluation_data, batch_size=test_batch_size, shuffle=True)

print(f"Number of samples in the evaluation dataset: {len(evaluation_data)}")
print(f"Number of test normal data: {len(test_normal_data)}")
print(f"Number of test anomaly samples: {len(dataset_anomaly[:n_test_anomaly])}")
print(f"Ratio of test anomaly: {len(dataset_anomaly[:n_test_anomaly]) / len(evaluation_data)}")


#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = graph_dataset.num_features
model = GRAPH_AUTOENCODER(num_features, hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

#%%
'''TRAIN PROCESS'''
torch.autograd.set_detect_anomaly(True)  

epochs = 100
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer)
    print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}')

    auc_score = evaluation(model, test_loader)
    wandb.log({"epoch": epoch, "train loss": train_loss, "test auc_score": auc_score})
    print(f'Epoch {epoch+1}: Validation AUC = {auc_score:.4f}')
    
wandb.finish()
