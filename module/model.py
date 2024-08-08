import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from module.utils import add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon


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
        print(batched_pos_subgraphs)
        pos_sub_z, pos_new_edge_index = self.process_subgraphs(batched_pos_subgraphs)
        pos_sub_z = torch.cat(pos_sub_z) # (number of nodes, embedded size)
        
        unique_pos_batch, new_pos_batch = torch.unique(pos_batch, return_inverse=True)
        
        # print("pos_x size:", pos_x.size())
        # print("pos_sub_z size:", pos_sub_z.size())
        # print("pos_batch size:", pos_batch.size())
        # print("new_pos_batch size:", new_pos_batch.size())
        # print("Unique values in new_pos_batch:", new_pos_batch.unique())
        
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
            bn_module = nn.BatchNorm1d(x.size(1)).to('cuda')
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
            bn_module = nn.BatchNorm1d(x.size(1)).to('cuda')
            x = bn_module(x)
        x = self.encoders_subgraphs[-1](x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x
   
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
