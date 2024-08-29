import torch
import random
import numpy as np

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_undirected, to_networkx


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def add_gaussian_perturbation(z, epsilon=0.1):
    """
    Add Gaussian perturbations to the encoder output z to create z'
    :param z: torch.Tensor, the output of the encoder
    :param epsilon: float, the standard deviation of the Gaussian noise to add
    :return: torch.Tensor, the perturbed output z'
    """
    # Gaussian noise generation
    noise = torch.randn_like(z) * epsilon
    z_prime = z + noise
        
    return z_prime


def randint_exclude(data, start_node, end_node):
    exclude = set(range(start_node, end_node))
    choices = list(set(range(data.x.size(0))) - exclude)
        
    return torch.tensor(choices[torch.randint(0, len(choices), (1,)).item()])


def extract_subgraph(data, node_idx, start_node, end_node, max_attempts=10, device='cuda'):
    attempts = 0
    subgraph = data.clone()
    while attempts < max_attempts:
        edge_index = to_undirected(subgraph.edge_index, num_nodes=subgraph.x.size(0))
        mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        node_indices = torch.cat((edge_index[0][mask], edge_index[1][mask])).unique()

        if node_indices.numel() > 0:  # Check if there are connected nodes
            subgraph.x = data.x[node_indices]
            edge_mask = torch.tensor([start in node_indices and end in node_indices for start, end in data.edge_index.t()], dtype=torch.bool)
            subgraph.edge_index = data.edge_index[:, edge_mask]
            subgraph.edge_attr = data.edge_attr[edge_mask] if 'edge_attr' in data else None
                    
            if 'batch' in data:
                node_to_graph = data.batch[node_indices]  # Find new graph IDs
                unique_graphs = node_to_graph.unique(sorted=True)
                batch_device = torch.arange(unique_graphs[0], unique_graphs[0] + 1).to(device)
                subgraph.batch = batch_device.repeat_interleave(
                torch.bincount(node_to_graph).sort(descending=True)[0][0]
                )
                    
            if 'y' in data:
                subgraph.y = data.y[unique_graphs]
                
            if 'ptr' in subgraph:
                del subgraph.ptr
                
            return subgraph

        # No connected nodes, select a new node_idx and try again
        if start_node <= node_idx < end_node:
            node_idx = torch.randint(start_node, end_node, size=(1,)).item()
        else:
            node_idx = randint_exclude(data, start_node, end_node).item()
        attempts += 1

    return None  # Return None after max_attempts
    

def batch_nodes_subgraphs(data):
    batched_target_nodes = []
    batched_initial_nodes = []
    batched_target_node_features = []
    pos_subgraphs = []
    neg_subgraphs = []
        
    start_node = 0        
    for i in range(len(data)): 
        num_nodes = (data.batch == i).sum().item() 
        end_node = start_node + num_nodes
            
        target_node = torch.randint(start_node, end_node, size=(1,)).item()
        pos_subgraph = extract_subgraph(data, node_idx=target_node, start_node=start_node, end_node=end_node)
        target_node_feature = data.x[target_node]
                        
        initial_node = randint_exclude(data, start_node=start_node, end_node=end_node).item()
        neg_subgraph = extract_subgraph(data, node_idx=initial_node, start_node=start_node, end_node=end_node)
            
        batched_target_nodes.append(target_node)
        batched_initial_nodes.append(initial_node)
        pos_subgraphs.append(pos_subgraph)
        neg_subgraphs.append(neg_subgraph)
        batched_target_node_features.append(target_node_feature)
            
        start_node = end_node
            
    batched_pos_subgraphs = Batch.from_data_list(pos_subgraphs)
    batched_neg_subgraphs = Batch.from_data_list(neg_subgraphs)
    batched_target_node_features = torch.stack(batched_target_node_features)
    # return batched_target_nodes, batched_initial_nodes, batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features
    return batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features
    
    
def adj_original(edge_index, batch):
    adj_matrices = []
    for batch_idx in torch.unique(batch):
        # 현재 그래프에 속하는 노드들의 마스크
        mask = (batch == batch_idx)
        # 현재 그래프의 에지 인덱스 추출
        sub_edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
        # 노드 인덱스를 0부터 시작하도록 재매핑
        _, sub_edge_index = torch.unique(sub_edge_index, return_inverse=True)
        sub_edge_index = sub_edge_index.reshape(2, -1)
        # 인접 행렬 생성
        adj_matrix = to_dense_adj(sub_edge_index, max_num_nodes=sum(mask).item())[0]
        adj_matrices.append(adj_matrix)
        
    return adj_matrices
    
    
def adj_recon(z, z_prime, batch):
    adj_recon_list = []
    adj_recon_prime_list = []
        
    # Iterate over each graph in the batch
    for batch_idx in torch.unique(batch):
        mask = (batch == batch_idx)
            
        # Select the latent vectors corresponding to the current graph
        z_graph = z[mask]
        z_prime_graph = z_prime[mask]
                
        # Reconstruct adjacency matrices for the current graph
        adj_recon_graph = torch.sigmoid(torch.mm(z_graph, z_graph.t()))
        adj_recon_prime_graph = torch.sigmoid(torch.mm(z_prime_graph, z_prime_graph.t()))

        # Append the reconstructed matrices to the lists
        adj_recon_list.append(adj_recon_graph)
        adj_recon_prime_list.append(adj_recon_prime_graph)
                
    return adj_recon_list, adj_recon_prime_list


def visualize(graph, color='skyblue', edge_color='blue'):
    G = to_networkx(graph, to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, edge_color=edge_color)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# GLADC
import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from packaging import version


def node_iter(G):
    if version.parse(nx.__version__) < version.parse("2.0"):
        return G.nodes()  # For NetworkX versions < 2.0
    else:
        return G.nodes()  # For NetworkX versions >= 2.0

def node_dict(G):
    if version.parse(nx.__version__) >= version.parse("2.1"):
        return G.nodes   # For NetworkX versions >= 2.1
    else:
        return G.node     # For older versions
    

def adj_process(adjs):
    g_num, n_num, n_num = adjs.shape
    adjs = adjs.detach()
    for i in range(g_num):
        adjs[i] += torch.eye(n_num).cuda()
        adjs[i][adjs[i]>0.] = 1.
        degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
        degree_matrix = torch.pow(degree_matrix,-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix)
        adjs[i] = torch.mm(degree_matrix, adjs[i])
    return adjs


def NormData(adj):
    adj=adj.tolist()
    adj_norm = normalize_adj(adj )
    adj_norm = adj_norm.toarray()
    #adj = adj + sp.eye(adj.shape[0])
    #adj = adj.toarray()
    #feat = feat.toarray()
    return adj_norm



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

