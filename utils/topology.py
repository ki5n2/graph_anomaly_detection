import numpy as np
import gudhi as gd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Batch

def split_batch_graphs(data):
    """배치 데이터를 개별 그래프로 분할"""
    graphs = []
    for i in range(len(data.ptr) - 1):
        start, end = data.ptr[i].item(), data.ptr[i + 1].item()
        x = data.x[start:end]
        mask = (data.edge_index[0] >= start) & (data.edge_index[1] < end)
        edge_index = data.edge_index[:, mask]
        edge_index = edge_index - start
        graphs.append((x, edge_index))
    return graphs

def compute_persistence_and_betti(graph_distance_matrix, max_dimension=2):
    """Persistent Homology 계산"""
    try:
        rips_complex = gd.RipsComplex(
            distance_matrix=graph_distance_matrix, 
            max_edge_length=2.0
        )
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        simplex_tree.compute_persistence()
        return simplex_tree.persistence()
    except Exception as e:
        print(f"Error in persistence computation: {str(e)}")
        return []

def process_batch_graphs(data):
    """배치 그래프에 대한 위상학적 특성 계산"""
    graphs = split_batch_graphs(data)
    true_stats_list = []
    
    for x, edge_index in graphs:
        try:
            distance_matrix = squareform(pdist(x.cpu().numpy(), metric='euclidean'))
            persistence_diagram = compute_persistence_and_betti(distance_matrix)
            
            stats = calculate_persistence_statistics(persistence_diagram)
            true_stats_list.append(stats)
                
        except Exception as e:
            true_stats_list.append(get_default_stats())
    
    data.true_stats = torch.tensor([
        [stats['mean_survival'], stats['max_survival'], 
         stats['variance_survival'], stats['mean_birth'], 
         stats['mean_death']]
        for stats in true_stats_list
    ], dtype=torch.float32)
    
    return data

def calculate_persistence_statistics(persistence_diagram):
    """Persistence diagram에서 통계 계산"""
    if not persistence_diagram:
        return get_default_stats()
    
    lifetimes = [death - birth for _, (birth, death) in persistence_diagram 
                 if death != float('inf')]
    births = [birth for _, (birth, death) in persistence_diagram]
    deaths = [death for _, (birth, death) in persistence_diagram 
             if death != float('inf')]
    
    return {
        "mean_survival": np.mean(lifetimes) if lifetimes else 0.0,
        "max_survival": np.max(lifetimes) if lifetimes else 0.0,
        "variance_survival": np.var(lifetimes) if lifetimes else 0.0,
        "mean_birth": np.mean(births) if births else 0.0,
        "mean_death": np.mean(deaths) if deaths else 0.0
    }

def get_default_stats():
    """기본 통계값 반환"""
    return {
        "mean_survival": 0.0,
        "max_survival": 0.0,
        "variance_survival": 0.0,
        "mean_birth": 0.0,
        "mean_death": 0.0
    }

def persistence_stats_loss(pred_stats, true_stats):
    """위상학적 통계량에 대한 손실 함수"""
    return F.mse_loss(pred_stats, true_stats)
