import torch
import torch.nn.functional as F
from utils.batch_utils import BatchUtils

class BertTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_embedding(self, train_loader):
        """BERT 마스크 토큰 재구성 훈련"""
        self.model.train()
        total_loss = 0
        num_sample = 0
        
        for data in train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            
            # 마스크 생성
            mask_indices = torch.rand(data.x.size(0), device=self.device) < 0.15
            
            # BERT 인코딩 및 마스크 토큰 예측
            node_embeddings, masked_outputs = self.model(
                data.x, data.edge_index, data.batch, data.num_graphs,
                mask_indices=mask_indices,
                is_pretrain=True
            )
            
            # 손실 계산
            mask_loss = torch.norm(
                masked_outputs - data.x[mask_indices], 
                p='fro'
            )**2 / mask_indices.sum()
            
            mask_loss.backward()
            self.optimizer.step()
            total_loss += mask_loss.item()
            num_sample += data.num_graphs
            
        return total_loss / len(train_loader), num_sample

    def train_edge_reconstruction(self, train_loader):
        """BERT 엣지 재구성 훈련"""
        self.model.train()
        total_loss = 0
        num_sample = 0
        
        for data in train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            
            # Edge reconstruction 수행
            node_embeddings, adj_recon_list = self.model(
                data.x, data.edge_index, data.batch, data.num_graphs,
                is_pretrain=True,
                edge_training=True
            )
            
            start_idx = 0
            edge_loss = 0
            
            for i in range(data.num_graphs):
                # 현재 그래프의 노드 수 계산
                mask = (data.batch == i)
                num_nodes = mask.sum().item()
                end_idx = start_idx + num_nodes
                
                # 현재 그래프의 edge_index 추출 및 조정
                graph_edges = data.edge_index[:, (data.edge_index[0] >= start_idx) & 
                                              (data.edge_index[0] < end_idx)]
                graph_edges = graph_edges - start_idx
                
                # 실제 adjacency matrix 생성
                true_adj = torch.zeros((self.model.encoder.edge_decoder.max_nodes,
                                      self.model.encoder.edge_decoder.max_nodes),
                                     device=self.device)
                true_adj[graph_edges[0], graph_edges[1]] = 1
                true_adj = true_adj + true_adj.t()
                true_adj = (true_adj > 0).float()
                
                # 손실 계산 (실제 노드가 있는 부분만)
                adj_recon = adj_recon_list[i]
                node_mask = torch.zeros_like(adj_recon, dtype=torch.bool)
                node_mask[:num_nodes, :num_nodes] = True
                
                mse_loss = torch.sum((adj_recon[node_mask] - true_adj[node_mask]) ** 2) / node_mask.sum()
                edge_loss += mse_loss
                
                start_idx = end_idx
            
            edge_loss = edge_loss / data.num_graphs
            
            edge_loss.backward()
            self.optimizer.step()
            
            total_loss += edge_loss.item()
            num_sample += data.num_graphs
            
        return total_loss / len(train_loader), num_sample

    def save_model(self, path):
        """모델 저장"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """모델 로드"""
        self.model.load_state_dict(torch.load(path))
