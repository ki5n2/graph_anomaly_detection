import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, max_nodes):
        super(BilinearEdgeDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        actual_nodes = z.size(0)
        
        # 노드 벡터 정규화
        z_norm = F.normalize(z, p=2, dim=1)
        
        # 코사인 유사도 계산
        cos_sim = torch.mm(z_norm, z_norm.t())
        adj = self.sigmoid(cos_sim)
        
        # 자기 자신과의 연결 제거
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        
        # max_nodes 크기로 패딩
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj
    