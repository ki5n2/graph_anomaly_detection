import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import BertEncoder
from .layers import FeatureDecoder
from .decoder import BilinearEdgeDecoder
from utils.topology import process_batch_graphs

class GraphAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT,
                 nhead, num_layers_BERT, num_layers, dropout_rate=0.1):
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
        
        self.transformer_d = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dims[-1],
                nhead=nhead,
                dim_feedforward=hidden_dims[-1] * 4,
                dropout=dropout_rate,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.edge_decoder = BilinearEdgeDecoder(max_nodes)
        
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        # 위상학적 특성을 위한 예측기 (5개의 통계량)
        self.stats_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 5)
        )
        
        self._init_weights()

    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, 
                is_pretrain=False, edge_training=False):
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
        
        # Fine-tuning phase
        node_embeddings = self.encoder(
            x, edge_index, batch, num_graphs,
            training=False,
            edge_training=False
        )
        
        # 배치 처리
        z_list = [node_embeddings[batch == i] for i in range(num_graphs)]
        max_nodes_in_batch = max(z.size(0) for z in z_list)
        
        # CLS 토큰 추가 및 패딩
        z_with_cls_batch, mask = self._prepare_batch(z_list, max_nodes_in_batch)
        
        # Transformer 처리
        encoded = self.transformer_d(z_with_cls_batch, src_key_padding_mask=mask)
        
        # CLS 토큰 출력 처리
        cls_output = encoded[:, 0, :]  # [batch_size, hidden_dim]
        
        # 노드 출력 처리
        node_outputs = []
        start_idx = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            node_outputs.append(encoded[i, 1:num_nodes+1, :])
        
        u = torch.cat(node_outputs, dim=0)
        
        # 특징 재구성
        u_prime = self.u_mlp(u)
        x_recon = self.feature_decoder(u_prime)
        
        # 위상학적 통계량 예측
        stats_pred = self.stats_predictor(cls_output)
        
        return x_recon, stats_pred

    def _prepare_batch(self, z_list, max_nodes_in_batch):
        """배치 준비 (CLS 토큰 추가 및 패딩)"""
        z_with_cls_list = []
        mask_list = []
        
        for z in z_list:
            num_nodes = z.size(0)
            
            # 패딩
            padding = torch.zeros(max_nodes_in_batch - num_nodes, 
                                z.size(1), device=z.device)
            z_padded = torch.cat([z, padding], dim=0)
            
            # CLS 토큰 추가
            z_with_cls = torch.cat([
                self.cls_token.repeat(1, 1, 1),
                z_padded.unsqueeze(0)
            ], dim=1)
            z_with_cls_list.append(z_with_cls)
            
            # 패딩 마스크 생성
            mask = torch.zeros(max_nodes_in_batch + 1, 
                             dtype=torch.bool, device=z.device)
            mask[num_nodes+1:] = True
            mask_list.append(mask)
        
        return torch.cat(z_with_cls_list, dim=0), torch.stack(mask_list)

def persistence_stats_loss(pred_stats, true_stats):
    """위상학적 통계량에 대한 손실 함수"""
    return F.mse_loss(pred_stats, true_stats)
