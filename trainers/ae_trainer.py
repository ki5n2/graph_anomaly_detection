import torch
from utils.batch_utils import BatchUtils
from utils.topology import process_batch_graphs, persistence_stats_loss
from utils.density import DensityEstimator
import numpy as np

class AutoencoderTrainer:
    def __init__(self, model, optimizer, device, dataset_name, current_time, alpha, gamma):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataset_name = dataset_name
        self.current_time = current_time
        self.alpha = alpha
        self.gamma = gamma
        self.density_estimator = DensityEstimator()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        reconstruction_errors = []
        
        for data in train_loader:
            # 위상학적 특성 계산
            data = process_batch_graphs(data)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            x_recon, stats_pred = self.model(
                data.x, data.edge_index, data.batch,
                data.num_graphs, is_pretrain=False
            )
            
            # 손실 계산
            loss = 0
            start_node = 0
            
            for i in range(data.num_graphs):
                num_nodes = (data.batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # 노드 재구성 손실
                if self.dataset_name == 'AIDS':
                    node_loss = torch.norm(
                        data.x[start_node:end_node] - 
                        x_recon[start_node:end_node], 
                        p='fro'
                    )**2
                else:
                    node_loss = torch.norm(
                        data.x[start_node:end_node] - 
                        x_recon[start_node:end_node], 
                        p='fro'
                    )**2 / num_nodes
                
                # 위상학적 손실
                stats_loss = persistence_stats_loss(
                    stats_pred[i],
                    data.true_stats[i]
                )
                
                # 전체 손실
                total_error = self.alpha * node_loss + self.gamma * stats_loss
                loss += total_error
                
                # 에러 정보 저장
                reconstruction_errors.append({
                    'reconstruction': node_loss.item() * self.alpha,
                    'topology': stats_loss.item() * self.gamma,
                    'type': 'train_normal'
                })
                
                start_node = end_node
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader), reconstruction_errors

    def evaluate(self, test_loader, train_errors, epoch, trial):
        self.model.eval()
        test_errors = []
        all_labels = []
        
        with torch.no_grad():
            for data in test_loader:
                data = process_batch_graphs(data)
                data = data.to(self.device)
                
                # Forward pass
                x_recon, stats_pred = self.model(
                    data.x, data.edge_index, data.batch,
                    data.num_graphs, is_pretrain=False
                )
                
                # 에러 수집
                start_node = 0
                for i in range(data.num_graphs):
                    num_nodes = (data.batch == i).sum().item()
                    end_node = start_node + num_nodes
                    
                    # 노드 재구성 에러
                    if self.dataset_name == 'AIDS':
                        node_loss = torch.norm(
                            data.x[start_node:end_node] - 
                            x_recon[start_node:end_node], 
                            p='fro'
                        )**2
                    else:
                        node_loss = torch.norm(
                            data.x[start_node:end_node] - 
                            x_recon[start_node:end_node], 
                            p='fro'
                        )**2 / num_nodes
                    
                    # 위상학적 에러
                    stats_loss = persistence_stats_loss(
                        stats_pred[i],
                        data.true_stats[i]
                    )
                    
                    test_errors.append({
                        'reconstruction': node_loss.item() * self.alpha,
                        'topology': stats_loss.item() * self.gamma,
                        'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
                    })
                    
                    start_node = end_node
                    all_labels.extend([data.y[i].item()])

        # 밀도 기반 이상치 탐지
        train_features = np.array([[e['reconstruction'], e['topology']] 
                                 for e in train_errors])
        test_features = np.array([[e['reconstruction'], e['topology']] 
                                for e in test_errors])
        
        self.density_estimator.fit(train_features)
        anomaly_scores = self.density_estimator.score_samples(test_features)
        
        metrics = self._calculate_metrics(anomaly_scores, all_labels)
        
        return metrics, test_errors

    def _calculate_metrics(self, scores, labels):
        # ROC 곡선 계산
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)
        
        # PR 곡선 계산
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall, precision)
        
        # 최적 임계값 찾기
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        pred_labels = (scores > optimal_threshold).astype(int)
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'precision': precision_score(labels, pred_labels),
            'recall': recall_score(labels, pred_labels),
            'f1': f1_score(labels, pred_labels)
        }
