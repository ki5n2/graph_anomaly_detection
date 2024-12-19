import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, dataset_name, current_time):
        self.dataset_name = dataset_name
        self.current_time = current_time
        
    def evaluate_model(self, model, test_loader, train_errors, epoch, trial, device):
        """모델 평가 및 시각화"""
        model.eval()
        all_labels = []
        all_scores = []
        reconstruction_errors_test = []
        
        with torch.no_grad():
            for data in test_loader:
                data = BatchUtils.process_batch_graphs(data)
                data = data.to(device)
                
                # Forward pass
                x_recon, stats_pred = model(data.x, data.edge_index, data.batch, 
                                          data.num_graphs, is_pretrain=False)
                
                # 에러 계산 및 결과 수집
                self._collect_errors(data, x_recon, stats_pred, reconstruction_errors_test)
                all_labels.extend(data.y.cpu().numpy())
        
        # 성능 평가
        metrics = self._calculate_metrics(reconstruction_errors_test)
        
        # 결과 시각화 및 저장
        self._visualize_results(train_errors, reconstruction_errors_test, 
                              epoch, trial, metrics)
        
        return metrics, reconstruction_errors_test

    def _collect_errors(self, data, x_recon, stats_pred, errors_list):
        """재구성 에러 수집"""
        start_node = 0
        for i in range(data.num_graphs):
            num_nodes = (data.batch == i).sum().item()
            end_node = start_node + num_nodes
            
            node_loss = self._calculate_node_loss(
                data.x[start_node:end_node],
                x_recon[start_node:end_node],
                num_nodes
            )
            
            stats_loss = persistence_stats_loss(
                stats_pred[i],
                data.true_stats[i]
            ).item() * gamma
            
            errors_list.append({
                'reconstruction': node_loss,
                'topology': stats_loss,
                'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
            })
            
            start_node = end_node

    def _calculate_metrics(self, test_errors):
        """성능 지표 계산"""
        normal_scores = np.array([e['reconstruction'] + e['topology'] 
                                for e in test_errors if e['type'] == 'test_normal'])
        anomaly_scores = np.array([e['reconstruction'] + e['topology'] 
                                 for e in test_errors if e['type'] == 'test_anomaly'])
        
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
        
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        auroc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        auprc = auc(recall, precision)
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'normal_scores': normal_scores,
            'anomaly_scores': anomaly_scores
        }

    def _visualize_results(self, train_errors, test_errors, epoch, trial, metrics):
        """결과 시각화 및 저장"""
        # 에러 분포 플롯
        self._plot_error_distribution(train_errors, test_errors, epoch, trial)
        
        # 밀도 등고선 플롯
        self._plot_density_contours(train_errors, test_errors, epoch, trial)
        
        # 결과 저장
        self._save_results(train_errors, test_errors, metrics, epoch, trial)

    def _plot_error_distribution(self, train_errors, test_errors, epoch, trial):
        """에러 분포 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 데이터 준비
        train_data = self._prepare_plot_data(train_errors, 'train_normal')
        test_normal = self._prepare_plot_data(test_errors, 'test_normal')
        test_anomaly = self._prepare_plot_data(test_errors, 'test_anomaly')
        
        # 일반 스케일 플롯
        self._plot_scatter(ax1, train_data, test_normal, test_anomaly, 
                          title=f'Error Distribution (Epoch {epoch})')
        
        # 로그 스케일 플롯
        self._plot_scatter(ax2, train_data, test_normal, test_anomaly,
                          title=f'Error Distribution - Log Scale (Epoch {epoch})',
                          log_scale=True)
        
        plt.tight_layout()
        self._save_plot(fig, f'error_distribution_epoch_{epoch}_fold_{trial}.png')
        plt.close()

    def _save_results(self, train_errors, test_errors, metrics, epoch, trial):
        """결과 저장"""
        results = {
            'train_errors': train_errors,
            'test_errors': test_errors,
            'metrics': metrics,
            'epoch': epoch,
            'trial': trial
        }
        
        save_path = (f'/root/default/GRAPH_ANOMALY_DETECTION/graph_anomaly_detection/'
                    f'results/{self.dataset_name}_time_{self.current_time}/'
                    f'results_epoch_{epoch}_fold_{trial}.json')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f)
