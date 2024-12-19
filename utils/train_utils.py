import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

class DensityBasedScoring:
    def __init__(self, bandwidth=0.5):
        self.kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """정상 데이터의 2D 특징에 대해 KDE를 학습"""
        X_scaled = self.scaler.fit_transform(X)
        self.kde.fit(X_scaled)
        
    def score_samples(self, X):
        """샘플들의 밀도 기반 이상 스코어 계산"""
        X_scaled = self.scaler.transform(X)
        log_density = self.kde.score_samples(X_scaled)
        log_density = np.nan_to_num(log_density, neginf=-10000)
        anomaly_scores = -log_density
        return np.clip(anomaly_scores, 0, 10000)

def persistence_stats_loss(pred_stats, true_stats):
    """위상 특징 손실 계산"""
    return F.mse_loss(pred_stats, true_stats)

def set_seed(seed):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
