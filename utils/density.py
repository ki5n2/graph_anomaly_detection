import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut

def scott_rule_bandwidth(X):
    """Scott의 규칙을 사용한 KDE bandwidth 계산"""
    n = len(X)
    d = X.shape[1]
    sigma = np.std(X, axis=0)
    bandwidth = np.power(n, -1./(d+4)) * sigma
    
    if d > 1:
        bandwidth = np.prod(bandwidth) ** (1./d)
    
    return bandwidth

def loocv_bandwidth_selection(X, bandwidths=None, cv=None):
    """Leave-one-out 교차 검증을 통한 optimal bandwidth 선택"""
    if bandwidths is None:
        scott_bw = scott_rule_bandwidth(X)
        bandwidths = np.logspace(
            np.log10(scott_bw/5), 
            np.log10(scott_bw*5), 
            20
        )
    
    if cv is None:
        cv = LeaveOneOut()
    
    cv_scores = {bw: 0.0 for bw in bandwidths}
    
    for train_idx, test_idx in cv.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        for bw in bandwidths:
            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            kde.fit(X_train)
            log_likelihood = kde.score(X_test)
            cv_scores[bw] += log_likelihood
    
    optimal_bandwidth = max(cv_scores.items(), key=lambda x: x[1])[0]
    
    return optimal_bandwidth, cv_scores

class DensityEstimator:
    """밀도 기반 이상치 탐지를 위한 클래스"""
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.kde = None
        
    def fit(self, X):
        """데이터에 맞춰 KDE 학습"""
        if self.bandwidth is None:
            self.bandwidth, _ = loocv_bandwidth_selection(X)
        
        self.kde = KernelDensity(
            bandwidth=self.bandwidth,
            kernel='gaussian'
        )
        self.kde.fit(X)
        return self
        
    def score_samples(self, X):
        """샘플의 밀도 점수 계산"""
        if self.kde is None:
            raise ValueError("Model not fitted yet!")
            
        log_density = self.kde.score_samples(X)
        log_density = np.nan_to_num(log_density, neginf=-10000)
        anomaly_scores = -log_density
        return np.clip(anomaly_scores, 0, 10000)
