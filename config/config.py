import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset-name", type=str, default='COX2',
                      help="Dataset name (COX2, AIDS, DHFR, Tox21_p53, Tox21_HSE, Tox21_MMP)")
    parser.add_argument("--data-root", type=str, default='./dataset',
                      help="Root directory for datasets")
    parser.add_argument("--assets-root", type=str, default="./assets",
                      help="Root directory for saving models and results")
    
    parser.add_argument("--n-head-BERT", type=int, default=2,
                      help="Number of attention heads in BERT")
    parser.add_argument("--n-layer-BERT", type=int, default=2,
                      help="Number of transformer layers in BERT")
    parser.add_argument("--n-head", type=int, default=2,
                      help="Number of attention heads in main transformer")
    parser.add_argument("--n-layer", type=int, default=2,
                      help="Number of transformer layers in main transformer")
    parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256],
                      help="Hidden dimensions for the model")
    
    parser.add_argument("--BERT-epochs", type=int, default=1,
                      help="Number of epochs for BERT pre-training")
    parser.add_argument("--epochs", type=int, default=300,
                      help="Number of epochs for main training")
    parser.add_argument("--patience", type=int, default=5,
                      help="Patience for early stopping")
    parser.add_argument("--batch-size", type=int, default=300,
                      help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=128,
                      help="Testing batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                      help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                      help="Weight decay for optimizer")
    parser.add_argument("--dropout-rate", type=float, default=0.1,
                      help="Dropout rate")
    
    parser.add_argument("--alpha", type=float, default=1.0,
                      help="Weight for node reconstruction loss")
    parser.add_argument("--beta", type=float, default=5.0,
                      help="Weight for edge reconstruction loss")
    parser.add_argument("--gamma", type=float, default=100.0,
                      help="Weight for topology loss")
    parser.add_argument("--gamma-cluster", type=float, default=0.5,
                      help="Weight for clustering loss")
    parser.add_argument("--node-theta", type=float, default=0.03,
                      help="Node threshold for anomaly detection")
    parser.add_argument("--adj-theta", type=float, default=0.01,
                      help="Adjacency threshold for anomaly detection")
    
    parser.add_argument("--n-cross-val", type=int, default=5,
                      help="Number of cross validation folds (not used for Tox21)")
    parser.add_argument("--random-seed", type=int, default=1,
                      help="Random seed for reproducibility")
    parser.add_argument("--log-interval", type=int, default=5,
                      help="Interval for logging")
    
    parser.add_argument("--kde-bandwidth", type=float, default=None,
                      help="Bandwidth for KDE (None for automatic selection)")
    parser.add_argument("--density-score-threshold", type=float, default=None,
                      help="Threshold for density-based anomaly detection")
    
    parser.add_argument("--save-model", type=bool, default=True,
                      help="Whether to save the model")
    parser.add_argument("--load-pretrained", type=bool, default=False,
                      help="Whether to load pretrained BERT")
    parser.add_argument("--save-results", type=bool, default=True,
                      help="Whether to save results")
    
    parser.add_argument("--plot-distributions", type=bool, default=True,
                      help="Whether to plot error distributions")
    parser.add_argument("--plot-density", type=bool, default=True,
                      help="Whether to plot density estimations")
    
    parser.add_argument("--results-dir", type=str, default="results",
                      help="Directory for saving results")
    parser.add_argument("--plots-dir", type=str, default="plots",
                      help="Directory for saving plots")
    parser.add_argument("--models-dir", type=str, default="models",
                      help="Directory for saving models")
    parser.add_argument("--bert-dir", type=str, default="bert_models",
                      help="Directory for saving BERT models")
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    for dir_path in [args.results_dir, args.plots_dir, args.models_dir, args.bert_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return args

def get_dataset_specific_params(dataset_name):
    """데이터셋별 특정 파라미터 반환"""
    params = {
        'AIDS': {
            'alpha': 1.0,
            'gamma': 100.0,
            'node_theta': 0.03
        },
        'COX2': {
            'alpha': 1.0,
            'gamma': 100.0,
            'node_theta': 0.03
        },
        'DHFR': {
            'alpha': 1.0,
            'gamma': 100.0,
            'node_theta': 0.03
        },
        'Tox21_p53': {
            'alpha': 1.0,
            'gamma': 50.0,
            'node_theta': 0.05
        },
        'Tox21_HSE': {
            'alpha': 1.0,
            'gamma': 50.0,
            'node_theta': 0.05
        },
        'Tox21_MMP': {
            'alpha': 1.0,
            'gamma': 50.0,
            'node_theta': 0.05
        }
    }
    return params.get(dataset_name, {})
