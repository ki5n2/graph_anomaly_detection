import os
import time
import json
import torch
import numpy as np
from config.config import get_args
from data.data_loader import get_data_loaders_TU, get_ad_dataset_Tox21
from models.autoencoder import GraphAutoencoder
from trainers.bert_trainer import BertTrainer
from trainers.ae_trainer import AutoencoderTrainer
from utils.core import set_seed, set_device

def main():
    # 설정 로드
    args = get_args()
    
    # 시드 설정
    set_seed(args.random_seed)
    
    # 디바이스 설정
    device = set_device()
    print(f"Using device: {device}")
    
    # 현재 시간 (결과 저장용)
    current_time = time.strftime("%Y_%m_%d_%H_%M")
    
    # Cross validation 결과 저장
    all_results = []
    epoch_results = {}
    
    # 데이터셋 타입에 따른 처리
    if args.dataset_name.startswith('Tox21'):
        loaders, meta = get_ad_dataset_Tox21(
            args.dataset_name, 
            args.batch_size, 
            args.test_batch_size
        )
        n_cross_val = 1  # Tox21은 cross validation 없음
    else:
        splits = get_ad_split_TU(args.dataset_name, args.n_cross_val)
        n_cross_val = args.n_cross_val
    
    # Cross validation 실행
    for trial in range(n_cross_val):
        print(f"\nStarting fold {trial + 1}/{n_cross_val}")
        fold_start = time.time()
        
        # 데이터 로더 생성
        if not args.dataset_name.startswith('Tox21'):
            loaders, meta = get_data_loaders_TU(
                args.dataset_name,
                args.batch_size,
                args.test_batch_size,
                splits[trial],
                args.dataset_name in ['AIDS', 'NCI1', 'DHFR']
            )
        
        train_loader = loaders['train']
        test_loader = loaders['test']
        
        # 모델 생성
        model = GraphAutoencoder(
            num_features=meta['num_feat'],
            hidden_dims=args.hidden_dims,
            max_nodes=meta['max_nodes'],
            nhead_BERT=args.n_head_BERT,
            nhead=args.n_head,
            num_layers_BERT=args.n_layer_BERT,
            num_layers=args.n_layer,
            dropout_rate=args.dropout_rate
        ).to(device)
        
        # BERT 사전학습
        bert_save_path = os.path.join(
            args.assets_root,
            f'bert_{args.dataset_name}_fold{trial}_time_{current_time}.pth'
        )
        os.makedirs(os.path.dirname(bert_save_path), exist_ok=True)
        
        if os.path.exists(bert_save_path):
            print("Loading pretrained BERT...")
            model.encoder.load_state_dict(torch.load(bert_save_path))
        else:
            print("Training BERT from scratch...")
            bert_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate
            )
            bert_trainer = BertTrainer(model, bert_optimizer, device)
            
            # 마스크 토큰 재구성 훈련
            print("Stage 1-1: Mask token reconstruction training...")
            for epoch in range(args.BERT_epochs):
                loss, _ = bert_trainer.train_embedding(train_loader)
                if (epoch + 1) % args.log_interval == 0:
                    print(f'BERT Epoch {epoch + 1}: Loss = {loss:.4f}')
            
            # 엣지 재구성 훈련
            print("\nStage 1-2: Edge reconstruction training...")
            for epoch in range(args.BERT_epochs):
                loss, _ = bert_trainer.train_edge_reconstruction(train_loader)
                if (epoch + 1) % args.log_interval == 0:
                    print(f'Edge Training Epoch {epoch + 1}: Loss = {loss:.4f}')
            
            # 학습된 BERT 저장
            torch.save(model.encoder.state_dict(), bert_save_path)
        
        # Autoencoder 학습
        print("\nStage 2: Training autoencoder...")
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        trainer = AutoencoderTrainer(
            model, optimizer, device,
            args.dataset_name, current_time,
            args.alpha, args.gamma
        )
        
        best_auroc = 0
        best_epoch_metrics = None
        
        for epoch in range(args.epochs):
            # 학습
            train_loss, train_errors = trainer.train_epoch(train_loader, epoch)
            
            # 평가
            if (epoch + 1) % args.log_interval == 0:
                metrics, test_errors = trainer.evaluate(
                    test_loader, train_errors,
                    epoch, trial
                )
                
                auroc = metrics['auroc']
                print(f'Epoch {epoch + 1}: '
                      f'Train Loss = {train_loss:.4f}, '
                      f'AUROC = {auroc:.4f}, '
                      f'AUPRC = {metrics["auprc"]:.4f}')
                
                # 최고 성능 모델 저장
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_epoch_metrics = metrics
                    checkpoint_path = os.path.join(
                        args.assets_root,
                        'checkpoints',
                        args.dataset_name,
                        f'model_fold{trial}_time_{current_time}.pt'
                    )
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    trainer.save_checkpoint(
                        checkpoint_path, epoch, optimizer, metrics
                    )
                
                # 에폭 결과 저장
                if (epoch + 1) % 10 == 0:
                    if epoch + 1 not in epoch_results:
                        epoch_results[epoch + 1] = {
                            'aurocs': [], 'auprcs': [],
                            'precisions': [], 'recalls': [],
                            'f1s': []
                        }
                    
                    for key in metrics:
                        if key in ['auroc', 'auprc', 'precision', 'recall', 'f1']:
                            epoch_results[epoch + 1][f'{key}s'].append(metrics[key])
        
        # 폴드 결과 저장
        all_results.append(best_epoch_metrics)
        fold_time = time.time() - fold_start
        print(f"Fold {trial + 1} finished in {fold_time:.2f} seconds")
    
    # 최종 결과 출력 및 저장
    final_metrics = {}
    for metric in ['auroc', 'auprc', 'precision', 'recall', 'f1']:
        values = [result[metric] for result in all_results]
        final_metrics[f'mean_{metric}'] = float(np.mean(values))
        final_metrics[f'std_{metric}'] = float(np.std(values))
    
    results_path = os.path.join(
        'results',
        f'{args.dataset_name}_time_{current_time}.json'
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': final_metrics,
            'epoch_results': epoch_results,
            'config': vars(args)
        }, f, indent=4)
    
    print("\nFinal Results:")
    for metric in ['auroc', 'auprc', 'f1']:
        print(f"{metric.upper()} = "
              f"{final_metrics[f'mean_{metric}']:.4f} ± "
              f"{final_metrics[f'std_{metric}']:.4f}")

if __name__ == "__main__":
    main()
