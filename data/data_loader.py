import os
import os.path as osp
import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
from .data_processing import read_graph_file

def get_ad_split_TU(dataset_name, n_cross_val):
    """TU 데이터셋에 대한 교차 검증 분할 생성"""
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
    dataset = TUDataset(path, name=dataset_name)
    
    data_list = []
    label_list = []
    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=n_cross_val, random_state=1, shuffle=True)
    splits = [(train_index, test_index) 
             for train_index, test_index in kfd.split(data_list, label_list)]

    return splits

def get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN):
    """TU 데이터셋에 대한 데이터 로더 생성"""
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
    dataset_ = TUDataset(path, name=dataset_name)
    
    # 노드 속성 및 라벨 처리
    prefix = os.path.join(path, dataset_name, 'raw', dataset_name)
    node_attrs = load_node_attributes(prefix)
    node_labels = load_node_labels(prefix)
    
    # 데이터셋 구성
    dataset = create_dataset(dataset_, node_attrs, node_labels, dataset_name)
    
    # 훈련/테스트 분할
    train_loader, test_loader, meta = create_data_loaders(
        dataset, split, dataset_AN, batch_size, test_batch_size
    )
    
    return {'train': train_loader, 'test': test_loader}, meta

def get_ad_dataset_Tox21(dataset_name, batch_size, test_batch_size, need_str_enc=False):
    """Tox21 데이터셋에 대한 데이터 로더 생성"""
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
    
    # 훈련 데이터 로드
    data_train = read_graph_file(dataset_name + '_training', path)
    
    # 데이터 전처리
    data_train, data_test = preprocess_tox21_data(data_train, dataset_name)
    
    # 데이터 로더 생성
    max_nodes = max([d.num_nodes for d in data_train + data_test])
    train_loader = DataLoader(data_train, batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(data_test, test_batch_size, shuffle=True)
    
    meta = {
        'num_feat': data_train[0].x.shape[1],
        'num_train': len(data_train),
        'max_nodes': max_nodes
    }
    
    return {'train': train_loader, 'test': test_loader}, meta

# Helper functions
def load_node_attributes(prefix):
    """노드 속성 로드"""
    filename = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
    return node_attrs

def load_node_labels(prefix):
    """노드 라벨 로드"""
    filename = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename) as f:
            for line in f:
                node_labels += [int(line.strip("\n")) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
        num_unique_node_labels = 0
    return node_labels, num_unique_node_labels