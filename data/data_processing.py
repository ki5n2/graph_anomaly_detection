import os
import re
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def node_iter(G):
    return G.nodes

def node_dict(G):
    return G.nodes

def read_graph_file(dataset_name, path):
    """그래프 파일 읽기 및 처리"""
    prefix = os.path.join(path, dataset_name, 'raw', dataset_name)
    
    # 그래프 인디케이터 읽기
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            graph_indic[i] = int(line.strip("\n"))
            i += 1

    # 노드 라벨 읽기
    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                node_labels += [int(line.strip("\n")) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
        node_labels = []
        num_unique_node_labels = 0

    # 노드 속성 읽기
    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    # 그래프 라벨 읽기
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []
    label_vals = []
    
    with open(filename_graphs) as f:
        for line in f:
            val = int(line.strip("\n"))
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {0: 0, 1: 1}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    # 인접 행렬 읽기
    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]

    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    # 그래프 생성
    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]
        
        # 노드 특성 추가
        for u in node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                node_dict(G)[u]['feat'] = node_attrs[u - 1]
        
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # 노드 인덱스 재매핑
        mapping = {n: it for it, n in enumerate(node_iter(G))}
        graphs.append(from_networkx(nx.relabel_nodes(G, mapping)))
                
    return graphs
