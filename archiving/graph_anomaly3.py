#%%
'''IMPORTS'''
import os
import time
import wandb
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import init
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import DataLoader, Dataset, Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, GAE, InnerProductDecoder
from torch_geometric.utils import add_self_loops, k_hop_subgraph, to_dense_adj, subgraph, to_undirected, to_networkx

from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

from module.loss import *

from util import set_seed, set_device, add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon, visualize, EarlyStopping

from load_data import *
from GraphBuild import *


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset')

parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--n-test-anomaly", type=int, default=400)

parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.1)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--dataset-AN", action="store_false")
parser.add_argument("--pretrained", action="store_false")

parser.add_argument('--max-nodes', type=int, default=0)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
'''OPTIONS'''
data_root: str = args.data_root
assets_root: str = args.assets_root
dataset_name: str = args.dataset_name

num_epochs: int = args.num_epochs
patience: list = args.patience
batch_size: int = args.batch_size
random_seed: int = args.random_seed
n_cross_val: int = args.n_cross_val
hidden_dims: list = args.hidden_dims
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
learning_rate: float = args.learning_rate

dataset_AN: bool = args.dataset_AN
pretrained: bool = args.pretrained

max_nodes: int = args.max_nodes

set_seed(random_seed)

device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#%%
'''MODEL 1'''
def gen_ran_output(x, adj, model, noise_model):
    for (adv_name, adv_param), (name, param) in zip(noise_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + 1.0 * torch.normal(0,torch.ones_like(param.data)*param.data.std()).cuda()     
    z_prime, z_prime_g = noise_model(x, adj)
    return z_prime, z_prime_g


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = True
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):

        y = torch.matmul(adj, x)

        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias

        return y


class Encoder(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout, batch):
        super(Encoder, self).__init__()
        self.gc1 = nn.Linear(feat_size, hiddendim, bias=False)
        self.gc4 = nn.Linear(hiddendim, outputdim, bias=False)
        self.proj_head = nn.Sequential(nn.Linear(outputdim, outputdim), nn.ReLU(inplace=True), nn.Linear(outputdim, outputdim))
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)
        self.batch=batch

    def forward(self, x, adj):
        z = self.leaky_relu(self.gc1(torch.matmul(adj, x)))
        z = self.dropout(z)
        z = self.gc4(torch.matmul(adj, z))
        z_g, _ = torch.max(z, dim=1)
        #out = global_add_pool(x,self.batch)
        z_g = self.proj_head(z_g)
        
        return z, z_g


class Attr_Decoder(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout):
        super(Attr_Decoder, self).__init__()

        self.gc1 = nn.Linear(outputdim, hiddendim, bias=False)
        self.gc4 = nn.Linear(hiddendim, feat_size, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, adj):
        x_ = self.leaky_relu(self.gc1(torch.matmul(adj, z)))
        x_ = self.dropout(x_)
        x_ = self.gc4(torch.matmul(adj, x_))
        
        return x_


class Stru_Decoder(nn.Module):
    def __init__(self, feat_size, outputdim, dropout):
        super(Stru_Decoder, self).__init__()

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z, adj):
        adj_ = z.permute(0, 2, 1)
        adj_ = torch.matmul(z, adj_)
        adj_ = self.sigmoid(adj_)
        return adj_


class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout, batch):
        super(GRAPH_AUTOENCODER, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hiddendim, outputdim, dropout, batch)
        self.attr_decoder = Attr_Decoder(feat_size, hiddendim, outputdim, dropout)
        self.struct_decoder = Stru_Decoder(feat_size, outputdim, dropout)
    
    def forward(self, z, adj):
        x_ = self.attr_decoder(z, adj)
        adj_ = self.struct_decoder(z, adj)
        z_tilde, z_tilde_g = self.shared_encoder(x_, adj_)

        return x_, adj_, z_tilde, z_tilde_g
    
    
#%%
def train(data_train_loader, data_test_loader, model, model_encoder):    
    optimizerG = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs=[]
    node_Feat=[]
    graph_Feat=[]
    
    max_AUC=0
    auroc_final = 0
    
    for epoch in range(num_epochs):
        total_time = 0
        total_lossG = 0.0
        model.train()
        for batch_idx, data in enumerate(data_train_loader):           
            begin_time = time.time()
            x = Variable(data['feats'].float(), requires_grad=False).cuda()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            adj_label = Variable(data['adj_label'].float(), requires_grad=False).cuda()

            z, z_g = model.shared_encoder(x, adj)
            z_prime, z_prime_g = gen_ran_output(x, adj, model.shared_encoder, noise_encoder)
            x_, adj_, z_tilde, z_tilde_g = model(z, adj)

            err_g_con_s, err_g_con_x = loss_func(adj_label, adj_, x, x_)

            node_loss = torch.mean(F.mse_loss(z, z_tilde, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            graph_loss = F.mse_loss(z_g, z_tilde_g, reduction='none').mean(dim=1).mean(dim=0)
            err_g_enc = loss_cal(z_prime_g, z_g)

            lossG = err_g_con_s + err_g_con_x + node_loss + graph_loss + err_g_enc
            
            optimizerG.zero_grad()
            
            lossG.backward()
          
            optimizerG.step()
          
            total_lossG += lossG
            
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        if (epoch+1)%10 == 0 and epoch > 0:
            epochs.append(epoch)
            model.eval()   
            loss = []
            y=[]

            for batch_idx, data in enumerate(data_test_loader):
               x = Variable(data['feats'].float(), requires_grad=False).cuda()
               adj = Variable(data['adj'].float(), requires_grad=False).cuda()

               z, z_g = model.shared_encoder(x, adj)
         
               x_, adj_, z_tilde, z_tilde_g = model(z, adj)
               
               loss_node = torch.mean(F.mse_loss(z, z_tilde, reduction='none'), dim=2).mean(dim=1).mean(dim=0)

               loss_graph = F.mse_loss(z_g, z_tilde_g, reduction='none').mean(dim=1)
            
               loss_ = loss_node + loss_graph
            
               loss_ = np.array(loss_.cpu().detach())
               
               loss.append(loss_)
               
               if data['label'] == 0:
                   y.append(1)
               else:
                   y.append(0) 
                              
            label_test = []
            
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)
            
            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            
            print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            
            if test_roc_ab > max_AUC:
                max_AUC = test_roc_ab
                
        auroc_final = max_AUC
        
    return auroc_final


#%%
'''MODEL 2'''
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y
    
    
class Encoder_(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, args=None):
        super(Encoder_, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1
        self.proj_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True), nn.Linear(embedding_dim, embedding_dim))
        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # conv
        z = self.conv_first(x, adj)
        z = self.act(z)
        if self.bn:
            z = self.apply_bn(z)
        z_g_all = []
        z_g, _ = torch.max(z, dim=1)
        z_g_all.append(z_g)
        for i in range(self.num_layers-2):
            z = self.conv_block[i](z, adj)
            z = self.act(z)
            if self.bn:
                z = self.apply_bn(z)
            z_g, _ = torch.max(z, dim=1)
            z_g_all.append(z_g)
            if self.num_aggs == 2:
                z_g = torch.sum(z, dim=1)
                z_g_all.append(z_g)
        z = self.conv_last(z, adj)
        z_g, _ = torch.max(z, dim=1)
        z_g_all.append(z_g)
        if self.num_aggs == 2:
            z_g = torch.sum(z, dim=1)
            z_g_all.append(z_g)
        if self.concat:
            z_g_output = torch.cat(z_g_all, dim=1)
        else:
            z_g_output = z_g
        z_g = self.proj_head(z_g)
        return z, z_g

               
class Att_Decoder_(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.1, args=None):
        super(Att_Decoder_, self).__init__()
        
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim


        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, z, adj, batch_num_nodes=None, **kwargs):
        # conv
        z = self.conv_first(z, adj)
        z = self.act(z)
        if self.bn:
            z = self.apply_bn(z)
        z_g_all = []
        z_g, _ = torch.max(z, dim=1)
        z_g_all.append(z_g)
        for i in range(self.num_layers-2):
            z = self.conv_block[i](z, adj)
            z = self.act(z)
            if self.bn:
                z = self.apply_bn(z)
            z_g, _ = torch.max(z, dim=1)
            z_g_all.append(z_g)
            if self.num_aggs == 2:
                z_g = torch.sum(z, dim=1)
                z_g_all.append(z_g)
        x_ = self.conv_last(z, adj)
        z_g, _ = torch.max(x_, dim=1)
        z_g_all.append(z_g)
        if self.num_aggs == 2:
            z_g = torch.sum(x_, dim=1)
            z_g_all.append(z_g)
        if self.concat:
            z_g_output = torch.cat(z_g_all, dim=1)
        else:
            z_g_output = z_g
        return x_

    
class Stru_Decoder_(nn.Module):
    def __init__(self, dropout):
        super(Stru_Decoder_, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, z, adj):
        z_ = z.permute(0, 2, 1)

        z = torch.matmul(z, z_) 
        adj_ = self.sigmoid(z)
        
        return adj_
    
    
class GRAPH_AUTOENCODER_(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, args=None):
        super(GRAPH_AUTOENCODER_, self).__init__()
        
        self.shared_encoder = Encoder_(input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, args=None)
        self.attr_decoder = Att_Decoder_(embedding_dim, hidden_dim, input_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.1, args=None)
        self.struct_decoder = Stru_Decoder_(dropout)
    
    def forward(self, x, adj):
        x_ = self.attr_decoder(x, adj)
        adj_ = self.struct_decoder(x, adj)
        z_tilde, z_tilde_g = self.shared_encoder(x_, adj_)

        return x_fake, s_fake, z_tilde, z_tilde_g
    
    
# %%

if __name__ == '__main__':
    DS = dataset_name
    set_seed(random_seed)
    
    graphs = read_graphfile(data_root, dataset_name, max_nodes=max_nodes)  
    datanum = len(graphs)
    
    if max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = max_nodes
        
    print(datanum)
    
    graphs_label = [graph.graph['label'] for graph in graphs]
    a=0
    b=0
    
    for graph in graphs:
        if graph.graph['label'] == 0:
            a=a+1
        else:
            b=b+1
    
    print(a,b,'!!!')

    kfd=StratifiedKFold(n_splits=n_cross_val, random_state=random_seed, shuffle=True)
    result_auc=[]

    for k, (train_index, test_index) in enumerate(kfd.split(graphs, graphs_label)):
        graphs_train_ = [graphs[i] for i in train_index]
        graphs_test = [graphs[i] for i in test_index]
        
        graphs_train = []
        for graph in graphs_train_:
            if graph.graph['label'] != 0:
                graphs_train.append(graph)
        # 즉, 그래프 라벨이 1인 것들만을 훈련데이터로 사용했음, 즉 그래프 라벨이 1인 것을 정상으로 보고 그래프 라벨이 "-1->0"인 것을 이상으로 보았음

        num_train = len(graphs_train)
        num_test = len(graphs_test)
        print(num_train, num_test)
            
        dataset_sampler_train = GraphBuild(graphs_train, features='default', normalize=False, max_num_nodes=max_nodes_num)
            
        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
                                                        shuffle=True,
                                                        batch_size=batch_size)
        
        dataset_sampler_test = GraphBuild(graphs_test, features='default', normalize=False, max_num_nodes=max_nodes_num)
        
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                    shuffle=False,
                                                    batch_size=1)

        num_features = dataset_sampler_train.feat_dim
        model = GRAPH_AUTOENCODER(num_features, hidden_dims[0], hidden_dims[1], 0.1, batch_size).cuda()
        noise_encoder= Encoder(num_features, hidden_dims[0], hidden_dims[1], 0.1, batch_size).cuda()
        
        result = train(data_train_loader, data_test_loader, model, noise_encoder)     
        result_auc.append(result)

    result_auc = np.array(result_auc)    
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))
    
    
# %%
