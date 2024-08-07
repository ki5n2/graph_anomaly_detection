import torch
import torch.nn.functional as F


def Triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
    loss = F.relu(distance_positive - distance_negative + margin)
    return loss


def loss_cal(z_i, z_j, t=0.2):
    batch_size, _ = z_i.size()
    z_i_abs = z_i.norm(dim=1)
    z_j_abs = z_j.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', z_i, z_j) / torch.einsum('i,j->ij', z_i_abs, z_j_abs)
    sim_matrix = torch.exp(sim_matrix / t)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss
