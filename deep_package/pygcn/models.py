import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
def laplacian(x,adj):
    n_1 = x.shape[0]
    dim = x.shape[1]

    expanded_1 = x.unsqueeze(1).expand(n_1, n_1, dim)
    expanded_2 = x.unsqueeze(0).expand(n_1, n_1, dim)
    ell = (expanded_2 - expanded_1) ** 2
    vv = torch.sum(ell,dim= [2])
    mm = torch.mul(adj,vv)
    sum_loss =  torch.sum(mm, dim=[0,1])
    
    return sum_loss, vv,ell