import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from UMNN.models.UMNN import MonotonicNN, IntegrandNN
import numpy as np
class quad_linear_nn(Module):
    
    def __init__(self, in_features, out_features, bias = True, debug=False):
    
        super(quad_linear_nn, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.debug=debug
        self.bias_flag = bias 
        if bias is True:
            self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()
    def reset_parameters(self):
        
       # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.uniform_(self.weight, a=-300.0, b=300.0)
        self.weight.data = -100*torch.ones(self.in_features, self.out_features) #(torch.abs(self.weight.data))**2
        self.weight.data[0,:]=torch.zeros(1,self.out_features)
        if self.debug is True:   
            self.weight.data = torch.zeros(self.in_features, self.out_features) #(torch.abs(self.weight.data))**2
            self.weight.data[0,:]=torch.ones(1,self.out_features)


#         print(self.weight.data)
        if self.bias_flag is True:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.bias, 0)
 
 
    def forward(self, x):
        w =  torch.exp(0.01*self.weight)
        if self.debug is True:
            w = self.weight
        output = torch.mm(x,w)
        relu=nn.ReLU()
        return  output+self.bias

    

class neural_net(nn.Module):
    def __init__(self, in_d, hidden_layers,bias=True, debug=False):
        super(neural_net, self).__init__()
#         self.net = []
#         hs = [in_d] + [1]
#         for h0, h1 in zip(hs, hs[1:]):
#             self.net.extend([
#                 quad_linear_nn(h0, h1),
#        #         nn.ReLU(),
#             ])
#       #  self.net.pop()  # pop the last ReLU for the output layer
#         self.net = nn.Sequential(*self.net)
        self.qq = quad_linear_nn(in_d, 1,bias,debug)
    def forward(self, x):
        return self.qq(x)
    
class dplp_mnn(nn.Module):
    def __init__(self, in_d, hidden_layers, pre_trained, device,nb_steps=50):
        super(dplp_mnn, self).__init__()
        self.mnn = MonotonicNN(in_d, hidden_layers, nb_steps, dev=device)
        self.pre_trained = pre_trained.to(device)
    def forward(self,x):
        if x.shape[0]+x.shape[1] <=2:
            xout= torch.sum(self.pre_trained.qq.weight)
        else:    
            xout = self.pre_trained(x)
            xout =self.mnn(xout,0*xout)

        return xout
    
class dplp_mnn_no_pre_trained(nn.Module):
    def __init__(self, in_d, hidden_layers, device,nb_steps=50):
        super(dplp_mnn_no_pre_trained, self).__init__()
        self.mnn = MonotonicNN(in_d, hidden_layers, nb_steps, dev=device)
    def forward(self,x):
#         if x.shape[0]+x.shape[1] <=2:
#             xout= torch.sum(self.pre_trained.qq.weight)
#         else:    
        xout =self.mnn(x,0*x)
        return xout
#  torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

def cut_off_at_K(posx,negx,K):
    x = torch.cat((posx,negx),0)
    K=min(K,x.shape[0])
    _, indices = torch.topk(x,K, dim=0)

    indices_pos = indices[indices[:,0]<posx.shape[0],0]
    indices_neg = indices[indices[:,0]>=posx.shape[0],0]
    
    all_posx = x[indices_pos,:]
    all_negx = x[indices_neg,:]
    
    return all_posx, all_negx


def cut_off_at_K_array(posx,negx,K):
    x=np.concatenate((posx,negx),0)
    K=min(K,x.shape[0])
#     print(x.shape,K)
    indices = np.argpartition(x, -K,axis=0)[-K:] 
    
    indices_pos = indices[indices[:,0]<posx.shape[0],0]
    indices_neg = indices[indices[:,0]>=posx.shape[0],0]
    
    all_posx = x[indices_pos,:]
    all_negx = x[indices_neg,:]
    return all_posx, all_negx

def pairwise_ranking_loss(predPos, predNeg, margin):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss 

def auc_tensor(predPos, predNeg):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = expanded_1 - expanded_2
    
    ell[ell>0]=1
    ell[ell==0]=0.5
    ell[ell<0]=0
 
    pos_auc =  torch.sum(ell,dim= [0, 1])
 
    auc = pos_auc /(n_1*n_2)
    return auc

def auc_compute_feature_summed_score(Pos,Neg,queries, init_dim,end_dim_excl,device):
  #  device = "cuda" if torch.cuda.is_available() else "cpu"
    end_dim=end_dim_excl
    
    auc_v = torch.zeros((len(queries),end_dim-init_dim)).to(device)
    for i in range(len(Pos)):
        pos = Pos[queries[i]][:,init_dim:end_dim].to(device)
        neg = Neg[queries[i]][:,init_dim:end_dim].to(device)
        auc_v[i,:]=auc_tensor(pos, neg) 

    auc_tv1=torch.mean(auc_v,dim=[0]).cpu()
    return torch.max(auc_tv1), auc_tv1

def auc_compute_baseline(Pos_score,Neg_score, Pos_noise, Neg_noise, queries, idx_of_one,e_diff, device,K=None,ifNumpy=True):

    init_dim = idx_of_one
    end_dim = idx_of_one+1
    
    auc_v = torch.zeros((len(queries),1)).to(device)
    auc_v0=torch.zeros((len(queries),1)).to(device)
    for i in range(len(Pos_score)):
        posx = (1/e_diff) * Pos_noise[queries[i]][:,init_dim:end_dim].to(device) + \
                Pos_score[queries[i]][:,init_dim:end_dim].to(device)
        
        negx = (1/e_diff) * Neg_noise[queries[i]][:,init_dim:end_dim].to(device) + \
                Neg_score[queries[i]][:,init_dim:end_dim].to(device)

        
        
        
        if K is not None and ifNumpy is True: 
            posx, negx = cut_off_at_K_array(posx.cpu().detach().numpy(),negx.cpu().detach().numpy(),K)
            posx = torch.FloatTensor(posx).to(device)
            negx = torch.FloatTensor(negx).to(device)
            
            
        if K is not None and ifNumpy is False: 
            posx, negx = cut_off_at_K(posx,negx,K)
 
        
        if posx.shape[0]<1:
            pass
            
        elif negx.shape[0]<1:
            auc_v[i,:]=1
            
        else:
            auc_v[i,:] = auc_tensor(posx, negx) 

                
    del posx, negx
    auc=torch.mean(auc_v,dim=[0]).item()
    return auc, auc_v.cpu().detach().numpy()


def auc_compute_from_model(model_dplp,Pos_score,Neg_score, Pos_noise,
                           Neg_noise, queries, init_dim,end_dim_excl,e_diff, device,K=None,ifNumpy=True):
  #  device = "cuda" if torch.cuda.is_available() else "cpu"
    fq=[]
    fq1=[]
    end_dim=end_dim_excl
    
    auc_v = torch.zeros((len(queries),1)).to(device)
    auc_v0 = torch.zeros((len(queries),1)).to(device)

    for i in range(len(Pos_score)):
        pos = (1/e_diff) * Pos_noise[queries[i]][:,init_dim:end_dim].to(device) + \
                Pos_score[queries[i]][:,init_dim:end_dim].to(device)
        posx = model_dplp(pos)
        
        neg = (1/e_diff) * Neg_noise[queries[i]][:,init_dim:end_dim].to(device) + \
                Neg_score[queries[i]][:,init_dim:end_dim].to(device)
        
        negx = model_dplp(neg)
        
        if K is not None and ifNumpy is True: 
            posx, negx = cut_off_at_K_array(posx.cpu().detach().numpy(),negx.cpu().detach().numpy(),K)
            posx = torch.FloatTensor(posx).to(device)
            negx = torch.FloatTensor(negx).to(device)
            
            
        if K is not None and ifNumpy is False: 
            posx, negx = cut_off_at_K(posx,negx,K)

        if posx.shape[0]<1:
            pass
    
        elif negx.shape[0]<1:
            auc_v[i,:]=1
            
        else:
            auc_v[i,:] = auc_tensor(posx, negx) 

    del posx, negx, pos, neg

    auc=torch.mean(auc_v,dim=[0]).item()
    return  auc,auc_v.cpu().detach().numpy()

def auc_compute_from_umnn(model_dplp,model_dplp_for_normz,Pos_score,Neg_score, Pos_noise,
                           Neg_noise, queries, init_dim,end_dim_excl,e_diff, device,K=None,ifNumpy=True):
  #  device = "cuda" if torch.cuda.is_available() else "cpu"
    fq=[]
    fq1=[]
    end_dim=end_dim_excl
    
    
    auc_v = torch.zeros((len(queries),1)).to(device)
    auc_v0 = torch.zeros((len(queries),1)).to(device)
    
    for i in range(len(queries)):
        test_pos = {}
        test_neg = {}
        test_pos['score'] = Pos_score[queries[i]][:,init_dim:end_dim].to(device)
        test_pos['noise'] = Pos_noise[queries[i]][:,init_dim:end_dim].to(device)
        
        test_neg['score'] = Neg_score[queries[i]][:,init_dim:end_dim].to(device)
        test_neg['noise'] = Neg_noise[queries[i]][:,init_dim:end_dim].to(device)
        
        
        fmax = torch.ones(1,1).to(device)
        xmax = model_dplp_for_normz(fmax)
        
        posx = model_dplp_for_normz(test_pos['score'].to(device))\
                                   + (xmax/e_diff) * test_pos['noise'][:,1:2].to(device)


        negx = model_dplp_for_normz(test_neg['score'].to(device))\
                                            + (xmax/e_diff) * test_neg['noise'][:,1:2].to(device)
        
        
        
        
        if K is not None and ifNumpy is True: 
            posx, negx = cut_off_at_K_array(posx.cpu().detach().numpy(),negx.cpu().detach().numpy(),K)
            posx = torch.FloatTensor(posx).to(device)
            negx = torch.FloatTensor(negx).to(device)
            
            
        if K is not None and ifNumpy is False: 
            posx, negx = cut_off_at_K(posx,negx,K)

        if posx.shape[0]<1:
            pass
    
        elif negx.shape[0]<1:
            auc_v[i,:]=1
            
        else:
            auc_v[i,:] = auc_tensor(posx, negx) 

    del posx, negx

    auc=torch.mean(auc_v,dim=[0]).item()
    return  auc,auc_v.cpu().detach().numpy()

def auc_compute_from_umnn_only(model_dplp,model_dplp_for_normz,Pos_score,Neg_score, Pos_noise,
                           Neg_noise, queries, init_dim,end_dim_excl,e_diff, device,K=None,ifNumpy=True):
  #  device = "cuda" if torch.cuda.is_available() else "cpu"
    fq=[]
    fq1=[]
    end_dim=end_dim_excl
    
    
    auc_v = torch.zeros((len(queries),1)).to(device)
    auc_v0 = torch.zeros((len(queries),1)).to(device)
    
    for i in range(len(queries)):
        test_pos = {}
        test_neg = {}
        test_pos['score'] = Pos_score[queries[i]][:,init_dim:end_dim].to(device)
        test_pos['noise'] = Pos_noise[queries[i]][:,init_dim:end_dim].to(device)
        
        test_neg['score'] = Neg_score[queries[i]][:,init_dim:end_dim].to(device)
        test_neg['noise'] = Neg_noise[queries[i]][:,init_dim:end_dim].to(device)
        
        
        fmax = torch.ones(1,1).to(device)
        xmax = model_dplp_for_normz(fmax)
        
        posx = model_dplp_for_normz(test_pos['score'].to(device))\
                                   + (xmax/e_diff) * test_pos['noise'][:,0:1].to(device)


        negx = model_dplp_for_normz(test_neg['score'].to(device))\
                                            + (xmax/e_diff) * test_neg['noise'][:,0:1].to(device)
        
        
        
        
        if K is not None and ifNumpy is True: 
            posx, negx = cut_off_at_K_array(posx.cpu().detach().numpy(),negx.cpu().detach().numpy(),K)
            posx = torch.FloatTensor(posx).to(device)
            negx = torch.FloatTensor(negx).to(device)
            
            
        if K is not None and ifNumpy is False: 
            posx, negx = cut_off_at_K(posx,negx,K)

        if posx.shape[0]<1:
            pass
    
        elif negx.shape[0]<1:
            auc_v[i,:]=1
            
        else:
            auc_v[i,:] = auc_tensor(posx, negx) 

    del posx, negx

    auc=torch.mean(auc_v,dim=[0]).item()
    return  auc,auc_v.cpu().detach().numpy()