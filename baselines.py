import copy
# import timeit
import math
from util import *
import pickle
import time
import sklearn
from graph_process import *
import torch
import torch.nn.parameter as Parameter
import random as rand
import numpy as np
import torch.nn as nn
import optim as optx
import neural_modules  as neux  
# from DPLP_plus import *
from sklearn.linear_model import LogisticRegression as logs


def gen_staircase_noise(Noise_test,queries, e_diff,idx_of_one):
    
    init_dim = idx_of_one
    end_dim = idx_of_one+1
    
    Pos_noise = Noise_test['pub']['pos']
    Neg_noise = Noise_test['pub']['neg']
    Noise={}
    Noise['pub']={}
    Noise['pub']['pos']={}
    Noise['pub']['neg']={}
    Noise['pub']['pos'] = Pos_noise
    Noise['pub']['neg'] = Neg_noise
    for i in range(len(Pos_noise)):
        posx =  Pos_noise[queries[i]][:,init_dim:end_dim] 
        negx =  Neg_noise[queries[i]][:,init_dim:end_dim] 
        
        pos_ones = torch.ones(posx.shape[0],posx.shape[1])
        neg_ones = torch.ones(negx.shape[0],negx.shape[1])
        
        sensPos = 1.0*torch.sign(torch.abs(posx))
        sensNeg = 1.0*torch.sign(torch.abs(negx))

        Spos =  2*torch.bernoulli(0.5 * pos_ones)-1
        Sneg =  2*torch.bernoulli(0.5 * neg_ones)-1
        
        ee=torch.FloatTensor([e_diff])
        b = math.exp(-ee)
        
        G1 = torch.distributions.geometric.Geometric((1-b) * pos_ones )
        G2 = torch.distributions.geometric.Geometric((1-b) * neg_ones )

        Gpos = G1.sample()
        Gneg = G2.sample()
        
        Upos = torch.rand(posx.shape[0],posx.shape[1])
        Uneg = torch.rand(negx.shape[0],negx.shape[1])
        
#         X ← S((1 − B)((G + γ U )D) + B((G + γ + (1 − γ )U )D)).
        bmat_pos = b*pos_ones
        bmat_neg = b*neg_ones
        
        
        gamma = -b/(1-b) + (b- 2 * (b**2) + 2*(b**4)-b**5)**0.33/(((1-b)**2)*(2**0.33) )
        gamma = min(0.999,gamma)
        Bpos = torch.bernoulli(((1-gamma)*bmat_pos) / (gamma+ (1-gamma)*bmat_pos))
        Bneg = torch.bernoulli(((1-gamma)*bmat_neg) / (gamma+ (1-gamma)*bmat_neg))

        X1  =  Spos * (1-Bpos) * (Gpos+ gamma * Upos)* sensPos +  Spos * Bpos*(Gpos+gamma+(1-gamma)*Upos)*sensPos
        X2  =  Sneg * (1-Bneg) * (Gneg+ gamma * Uneg)* sensNeg + Sneg * Bneg*(Gneg+gamma+(1-gamma)*Uneg)*sensNeg
        
        Noise['pub']['pos'][queries[i]][:,init_dim:end_dim] = X1
        Noise['pub']['neg'][queries[i]][:,init_dim:end_dim] = X2
        
    return Noise
        
class all_dp_algos:
    def __init__(self, dataset_name,qfrac,test_frac,
                 dev_frac, prot_frac,e_diff, score_def, machine,device,K=None,ifNumpy=True,wo_priv=False):
        
        stf_linear = \
            load_linear_optim_output(dataset_name,qfrac,test_frac, dev_frac, prot_frac,e_diff, score_def,machine, 
                                     noise_type='gumbel')
        
#         self.stf_umnn_only = \
#             load_only_umnn_optim_output(dataset_name,qfrac,test_frac, dev_frac, prot_frac,e_diff, score_def,machine, 
#                                      noise_type='gumbel')
        
        if wo_priv==False:
            stf_linear_umnn = \
                load_linear_umnn_optim_output(dataset_name,qfrac,test_frac, dev_frac, prot_frac,e_diff, score_def,machine, 
                                 noise_type='gumbel')
            self.stf_linear_umnn = stf_linear_umnn

        
        self.ifnumpy=ifNumpy
        self.K = None
        if K is not None:
            self.K=K
        self.specs  = stf_linear['specs']
        self.qf_test_dev_prot_dict = make_dict(qfrac,test_frac,dev_frac,prot_frac)
        self.stf_linear = stf_linear
        self.device=device 
        self.dataset_name = dataset_name
        self.score_def=score_def
        self.end_to_end=False
        self.machine=machine
        self.e_diff= e_diff
        self.auc_baseline = {}
        
        
    def lap_exp(self,noise_type):
        
        score_noise = optx.read_for_optim(self.dataset_name,self.qf_test_dev_prot_dict,self.score_def,noise_type,\
                                                           self.end_to_end,MACHINE=self.machine)
        e_diff= 2.0*self.e_diff if noise_type=='laplace' else self.e_diff
        
        specs=self.specs
        Score_test = score_noise['Score']['test']

        Noise_test = score_noise['Noise']['test']
        
        queries  = score_noise['FQ']
        power_vector = score_noise['power_vector']
        
        pv = np.array(power_vector)
        idx_of_one=np.where(pv >0.99)[0][0]
        
        nb_epoch = specs['nb_epoch']
        
        init_dim = 20
        choose_dim =  len(power_vector) 
        

        auc,auc_v= neux.auc_compute_baseline(Score_test['pub']['pos'], Score_test['pub']['neg'], \
                                                Noise_test['pub']['pos'], Noise_test['pub']['neg'], \
                                                queries, idx_of_one, e_diff,self.device,self.K,ifNumpy=self.ifnumpy) 
    
    
      
        stdd=np.std(auc_v)
        
        return auc,stdd,auc_v

    def staircase_noise(self):
        
        score_noise = optx.read_for_optim(self.dataset_name,self.qf_test_dev_prot_dict,self.score_def,'gumbel',\
                                                           self.end_to_end,MACHINE=self.machine)
        e_diff= 2.0*self.e_diff
        
        specs=self.specs
        Score_test = score_noise['Score']['test']

        Noise_test = score_noise['Noise']['test']
        
        queries  = score_noise['FQ']
        power_vector = score_noise['power_vector']
        
        pv = np.array(power_vector)
        idx_of_one=np.where(pv >0.99)[0][0]
        
        nb_epoch = specs['nb_epoch']
        
        init_dim = 20
        choose_dim =  len(power_vector)
        
        Noise_test = gen_staircase_noise(Noise_test,queries, e_diff,idx_of_one)
        

        
        auc,auc_v= neux.auc_compute_baseline(Score_test['pub']['pos'], Score_test['pub']['neg'], \
                                                Noise_test['pub']['pos'], Noise_test['pub']['neg'], \
                                                queries, idx_of_one, e_diff,self.device,self.K,ifNumpy=self.ifnumpy) 
    
      
        stdd=np.std(auc_v)
        
        return auc,stdd,auc_v

    def linear_dplp(self):
        self.score_noise = optx.read_for_optim(self.dataset_name,self.qf_test_dev_prot_dict,self.score_def,'gumbel',\
                                                           self.end_to_end,MACHINE=self.machine)
        stf_linear=self.stf_linear

        model_dplp_epoch_end = self.stf_linear['model_epoch_last_but_one'].to(self.device)
        
        
        
        specs=self.specs
        score_noise= self.score_noise
        Score_test = score_noise['Score']['test']

        Noise_test = score_noise['Noise']['test']
        
        queries  = score_noise['FQ']
        power_vector = score_noise['power_vector']
        nb_epoch = specs['nb_epoch']
        init_dim = 20
        choose_dim =  len(power_vector) 
        
        
        auc,auc_v = neux.auc_compute_from_model(model_dplp_epoch_end,Score_test['pub']['pos'], Score_test['pub']['neg'], \
                                    Noise_test['pub']['pos'], Noise_test['pub']['neg'], \
                                    queries, init_dim,choose_dim,self.e_diff,self.device,self.K,ifNumpy=self.ifnumpy)
        
        stdd=np.std(auc_v)

        
        return  auc , stdd,  auc_v
        

    def umnn_dplp(self):
        
        self.score_noise = optx.read_for_optim(self.dataset_name,self.qf_test_dev_prot_dict,self.score_def,'gumbel',\
                                                   self.end_to_end,MACHINE=self.machine)
        
        stf_linear_umnn=self.stf_linear_umnn

        model_dplp_epoch_end = self.stf_linear_umnn['model_epoch_last_but_one'].to(self.device)
        model_dplp_for_normz = self.stf_linear_umnn['model_epoch_last'].to(self.device)

        model_dplp_epoch_end.mnn.device=self.device    
        model_dplp_for_normz.mnn.device = self.device
        specs=self.specs
        score_noise= self.score_noise
        Score_test = score_noise['Score']['test']

        Noise_test = score_noise['Noise']['test']
        
        queries  = score_noise['FQ']
        power_vector = score_noise['power_vector']
        nb_epoch = specs['nb_epoch']
        init_dim = 20
        choose_dim =  len(power_vector) 

        
        auc,auc_v = neux.auc_compute_from_umnn(model_dplp_epoch_end,model_dplp_for_normz,\
                                               Score_test['pub']['pos'], Score_test['pub']['neg'], \
                                    Noise_test['pub']['pos'], Noise_test['pub']['neg'], \
                                    queries, init_dim,choose_dim,self.e_diff,self.device,self.K,ifNumpy=self.ifnumpy)
        
 
        
        stdd=np.std(auc_v)
        
        return  auc, stdd, auc_v

    
#     def only_umnn_dplp(self):
        
#         self.score_noise = optx.read_for_optim(self.dataset_name,self.qf_test_dev_prot_dict,self.score_def,'gumbel',\
#                                                    self.end_to_end,MACHINE=self.machine)
        
#         stf_umnn_only=self.stf_umnn_only
#         self.auc_end_linear_dplp = stf_umnn_only['mean_auc_test'][len(stf_umnn_only['mean_auc_test'])-1]
#         model_dplp_epoch_end = self.stf_umnn_only['model_epoch_last_but_one'].to(self.device)
#         model_dplp_for_normz = self.stf_umnn_only['model_epoch_last'].to(self.device)

#         model_dplp_epoch_end.mnn.device=self.device    
#         model_dplp_for_normz.mnn.device = self.device
#         specs=self.specs
#         score_noise= self.score_noise
#         Score_test = score_noise['Score']['test']

#         Noise_test = score_noise['Noise']['test']
        
#         queries  = score_noise['FQ']
#         power_vector = score_noise['power_vector']
#         nb_epoch = specs['nb_epoch']
        
        
#         pv = np.array(power_vector)
#         idx_of_one=np.where(pv >0.99)[0][0]
        
#         init_dim = idx_of_one
#         choose_dim = idx_of_one+1

        
#         auc,auc_v = neux.auc_compute_from_umnn_only(model_dplp_epoch_end,model_dplp_for_normz,\
#                                                Score_test['pub']['pos'], Score_test['pub']['neg'], \
#                                     Noise_test['pub']['pos'], Noise_test['pub']['neg'], \
#                                     queries, init_dim,choose_dim,self.e_diff,self.device,self.K,ifNumpy=self.ifnumpy)
        
 
        
#         stdd=np.std(auc_v)
        
#         return self.auc_end_linear_dplp, auc, stdd, auc_v

    
    def no_privacy(self,noise_type):
        
        score_noise = optx.read_for_optim(self.dataset_name,self.qf_test_dev_prot_dict,self.score_def,noise_type,\
                                                           self.end_to_end,MACHINE=self.machine)
        e_diff= 1e15
        
        specs=self.specs
        Score_test = score_noise['Score']['test']

        Noise_test = score_noise['Noise']['test']
        
        queries  = score_noise['FQ']
        power_vector = score_noise['power_vector']
        
        pv = np.array(power_vector)
        idx_of_one=np.where(pv >0.99)[0][0]
        
        nb_epoch = specs['nb_epoch']
        
        init_dim = 20
        choose_dim =  len(power_vector) 
        

        auc,auc_v= neux.auc_compute_baseline(Score_test['pub']['pos'], Score_test['pub']['neg'], \
                                                Noise_test['pub']['pos'], Noise_test['pub']['neg'], \
                                                queries, idx_of_one, e_diff,self.device,self.K,ifNumpy=self.ifnumpy) 
    
    
      
        stdd=np.std(auc_v)
        
        return auc,stdd,auc_v
    
        
        