
from util import *
import pickle
import time
# from diffPriv import *
import DPLP_plus
import sklearn
import torch
import torch.nn.parameter as Parameter
import random as rand
import numpy as np
import torch.nn as nn
import optim as optim
torch.cuda.empty_cache()
from neural_modules  import *  
import baselines as bsl
import argparse

def utility_vs_prot(dataset_name,score_def,device,machine_name):

    qfrac=0.8
    test_frac=0.2
    dev_frac=0.3
    utility_vs_prot_edges_mat_mean={}
    utility_vs_prot_edges_mat_std={}
    

    eev=[0.05]
    
    
    for e_diff in eev:


        utility_vs_prot_edges_mat_mean[e_diff] = np.zeros((5,5))
        utility_vs_prot_edges_mat_std[e_diff] = np.zeros((5,5))

        i=-1
        for prot_frac in [0.1,0.2,0.3,0.4,0.45]:
            
            i=i+1
            expts = bsl.all_dp_algos(dataset_name,
                                     qfrac,test_frac, dev_frac, prot_frac,e_diff, 
                                     score_def, machine_name,device,K=30,ifNumpy=False)

            stair,stair_std,_ = expts.staircase_noise()
            lp,lp_std,_ = expts.lap_exp('laplace')
            expon, expon_std,_ = expts.lap_exp('gumbel')


            _,lin, lin_std,_ = expts.linear_dplp()
            _,umnn, umnn_std,_ = expts.umnn_dplp()

            utility_vs_prot_edges_mat_mean[e_diff][i,:] = np.array([umnn,lin,stair,lp,expon])
            utility_vs_prot_edges_mat_std[e_diff][i,:] = np.array([umnn_std,lin_std,stair_std,lp,expon_std])
            
    util_prot={'mean':utility_vs_prot_edges_mat_mean, 'std':utility_vs_prot_edges_mat_std}
    
    PATH='forpaper/'
    FILENAME=PATH+'util_prot_' + dataset_name + '_'+score_def
    save_into_pickle(util_prot,FILENAME)

def utility_vs_eeval(dataset_name,score_def,device,machine_name):

    qfrac=0.8
    test_frac=0.2
    dev_frac=0.3
    utility_vs_e_mat_mean={}
    utility_vs_e_mat_std={}

    protv=[0.3]
    
    
    for prot_frac in protv:
        eev=[0.05,0.15,0.25,0.35,0.45]

        utility_vs_e_mat_mean[prot_frac] = np.zeros((len(eev),5))
        utility_vs_e_mat_std[prot_frac] = np.zeros((len(eev),5))

        i=-1
        for e_diff in eev:
            i=i+1
            expts = bsl.all_dp_algos(dataset_name,
                                     qfrac,test_frac, dev_frac, prot_frac,e_diff, 
                                     score_def, machine_name,device,K=30,ifNumpy=False)

            stair,stair_std,_ = expts.staircase_noise()
            lp,lp_std,_ = expts.lap_exp('laplace')
            expon, expon_std,_ = expts.lap_exp('gumbel')


            _,lin, lin_std,_ = expts.linear_dplp()
            _,umnn, umnn_std,_ = expts.umnn_dplp()

            utility_vs_e_mat_mean[prot_frac][i,:] = np.array([umnn,lin,stair,lp,expon])
            utility_vs_e_mat_std[prot_frac][i,:] = np.array([umnn_std,lin_std,stair_std,lp,expon_std])
            
    util_e={'mean':utility_vs_e_mat_mean, 'std':utility_vs_e_mat_std}
    
    
    PATH='forpaper/'
    FILENAME=PATH+'util_e_forpaper_' + dataset_name + '_'+score_def
    save_into_pickle(util_e,FILENAME)

    
parser = argparse.ArgumentParser()

    
parser.add_argument('--dataset', type=str, required = True, help='name of dataset.')
parser.add_argument('--device', type=str, required = True, help='cpu/gpu.')


args = parser.parse_args()
args.machine="PATHS"
device=args.device
 

dataset_name=args.dataset
 


for score_def in ['AA','PA','JC','CN','GCN','Node2Vec','PRUNE','DeepWalk','LINE','Struc2Vec']: 
    utility_vs_prot(args.dataset,score_def,device,args.machine)
    utility_vs_eeval(args.dataset,score_def,device,args.machine)