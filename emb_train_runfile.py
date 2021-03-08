###
import copy
# import timeit
from util import *
import pickle
import time
import configparser
import argparse
import DPLP_plus
import numpy as np
import torch
import optim
#start_time=time.time()
import os


####################################################################################################################
####################################################################################################################

def train_compute(dataset_name, score_noise,specs):
     
    train_dplp = None
    
    train_dplp = optim.optim(dataset_name,score_noise, specs)
    train_dplp.gen_logfile_name(specs['qf_test_dev_prot_dict'])
    train_dplp.train()
    if specs['if_save'] is True:
        train_dplp.save()
    
    
def train_compute_umnn(dataset_name, score_noise,specs):
     
    train_dplp = None
    
    train_dplp = optim.optim(dataset_name,score_noise, specs)
    train_dplp.gen_logfile_name(specs['qf_test_dev_prot_dict'],umnn=True)
    train_dplp.prepare_for_umnn()
    train_dplp.train_umnn()
    if specs['if_save'] is True:
        train_dplp.save_umnn()
                                

####################################################################################################################                
def train_over_the_loop(dataset_name, loop_specs):
    
   

    qfracv = loop_specs['qfracv']
    test_fracv = loop_specs['test_fracv']
    dev_fracv = loop_specs['dev_fracv']
    prot_fracv = loop_specs['prot_fracv']
    eev = loop_specs['eev']
    noise_vec = loop_specs['noise_v']
 
    
    for qfrac in qfracv: # [0.6, 0.8]
        for test_frac in test_fracv: #[0.20, 0.10]:
            for dev_frac in dev_fracv: #[0.30, 0.20, 0.10]:
                for prot_frac in prot_fracv: #np.arange(0.05,0.50,0.05):
                    for score_def in ['GCN','Node2Vec']:
                                    #['GCN','Node2Vec']:#,'PRUNE','DeepWalk','LINE','Struc2Vec']:
                        for noise_type in noise_vec:
                            
                            qf_test_dev_prot_dict={}

                            qf_test_dev_prot_dict['qfrac'] = qfrac
                            qf_test_dev_prot_dict['test_frac'] =  test_frac
                            qf_test_dev_prot_dict['dev_frac'] = dev_frac
                            qf_test_dev_prot_dict['prot_frac'] = prot_frac
                            end_to_end = loop_specs['end_to_end']
                            machine_name = loop_specs['machine']
                            
                            
                            score_noise = optim.read_for_optim(dataset_name,qf_test_dev_prot_dict,\
                                                               score_def,noise_type,\
                                                               end_to_end,MACHINE=machine_name)

                            for e_diff in eev:
                                specs=loop_specs                            
                                specs['e'] = e_diff 
                                specs['score_def']= score_def
                                specs['noise']=noise_type
                                specs['prot_frac'] = prot_frac 
                                specs['qf_test_dev_prot_dict'] = qf_test_dev_prot_dict
                    
                                if specs['umnn'] is False:
#                                     pass
                                    train_compute(dataset_name, score_noise,specs)
                                else:
                                    train_compute_umnn(dataset_name, score_noise,specs)
####################################################################################################################
# 
                                 
    
parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str, required = True, help='name of dataset.')


parser.add_argument('--device',type=str, required = True, help="cpu/gpu")

parser.add_argument('--umnn', action='store_true')




args = parser.parse_args()

args.machine="PATHS"
config = configure(args.machine)
dataset_name = args.dataset
device = args.device


loop_specs={}

loop_specs['sub_batch']=2   
loop_specs['init_dim']=20
loop_specs['choose_dim']=None
loop_specs['nb_epoch']=50
loop_specs['hidden']=1
loop_specs['device'] = device   
loop_specs['lr']=0.1    
loop_specs['weight_decay']=1e-5
loop_specs['margin'] =0.1

loop_specs['if_debug'] = False   

loop_specs['qfracv'] = [0.8]
loop_specs['test_fracv'] = [0.2]
loop_specs['dev_fracv'] = [0.3]
# loop_specs['prot_fracv'] = [0.1,0.3]
loop_specs['prot_fracv'] = [0.1,0.2,0.3, 0.4,0.45]

# loop_specs['prot_fracv'] = [0.3]

loop_specs['eev'] = [0.05,0.15,0.25,0.35,0.45] 
## the above the array for e_diff/2 ---- it adjusts the factor in the denominator of exp. mechanism. So e_diff=0.1
## corresponds to 0.05 in the above list.

loop_specs['prot_fracv'] = [0.3]
loop_specs['eev'] = [0.05]

loop_specs['machine'] = args.machine
loop_specs['end_to_end'] = False
loop_specs['noise_v']=['gumbel']
loop_specs['umnn']=args.umnn
loop_specs['pre_trained_flag']=True

if loop_specs['umnn'] is True:
    loop_specs['nb_epoch']=10

loop_specs['if_save']=True
ttpath = config['deep_train_path'] + args.dataset
os.makedirs(ttpath, exist_ok=True)

train_over_the_loop(dataset_name, loop_specs)
