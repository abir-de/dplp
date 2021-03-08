###
import copy
# import timeit
from util import *
from graph_process import *
import pickle
import time
import configparser
import argparse
import DPLP_plus
import numpy as np
import torch
#start_time=time.time()
import os

import all_embeddings as deepx

####################################################################################################################           
def prepare_score_matrix(dataset_name, loop_specs):
    
   

    qfracv = loop_specs['qfracv']
    test_fracv = loop_specs['test_fracv']
    dev_fracv = loop_specs['dev_fracv']
    prot_fracv = loop_specs['prot_fracv']
#     eev = loop_specs['eev']
#     noise_vec = loop_specs['noise_v']
    machine_name = loop_specs['machine']
    
    for qfrac in qfracv: # [0.6, 0.8]
        for test_frac in test_fracv: #[0.20, 0.10]:
            for dev_frac in dev_fracv: #[0.30, 0.20, 0.10]:
                
                
                qf_test_dev_prot_dict={}
                qf_test_dev_prot_dict['qfrac'] = qfrac
                qf_test_dev_prot_dict['test_frac'] =  test_frac
                qf_test_dev_prot_dict['dev_frac'] = dev_frac
                qf_test_dev_prot_dict['prot_frac'] = 0.3

                graph_data = read_graph_file(dataset_name, qf_test_dev_prot_dict,machine_name)
                

                for score_def in ['GCN','Node2Vec','PRUNE','DeepWalk','LINE','Struc2Vec']:
                    specs=loop_specs[score_def]
                    specs['qf_test_dev_prot_dict'] = qf_test_dev_prot_dict
                    specs['machine'] = machine_name
                    specs['device'] = loop_specs['device']
                    
                    dx = deepx.Deep_embedding(score_def,dataset_name,graph_data,specs)
                    dx.train()
                    
                    for prot_frac in prot_fracv: #np.arange(0.05,0.50,0.05):
                        
                        qf_test_dev_prot_dict['prot_frac'] = prot_frac
                        specs['qf_test_dev_prot_dict'] = qf_test_dev_prot_dict
                        dx.specs=specs
                        dx.save()

                            


                          #####################################################################################################################
def compute_scores_and_sensitivity(dataset_name,  qfracv, test_fracv, dev_fracv, prot_fracv):
    
   

    qfrac = qfracv
    test_frac = test_fracv
    dev_frac = dev_fracv
    prot_frac = prot_fracv
    
    
 
    
    for qfrac in qfracv: # [0.6, 0.8]
        for test_frac in test_fracv: #[0.20, 0.10]:
            for dev_frac in dev_fracv: #[0.30, 0.20, 0.10]:
                for prot_frac in prot_fracv: #np.arange(0.05,0.50,0.05):
                    #start_time = time.time()
            
                    q_str = format(qfrac, '.2f')
                    
                    test_str = format(test_frac, '.2f')
                    
                    dev_str = format(dev_frac, '.2f')
                    
                    prot_str = format(prot_frac, '.2f')


                    file_name = out_path + dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str
                    

                    graph_data_read = read_from_pickle(file_name)
                    Graph = graph_data_read['Graph']
                    
                    
                    score_dict={'test': {}, 'dev': {}}
                    sens_dict={'test': {}, 'dev': {}}

                    
                    for score_def in  ['GCN','Node2Vec','PRUNE','DeepWalk','LINE','Struc2Vec']:
                        
                        
                        score_mat=load_embeddings(dataset_name,qfrac,\
                                                        test_frac, dev_frac, prot_frac, score_def,deep_score_path)
                        
                        
                        
                        score1= compute_raw_scores(Graph.G_sample, Graph.protected_graph,\
                                                   Graph.queries, Graph.query_test_set, score_def,score_mat)
                        sens1 = compute_sensitivity(Graph.G_sample, Graph.queries, score_def, Graph.protected_graph)
                        
                        score2= compute_raw_scores(Graph.G_true_train,  Graph.protected_graph,\
                                                   Graph.queries, Graph.query_dev_set, score_def,score_mat)
                        sens2 = compute_sensitivity(Graph.G_true_train, Graph.queries, score_def, Graph.protected_graph)
                        
                        
                        score_dict['test'][score_def] = score1 
                        sens_dict['test'][score_def] = sens1
                    

                        score_dict['dev'][score_def] = score2 
                        sens_dict['dev'][score_def] = sens2


                    score_file_name = 'deep_score_and_sens_'+ dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str

                    score_file =  deep_score_path + score_file_name
                    #print((time.time()-start_time) / 60)
                    raw_score_sens_data={}
                    raw_score_sens_data['score_data'] = score_dict
                    raw_score_sens_data['sens_data'] = sens_dict
                    save_into_pickle(raw_score_sens_data,score_file)


####################################################################################################################
####################################################################################################################
####################################################################################################################

### any saving from now on would use torch.save##

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
def prepare_training_and_test_outlayer(dataset_name, qf_test_dev_prot_dict, power_vector,device):
    
    qfracv = qf_test_dev_prot_dict['qfrac']
    test_fracv = qf_test_dev_prot_dict['test_frac']
    dev_fracv =  qf_test_dev_prot_dict['dev_frac']
    prot_fracv =  qf_test_dev_prot_dict['prot_frac']
 
    print("STARTED..")
    
    for qfrac in qfracv: # [0.6, 0.8]
        for test_frac in test_fracv: #[0.20, 0.10]:
            for dev_frac in dev_fracv: #[0.30, 0.20, 0.10]:
                for prot_frac in prot_fracv: #np.arange(0.05,0.50,0.05):
                    #start_time = time.time()
            
                    q_str = format(qfrac, '.2f')
                    
                    test_str = format(test_frac, '.2f')
                    
                    dev_str = format(dev_frac, '.2f')
                    
                    prot_str = format(prot_frac, '.2f')


                    graph_file = out_path + dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str
                    
                    
                    
                    score_file = deep_score_path + 'deep_score_and_sens_'+ dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str

                    

                    graph_data = read_from_pickle(graph_file)
                    score_sens = read_from_pickle(score_file)
 

                    for test_or_dev in ['dev', 'test']:

                        for score_def in  ['GCN','Node2Vec','PRUNE','DeepWalk','LINE','Struc2Vec']:
                            Dplp=None
                            Dplp = DPLP_plus.DPLP(graph_data,score_sens,power_vector)
                            Dplp.score_scaling_and_noise_gen(test_or_dev,score_def)
                            Dplp.store_noise_score(test_or_dev,score_def,'--',device,'score')
                            ppred_score =  Dplp.data_for_training_test
                            
                            
                            for noise_def in ['gumbel', 'laplace']:

                                Dplp.store_noise_score(test_or_dev,'--',noise_def,device,'noise')
                                ppred_noise =  Dplp.data_for_training_test
                                
                                
                                dplp_compressed = {}
                                dplp_compressed['FQ'] = Dplp.filtered_queries_priv
                                dplp_compressed['power_vector']=Dplp.power_vector
                                dplp_compressed['pred'] =ppred_score
                                

                                dplp_file_name = 'deep_dplp_plus_training_input_'+ dataset_name + "_qfrac_" + q_str\
                                                                    + "_test_frac_" + test_str \
                                                                    + "_dev_frac_" + dev_str \
                                                                    + "_prot_frac_" + prot_str\
                                                                    + "_test_or_dev_" + test_or_dev\
                                                                    + "_score_or_noise_" + 'score'\
                                                                    + "_" + score_def
                                dplp_file =  embd_dplp_path + dplp_file_name

                                if noise_def == 'gumbel':
                                    save_into_pickle(dplp_compressed,dplp_file)
                                
                                
                                dplp_compressed = {}
                                dplp_compressed['FQ'] = Dplp.filtered_queries_priv
                                dplp_compressed['power_vector']=Dplp.power_vector
                                dplp_compressed['pred'] =ppred_noise
                                
                                dplp_file_name = 'deep_dplp_plus_training_input_'+ dataset_name + "_qfrac_" + q_str\
                                                                    + "_test_frac_" + test_str \
                                                                    + "_dev_frac_" + dev_str \
                                                                    + "_prot_frac_" + prot_str\
                                                                    + "_test_or_dev_" + test_or_dev\
                                                                    + "_score_or_noise_" + 'noise'\
                                                                    + "_" + noise_def

                                dplp_file =  embd_dplp_path + dplp_file_name

                                save_into_pickle(dplp_compressed,dplp_file)                                
                                                               
                                 
                                
    
parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str, required = True, help='name of dataset.')


parser.add_argument('--for_score', action='store_true', help='ScoreOnly.')
parser.add_argument('--device',type=str, required = True, help="cpu/gpu")


parser.add_argument('--dplp_prepare', action='store_true', help='organize scores')
parser.add_argument('--score_matrix_deep', action='store_true', help='compute_scores')

args = parser.parse_args()
args.machine="PATHS"

config = configure(args.machine)

raw_data_path = config['raw_data_path']
out_path = config['out_graph_path']
deep_score_path = config['deep_score_path']
embd_dplp_path = config['embd_dplp_path'] + args.dataset + "_plus/"


device=args.device
qfracv =[0.8]
test_fracv =[0.20]
dev_fracv =[0.30]
prot_fracv =  [0.1, 0.2, 0.3, 0.4, 0.45]



    
dataset_name = args.dataset

loop_specs={}

###PRUNE###########
specs={}
specs['lamb']=0.01
specs['dimensions']=80
specs['lr']=1e-4             
specs['nb_epochs']=50
specs['gpu']=0.1
specs['batch_size']=300

loop_specs['PRUNE'] = specs
########################


#########N2V##########
specs={}
specs['p'] = 1
specs['q'] = 1
specs['walk_length'] = 80
specs['num_walks'] = 10
specs['dimensions'] = 80
specs['window_size'] =10
specs['workers']=8
specs['iters']= 1
loop_specs['Node2Vec'] = specs
#######################


#########GCN##########  
specs={}
specs['hidden']=20
specs['nout']=80
specs['dimensions']=specs['nout']
specs['dropout']=0.3
specs['lr'] = 0.01
specs['wt_decay'] = 5e-4
specs['nb_epochs'] = 80

loop_specs['GCN'] = specs
########################



specs={}
specs['dimensions']=80
specs['nb_epochs'] = 80

loop_specs['LINE']=specs
loop_specs['DeepWalk']= specs
loop_specs['Struc2Vec']= specs

loop_specs['qfracv'] = qfracv
loop_specs['test_fracv'] = test_fracv
loop_specs['dev_fracv'] = dev_fracv
loop_specs['prot_fracv'] = prot_fracv
loop_specs['device'] = device   
loop_specs['machine'] = args.machine   


if args.score_matrix_deep == True:
    os.makedirs(deep_score_path, exist_ok=True)

    prepare_score_matrix(dataset_name, loop_specs)

if args.for_score == True:

    compute_scores_and_sensitivity(dataset_name,  qfracv, test_fracv, dev_fracv, prot_fracv)
    
    
    
    
qf_test_dev_prot_dict={}
if args.dplp_prepare == True:
    
    os.makedirs(embd_dplp_path, exist_ok=True)
    qf_test_dev_prot_dict['qfrac'] = qfracv
    qf_test_dev_prot_dict['test_frac'] =  test_fracv
    qf_test_dev_prot_dict['dev_frac'] = dev_fracv
    qf_test_dev_prot_dict['prot_frac'] = prot_fracv
    power_vector = list(np.arange(0.1,2,0.01))

    prepare_training_and_test_outlayer(dataset_name, qf_test_dev_prot_dict,power_vector,device)
    
