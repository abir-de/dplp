###
import copy

from util import *
from graph_process import *
import pickle
import time
import configparser
import argparse
import DPLP_plus
import numpy as np
import torch

import os


####################################################################################################################
####################################################################################################################

def process_graph(dataset_name,  qfracv, test_fracv, dev_fracv, prot_fracv):
 
    read_file = raw_data_path + dataset_name +'.txt'

 
    qfrac = qfracv
    test_frac = test_fracv
    dev_frac = dev_fracv
    prot_frac = prot_fracv
    
    
 
    graph_data={}
    
    for qfrac in qfracv: # [0.6, 0.8]
        Graph = None
        Graph = create_graph(read_file)
        Graph.create_query_set(qfrac)
        
        
        
        for test_frac in test_fracv: #[0.20, 0.10]:
            Graph1 = copy.deepcopy(Graph)
            Graph1.split_graph_for_training(test_frac)
            
            for dev_frac in dev_fracv: #[0.30, 0.20, 0.10]:
                Graph2 = copy.deepcopy(Graph1)
                Graph2.split_graph_for_validation(dev_frac)
                
                for prot_frac in prot_fracv: #np.arange(0.05,0.50,0.05):
                    Graph3 = copy.deepcopy(Graph2)
                    Graph3.create_protected_graph(prot_frac)
                    Graph3.filter_query_private()
                    
                    
 
                    graph_data['Graph'] = Graph3
                    graph_data['name'] = dataset_name
                    graph_data['test_frac'] = test_frac
                    graph_data['qfrac'] = qfrac
                    graph_data['dev_frac'] = dev_frac
                    graph_data['prot_frac'] = prot_frac
            
                    q_str = format(qfrac, '.2f')
                    
                    test_str = format(test_frac, '.2f')
                    
                    dev_str = format(dev_frac, '.2f')
                    
                    prot_str = format(prot_frac, '.2f')


                    file_name = dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str

                    graph_file_pickle = out_path + file_name 
                    save_into_pickle(graph_data,graph_file_pickle)
                

####################################################################################################################                
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

                    
                    for score_def in ['AA', 'PA', 'JC', 'CN']:
                        
                        score1= compute_raw_scores(Graph.G_sample, Graph.protected_graph,\
                                                   Graph.queries, Graph.query_test_set, score_def)
                        sens1 = compute_sensitivity(Graph.G_sample, Graph.queries, score_def, Graph.protected_graph)
                        
                        score2= compute_raw_scores(Graph.G_true_train,  Graph.protected_graph,\
                                                   Graph.queries, Graph.query_dev_set, score_def)
                        sens2 = compute_sensitivity(Graph.G_true_train, Graph.queries, score_def, Graph.protected_graph)
                        
                        
                        score_dict['test'][score_def] = score1 
                        sens_dict['test'][score_def] = sens1
                    

                        score_dict['dev'][score_def] = score2 
                        sens_dict['dev'][score_def] = sens2


                    score_file_name = 'score_and_sens_'+ dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str

                    score_file =  score_path + score_file_name
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
                    
                    
                    
                    score_file = score_path + 'score_and_sens_'+ dataset_name + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str

                    

                    graph_data = read_from_pickle(graph_file)
                    score_sens = read_from_pickle(score_file)
 

                    for test_or_dev in ['dev', 'test']:

                        for score_def in ['AA','CN','JC','PA']:
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
                                

                                dplp_file_name = 'dplp_plus_training_input_'+ dataset_name + "_qfrac_" + q_str\
                                                                    + "_test_frac_" + test_str \
                                                                    + "_dev_frac_" + dev_str \
                                                                    + "_prot_frac_" + prot_str\
                                                                    + "_test_or_dev_" + test_or_dev\
                                                                    + "_score_or_noise_" + 'score'\
                                                                    + "_" + score_def
                                dplp_file =  dplp_path + dplp_file_name

                                if noise_def == 'gumbel':
                                    save_into_pickle(dplp_compressed,dplp_file)
                                
                                
                                dplp_compressed = {}
                                dplp_compressed['FQ'] = Dplp.filtered_queries_priv
                                dplp_compressed['power_vector']=Dplp.power_vector
                                dplp_compressed['pred'] =ppred_noise
                                
                                dplp_file_name = 'dplp_plus_training_input_'+ dataset_name + "_qfrac_" + q_str\
                                                                    + "_test_frac_" + test_str \
                                                                    + "_dev_frac_" + dev_str \
                                                                    + "_prot_frac_" + prot_str\
                                                                    + "_test_or_dev_" + test_or_dev\
                                                                    + "_score_or_noise_" + 'noise'\
                                                                    + "_" + noise_def

                                dplp_file =  dplp_path + dplp_file_name

                                save_into_pickle(dplp_compressed,dplp_file)                                
                                                               
                                 
                                 
    
parser = argparse.ArgumentParser()

parser.add_argument('--qfrac', type=float, default=1.1, help='fraction of queries.')
parser.add_argument('--test_frac', type=float, default=1.2, help='fraction of test edges.')
parser.add_argument('--dev_frac', type=float, default=1.3, help='fraction of validation edges.')
parser.add_argument('--prot_frac', type=float, default=1.1, help='fraction of protected edges.')
parser.add_argument('--dataset', type=str, required = True, help='name of dataset.')


parser.add_argument('--for_score', action='store_true', help='ScoreOnly.')
parser.add_argument('--gen_graph', action='store_true', help='Only process graph.')


parser.add_argument('--dplp_prepare', action='store_true', help='organize scores for dplp run')
 

args = parser.parse_args()

args.machine="PATHS"

config = configure(args.machine)

raw_data_path = config['raw_data_path']
out_path = config['out_graph_path']
score_path = config['score_path']
dplp_path = config['dplp_path'] + args.dataset + "_plus/"

qfracv =[0.8]
test_fracv =[0.20]
dev_fracv =[0.30]
prot_fracv = [0.1, 0.2, 0.3, 0.4, 0.45]


dataset_name = args.dataset



if args.gen_graph==True:
    process_graph(dataset_name,  qfracv, test_fracv, dev_fracv,prot_fracv)


if args.for_score == True:
    compute_scores_and_sensitivity(dataset_name,  qfracv, test_fracv, dev_fracv, prot_fracv)
    

if args.dplp_prepare == True:
    qf_test_dev_prot_dict={}
    dpath = config['dplp_path'] + args.dataset +'_plus'
    os.makedirs(dpath, exist_ok=True)
    qf_test_dev_prot_dict['qfrac'] = qfracv
    qf_test_dev_prot_dict['test_frac'] =  test_fracv
    qf_test_dev_prot_dict['dev_frac'] = dev_fracv
    qf_test_dev_prot_dict['prot_frac'] = prot_fracv
    power_vector = list(np.arange(0.1,2,0.01))
    device = "cpu"
    

    
    prepare_training_and_test_outlayer(dataset_name, qf_test_dev_prot_dict,power_vector,device)
    

