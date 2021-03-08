import copy
from util import *
import pickle
import time
import DPLP_plus
import sklearn
from graph_process import *
import torch
import torch.nn.parameter as Parameter
import random as rand
import numpy as np
import torch.nn as nn
torch.cuda.empty_cache()
from neural_modules  import *  
import os

def read_for_optim(dataset_name,qf_test_dev_prot_dict,score_def,noise_type,end_to_end,MACHINE):
    
    
    deep_flag = if_deep(score_def)
    
    if end_to_end is False:
        start_time = time.time()

        test_or_dev = 'dev'

        score_or_noise = 'score'
        pred_compress = read_dplp_plus(dataset_name, qf_test_dev_prot_dict, \
                           test_or_dev, score_or_noise, score_def=score_def,machine=MACHINE,deep=deep_flag)

        power_vector = pred_compress['power_vector']
        Filtered_q = pred_compress['FQ']
        Score_dev = pred_compress['pred'][score_or_noise]
        #################################################################################
        score_or_noise = 'noise'
        pred_compress = read_dplp_plus(dataset_name, qf_test_dev_prot_dict, \
                           test_or_dev, score_or_noise, noise_def=noise_type,machine=MACHINE,deep=deep_flag)

        Noise_dev = pred_compress['pred'][score_or_noise]


        ##################################################################################
        ##################################################################################
 

        test_or_dev = 'test'

        score_or_noise = 'score'
        pred_compress = read_dplp_plus(dataset_name, qf_test_dev_prot_dict, \
                           test_or_dev, score_or_noise, score_def=score_def,machine=MACHINE,deep=deep_flag)

        Score_test = pred_compress['pred'][score_or_noise]
        #################################################################################
        score_or_noise = 'noise'
        pred_compress = read_dplp_plus(dataset_name, qf_test_dev_prot_dict, \
                           test_or_dev, score_or_noise, noise_def=noise_type,machine=MACHINE,deep=deep_flag)

        Noise_test = pred_compress['pred'][score_or_noise]


    if end_to_end == True:

            
        graph_data = read_graph_file(dataset_name, qf_test_dev_prot_dict,machine=MACHINE)
        score_sens = read_score_file(dataset_name, qf_test_dev_prot_dict,machine=MACHINE,deep=deep_flag)
        
        
        test_or_dev='dev'
        device="cpu"
        start_time=time.time()
        power_vector = list(np.arange(0.1,2,0.01))  
        Dplp1 = DPLP_plus.DPLP(graph_data,score_sens,power_vector)
        Dplp1.score_scaling_and_noise_gen(test_or_dev,score_def)

        score_or_noise='noise' 
        ppred = Dplp1.store_noise_score(test_or_dev,score_def,noise_type,device,score_or_noise)
        Noise_dev = Dplp1.data_for_training_test[score_or_noise]

        score_or_noise='score'
        Dplp1.store_noise_score(test_or_dev,score_def,noise_type,device,score_or_noise)
        Score_dev = Dplp1.data_for_training_test[score_or_noise]

        Filtered_q = Dplp1.filtered_queries_priv

        print((time.time()-start_time)/60)
        ##########################################################
        ###########################################################
        test_or_dev='test'
        start_time=time.time()
        power_vector = list(np.arange(0.1,2,0.01))  
        Dplp2 = DPLP_plus.DPLP(graph_data,score_sens,power_vector)
        Dplp2.score_scaling_and_noise_gen(test_or_dev,score_def)

        score_or_noise='noise'
        Dplp2.store_noise_score(test_or_dev,score_def,noise_type,device,score_or_noise)
        Noise_test = Dplp2.data_for_training_test[score_or_noise]

        score_or_noise='score'
        Dplp2.store_noise_score(test_or_dev,score_def,noise_type,device,score_or_noise)
        Score_test = Dplp2.data_for_training_test[score_or_noise]

    score_noise={}
    score_noise['Score']={}
    score_noise['Noise']={}

    score_noise['Score']['dev'] =  Score_dev
    score_noise['Score']['test'] =  Score_test

    score_noise['Noise']['dev'] =  Noise_dev
    score_noise['Noise']['test'] =  Noise_test

    score_noise['FQ'] =  Filtered_q
    score_noise['power_vector'] =  power_vector    
    
    return score_noise


class optim:

    def __init__(self,dataset_name,score_noise, specs):
            
        self.sub_batch_limit = specs['sub_batch']    ## sub_batch_limit number of queries are accumaulated as one batch
        self.init_dim= specs['init_dim']
        self.nb_epoch =  specs['nb_epoch']
        self.hidden_dim = specs['hidden']
        self.device = specs['device']   
        self.lr=specs['lr']    
        self.wd = specs['weight_decay']
        self.margin = specs['margin']
        self.score_def = specs['score_def']
        self.noise_type=specs['noise']
        self.e_diff = specs['e']
        self.if_debug = specs['if_debug']
        self.specs=specs
        self.prot_frac = specs['prot_frac']
        self.dataset_name = dataset_name
        self.power_vector = score_noise['power_vector']
        pv = np.array(self.power_vector)
        self.idx_of_one=np.where(pv >0.99)[0][0]
        
        self.deep_flag = if_deep(self.score_def)
        if specs['choose_dim'] is None:
            self.choose_dim = len(self.power_vector)

        self.queries =  score_noise['FQ']
        self.pre_trained_flag = specs['pre_trained_flag']
        self.Score_dev = score_noise['Score']['dev']
        self.Score_test = score_noise['Score']['test']

        self.Noise_dev = score_noise['Noise']['dev']
        self.Noise_test = score_noise['Noise']['test']
        
            
    def gen_logfile_name(self, qf_test_dev_prot_dict,umnn=False):
        dataset_name= self.dataset_name
        e_diff = self.e_diff
        score_def = self.score_def
        noise_type = self.noise_type
        if self.deep_flag==True:
            logpath='deep_dumps/'
        else:
            logpath='dump/'
        os.makedirs(logpath, exist_ok=True)
        
        prot_frac =  qf_test_dev_prot_dict['prot_frac']

        prot_str = format(prot_frac, '.2f')

        e_str = format(e_diff, '.6f')
        
        
        
        logfile = logpath +'Optim_log'+ dataset_name \
                    + "_e_diff_" + e_str\
                    + "_prot_frac_" + prot_str\
                    + "_score_def_" + score_def\
                    +"_noise_" + noise_type
        
        if umnn is True:
            logfile = logpath +'Optim_log_umnn'+ dataset_name \
                        + "_e_diff_" + e_str\
                        + "_prot_frac_" + prot_str\
                        + "_score_def_" + score_def\
                        +"_noise_" + noise_type
            if self.pre_trained_flag is False:
                logfile = logpath +'no_pre_trained_Optim_log_umnn'+ dataset_name \
                        + "_e_diff_" + e_str\
                        + "_prot_frac_" + prot_str\
                        + "_score_def_" + score_def\
                        +"_noise_" + noise_type                

        self.logfile = logfile
        self.f_handle = open(logfile, "w")
        return self
    
    def train(self):
        
        
        dataset_name= self.dataset_name

        queries = self.queries
        nb_epoch = self.nb_epoch
        init_dim = self.init_dim
        choose_dim = self.choose_dim
        if_debug = self.if_debug
        margin = self.margin
        hidden_dim = self.hidden_dim
        e_diff = self.e_diff 
        device = self.device
        
        Score_dev=self.Score_dev
        Noise_dev=self.Noise_dev
        
        Score_test=self.Score_test
        Noise_test=self.Noise_test
        
        vv_limit = self.sub_batch_limit if if_debug is False else (len(queries)+10) 
        
        lr = self.lr
        wd = self.wd
        
        qsize= len(queries)

        model_dplp = neural_net(choose_dim-init_dim, [hidden_dim], bias=True, debug=if_debug).to(device)

        optim_dplp = torch.optim.Adam(model_dplp.parameters(), lr, weight_decay=wd)

        Store_opt={}

        auc_dev_np = np.zeros((qsize,nb_epoch))
        auc_test_np = np.zeros((qsize,nb_epoch))

        Mean_auc_dev = np.zeros((nb_epoch))
        Mean_auc_test = np.zeros((nb_epoch))

        Mean_avg_loss = np.zeros((nb_epoch))
        Number_of_sums = np.zeros((nb_epoch))

        f_handle=self.f_handle


        for epoch in range(0, nb_epoch):
#             stt = time.time()


            [avg_loss_dplp,vv,ss,loss] = [0.0]*4

            auc_v = torch.zeros((qsize,1))
            auc_t = torch.zeros((qsize,1))



            model_dplp_epoch_end = neural_net(choose_dim-init_dim, [hidden_dim], bias=True, debug=if_debug).to(device)
            model_dplp_epoch_end = copy.deepcopy(model_dplp)

            for i in range(qsize):

                vv=vv+1
                pos = Score_dev['priv']['pos'][queries[i]][:,init_dim:choose_dim] + \
                        Noise_dev['priv']['pos'][queries[i]][:,init_dim:choose_dim]/e_diff 
                neg = Score_dev['priv']['neg'][queries[i]][:,init_dim:choose_dim] + \
                        Noise_dev['priv']['neg'][queries[i]][:,init_dim:choose_dim]/e_diff 

                pos = pos.to(device)
                neg = neg.to(device)


                Pos = pos.requires_grad_()
                Neg = neg.requires_grad_()


                ss = ss + Pos.shape[0] * Neg.shape[0] 

                predPos = model_dplp(Pos)
                predNeg = model_dplp(Neg)


                margin =0.1
                loss  = loss+ pairwise_ranking_loss(predPos, predNeg, margin)

                if vv==vv_limit or i==qsize-1:
                    optim_dplp.zero_grad()
                    loss.backward()
                    optim_dplp.step()
                    avg_loss_dplp += loss.item()
                    [vv,loss] = [0]*2





                auc_v[i,:] = auc_tensor(predPos, predNeg)


            auc_v1= auc_v.squeeze(1)
            auc_dev_np[:,epoch] = auc_v1.cpu().detach().numpy()

            Mean_auc_dev[epoch]=torch.mean(auc_v,dim=[0]).cpu().detach().numpy()
            Mean_avg_loss[epoch] = avg_loss_dplp
            Number_of_sums[epoch] = ss

            f_handle.write("Epoch: " + str(epoch) + \
                           "\t LOSS:  "+ format(avg_loss_dplp, '0.6f') +\
                            "\t AUC_dev:  "+ format(torch.mean(auc_v,dim=[0]).item(), '0.6f') + '\n' )

            del auc_v, auc_v1,auc_t

        Store_opt['auc_dev'] = auc_dev_np

        Store_opt['mean_auc_dev'] = Mean_auc_dev

        Store_opt['mean_avg_loss'] = Mean_avg_loss   
        Store_opt['sum_n'] = Number_of_sums   
        Store_opt['model_epoch_last_but_one']= model_dplp_epoch_end
        Store_opt['model_epoch_last']= model_dplp
        Store_opt['specs']= self.specs
        self.Store_opt = Store_opt
        f_handle.close()
        self.f_handle = None
        
        
        return self

    
    
    def prepare_for_umnn(self):
        
        qf_test_dev_prot_dict = self.specs['qf_test_dev_prot_dict']
        
        qfrac = qf_test_dev_prot_dict['qfrac']
        test_frac = qf_test_dev_prot_dict['test_frac']
        dev_frac =  qf_test_dev_prot_dict['dev_frac']
        prot_frac =  qf_test_dev_prot_dict['prot_frac']
        score_def = self.specs['score_def']
        machine =  self.specs['machine']
        dataset_name = self.dataset_name
        e_diff = self.e_diff
        self.stf_linear = \
            load_linear_optim_output(dataset_name, qfrac, test_frac, dev_frac, prot_frac, e_diff, score_def, machine, 
                             noise_type='gumbel')
        
        self.pre_trained = self.stf_linear['model_epoch_last'].to(self.device)
    
    def train_umnn(self):
        
        dataset_name= self.dataset_name

        queries = self.queries
        nb_epoch = self.nb_epoch
        init_dim = self.idx_of_one if self.pre_trained_flag is False else self.init_dim
        choose_dim = self.idx_of_one+1 if self.pre_trained_flag is False else self.choose_dim
        if_debug = self.if_debug
        margin = self.margin
        hidden_dim = self.hidden_dim
        e_diff = self.e_diff 
        device = self.device
        
        Score_dev=self.Score_dev
        Noise_dev=self.Noise_dev
        
        Score_test=self.Score_test
        Noise_test=self.Noise_test
        pre_trained = self.pre_trained.to(device)
        
        vv_limit = 1#self.sub_batch_limit if if_debug is False else (len(queries)+10) 
        
         
        lr = self.lr
        wd = self.wd
        
        qsize= len(queries)
        
        if self.pre_trained_flag is True:
            pre_trained = self.pre_trained
            for param in pre_trained.parameters():
                param.requires_grad = False

        in_d = 2 #choose_dim-init_dim+1
        hidden_layers = [20]
        

        
        
        if self.pre_trained_flag is False:

            model_dplp= dplp_mnn_no_pre_trained(in_d, hidden_layers, self.device,nb_steps=50).to(device)
        else:
            model_dplp= dplp_mnn(in_d, hidden_layers, pre_trained, self.device,nb_steps=50).to(device)

        
        optim_dplp = torch.optim.Adam(model_dplp.parameters(), lr, weight_decay=wd)
        
        Store_opt={}

        auc_dev_np = np.zeros((qsize,nb_epoch))
        auc_test_np = np.zeros((qsize,nb_epoch))

        Mean_auc_dev = np.zeros((nb_epoch))
        Mean_auc_test = np.zeros((nb_epoch))

        Mean_avg_loss = np.zeros((nb_epoch))
        Number_of_sums = np.zeros((nb_epoch))

        f_handle=self.f_handle


        for epoch in range(0, nb_epoch):

            

            [avg_loss_dplp,vv,ss,loss] = [0.0]*4

            auc_v = torch.zeros((qsize,1))
            auc_t = torch.zeros((qsize,1))

            model_dplp_epoch_end = copy.deepcopy(model_dplp)
            for i in range(qsize):
                dev_pos={}
                dev_neg={}
                test_pos={}
                test_neg={}
                
                vv=vv+1
                dev_pos['score'] = Score_dev['pub']['pos'][queries[i]][:,init_dim:choose_dim] 
                dev_pos['noise'] = Noise_dev['pub']['pos'][queries[i]]\
                                                            [:,init_dim:choose_dim]
                
                dev_neg['score'] = Score_dev['pub']['neg'][queries[i]][:,init_dim:choose_dim]
                dev_neg['noise'] = Noise_dev['pub']['neg'][queries[i]]\
                                                            [:,init_dim:choose_dim].to(device)

                ###################################################################################
                ###################################################################################
                
              
            
                ss = ss + dev_pos['score'].shape[0] * dev_pos['score'].shape[0] 

                
                fmax = torch.ones(1,1).to(device)
                xmax = model_dplp(fmax)

                predPos = model_dplp(dev_pos['score'].to(device)) + xmax * dev_pos['noise'][:,0:1].to(device) / e_diff

                predNeg = model_dplp(dev_neg['score'].to(device)) + xmax * dev_neg['noise'][:,0:1].to(device)  / e_diff



                margin =0.1
                loss  = loss+ pairwise_ranking_loss(predPos.to(device), predNeg.to(device), margin)

                if vv==vv_limit or i==qsize-1:
                    optim_dplp.zero_grad()
                    
                    loss.backward()
                    optim_dplp.step()
                    avg_loss_dplp += loss.item()
                    [vv,loss] = [0]*2
                
                del dev_pos, dev_neg, fmax




                auc_v[i,:] = auc_tensor(predPos, predNeg)


            auc_v1= auc_v.squeeze(1)

            auc_dev_np[:,epoch] = auc_v1.cpu().detach().numpy()

            Mean_auc_dev[epoch]=torch.mean(auc_v,dim=[0]).cpu().detach().numpy()

            Mean_avg_loss[epoch] = avg_loss_dplp
            Number_of_sums[epoch] = ss

            f_handle.write("Epoch: " + str(epoch) + \
                           "\t LOSS:  "+ format(avg_loss_dplp, '0.6f') +\
                            "\t AUC_dev:  "+ format(torch.mean(auc_v,dim=[0]).item(), '0.6f') + '\n')
        Store_opt['auc_dev'] = auc_dev_np

        Store_opt['mean_auc_test'] = Mean_auc_dev

        Store_opt['mean_avg_loss'] = Mean_avg_loss   
        Store_opt['sum_n'] = Number_of_sums   
        Store_opt['model_epoch_last_but_one']= model_dplp_epoch_end
        Store_opt['model_epoch_last']= model_dplp
        Store_opt['specs']= self.specs
        self.Store_opt_umnn = Store_opt
        f_handle.close()
        self.f_handle = None
        
        
        return self    
    
    
    
    def save(self):
        dataset_name = self.dataset_name
        machine = self.specs['machine']
        config = configure(machine)
        
        if self.deep_flag==True:
            train_path = config['deep_train_path']  + dataset_name + '/'
        else:
            train_path = config['train_path']  + dataset_name + '/'
            
        score_def = self.score_def
        noise_type = self.noise_type  
        e_diff = self.e_diff

        
        qf_test_dev_prot_dict = self.specs['qf_test_dev_prot_dict']
        
        qfrac = qf_test_dev_prot_dict['qfrac']
        test_frac = qf_test_dev_prot_dict['test_frac']
        dev_frac =  qf_test_dev_prot_dict['dev_frac']
        prot_frac =  qf_test_dev_prot_dict['prot_frac']
        
        q_str = format(qfrac, '.2f')

        test_str = format(test_frac, '.2f')

        dev_str = format(dev_frac, '.2f')

        prot_str = format(prot_frac, '.2f')
        

        
        e_str = format(e_diff, '.2f')
        
        if e_diff<0.01:
            e_str = format(e_diff, '.6f')
        
        
        if self.deep_flag==True:
            train_path = config['deep_train_path']  + dataset_name + '/'
            
            train_file_name = 'deep_optim_train_'+ dataset_name  + "_qfrac_" + q_str\
                                                            + "_test_frac_" + test_str \
                                                            + "_dev_frac_" + dev_str \
                                                            + "_prot_frac_" + prot_str\
                                                            + "_score_def_" + score_def\
                                                            + "_e_p_" + e_str\
                                                            + "_noise_type_" + noise_type
            
        else:
            train_path = config['train_path']  + dataset_name + '/'
        
            train_file_name = 'optim_train_'+ dataset_name  + "_qfrac_" + q_str\
                                                            + "_test_frac_" + test_str \
                                                            + "_dev_frac_" + dev_str \
                                                            + "_prot_frac_" + prot_str\
                                                            + "_score_def_" + score_def\
                                                            + "_e_p_" + e_str\
                                                            + "_noise_type_" + noise_type

        
        train_file =  train_path + train_file_name
        stf = self.Store_opt
        save_into_pickle(stf,train_file)
        return self
    
    def save_umnn(self):
        dataset_name = self.dataset_name
        machine = self.specs['machine']
        config = configure(machine)
        
        if self.deep_flag==True:
            train_path = config['deep_train_path']  + dataset_name + '/'
        else:
            train_path = config['train_path']  + dataset_name + '/'
            
        score_def = self.score_def
        noise_type = self.noise_type  
        e_diff = self.e_diff

        
        qf_test_dev_prot_dict = self.specs['qf_test_dev_prot_dict']
        
        qfrac = qf_test_dev_prot_dict['qfrac']
        test_frac = qf_test_dev_prot_dict['test_frac']
        dev_frac =  qf_test_dev_prot_dict['dev_frac']
        prot_frac =  qf_test_dev_prot_dict['prot_frac']
        
        q_str = format(qfrac, '.2f')

        test_str = format(test_frac, '.2f')

        dev_str = format(dev_frac, '.2f')

        prot_str = format(prot_frac, '.2f')
        
        e_str = format(e_diff, '.2f')
        
        
                
        if e_diff<0.01:
            e_str = format(e_diff, '.6f')

        if self.pre_trained_flag is True:    

            if self.deep_flag==True:
                train_path = config['deep_train_path']  + dataset_name + '/'

                train_file_name = 'deep_umnn_optim_train_'+ dataset_name  + "_qfrac_" + q_str\
                                                                + "_test_frac_" + test_str \
                                                                + "_dev_frac_" + dev_str \
                                                                + "_prot_frac_" + prot_str\
                                                                + "_score_def_" + score_def\
                                                                + "_e_p_" + e_str\
                                                                + "_noise_type_" + noise_type

            else:
                train_path = config['train_path']  + dataset_name + '/'

                train_file_name = 'optim_umnn_train_'+ dataset_name  + "_qfrac_" + q_str\
                                                                + "_test_frac_" + test_str \
                                                                + "_dev_frac_" + dev_str \
                                                                + "_prot_frac_" + prot_str\
                                                                + "_score_def_" + score_def\
                                                                + "_e_p_" + e_str\
                                                                + "_noise_type_" + noise_type
            

        if self.pre_trained_flag is False:    
            
            if self.deep_flag==True:
                train_path = config['deep_train_path']  + dataset_name + '/'

                train_file_name = 'no_pre_trained_deep_umnn_optim_train_'+ dataset_name  + "_qfrac_" + q_str\
                                                                + "_test_frac_" + test_str \
                                                                + "_dev_frac_" + dev_str \
                                                                + "_prot_frac_" + prot_str\
                                                                + "_score_def_" + score_def\
                                                                + "_e_p_" + e_str\
                                                                + "_noise_type_" + noise_type

            else:
                train_path = config['train_path']  + dataset_name + '/'

                train_file_name = 'no_pre_trained_optim_umnn_train_'+ dataset_name  + "_qfrac_" + q_str\
                                                                + "_test_frac_" + test_str \
                                                                + "_dev_frac_" + dev_str \
                                                                + "_prot_frac_" + prot_str\
                                                                + "_score_def_" + score_def\
                                                                + "_e_p_" + e_str\
                                                                + "_noise_type_" + noise_type

            
        train_file =  train_path + train_file_name
        stf = self.Store_opt_umnn
        save_into_pickle(stf,train_file)
        return self