#from __future__ import division
import numpy as np
from util import *
from metrics import *
from graph_process import *
import torch
def power_mat(x,a):
    return np.power(x,1/a, dtype=np.float64)

def normalize_arr(x,y=None):
    if y is not None:
        y1 = y+1 if y>1.01 else 1 
        return x / y1
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 0.001)

def nonlinear_transform_plus(Graph, score, sens, power_exponent):
    score_trfmd={}
    sens_trfmd={}
    
    for i in range(Graph.qsize):
        
        yy = np.array(score[Graph.queries[i]], dtype=np.float64)
        yy[:,3] = normalize_arr(yy[:,3],sens[Graph.queries[i]])
        
        yyFinal= np.zeros((yy.shape[0], len(power_exponent) +5 ))
        yyFinal[:,0:5] = yy
        
        pv1=np.array([power_exponent],dtype=np.float64)
        pv=np.repeat(pv1, yy.shape[0], axis=0)
        
        yy1 = yy[:,3].reshape(yy.shape[0],1) 
        yy2 = np.repeat(yy1, len(power_exponent), axis=1)
        
        yy2 = power_mat(yy2,pv)
        yyFinal[:,5:] = yy2
        
        score_trfmd[Graph.queries[i]] = yyFinal
        
        sens_mat =  np.ones(pv1.shape) if sens[Graph.queries[i]]>0 else np.zeros(pv1.shape)
               
        sens_trfmd[Graph.queries[i]] = np.repeat(sens_mat, yy.shape[0], axis=0)
    return score_trfmd, sens_trfmd
##############################################################################################################################
    
#invarianto score_def e, but changes with prot edges
def gumbel_plus(Graph, Delta_mat,score):
    gum_noise={}
        
    for i in range(Graph.qsize):
        Del = Delta_mat[Graph.queries[i]]
        
        Zmat =  0.0*Del
        gum =  np.random.gumbel(Zmat, Del,  Del.shape)
        
        v= gum[:,0].reshape(gum.shape[0],1)
        yy = np.matrix(score[Graph.queries[i]])
        yy[:,3] = - 1e7 *np.abs(v)
        yy[:,5:] = gum
        
        gum_noise[Graph.queries[i]] = np.array(yy)
    return gum_noise

def laplace_plus(Graph, Delta_mat,score):
    lap_noise={}
        
    for i in range(Graph.qsize):
        Del = Delta_mat[Graph.queries[i]]
        Zmat =  0.0*Del
        lap =  np.random.laplace(Zmat, Del,  Del.shape) 
        
        v= lap[:,0].reshape(lap.shape[0],1)
        yy = np.matrix(score[Graph.queries[i]])
        yy[:,3] = - 1e7 *np.abs(v)
        yy[:,5:] = lap
        
        lap_noise[Graph.queries[i]] = np.array(yy)
    return lap_noise
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

class DPLP:
    def __init__(self,graph_data,score_sens,power_vector):
        self.metadata=graph_data
        self.score_sens = score_sens
        self.power_vector=power_vector
        Graph=self.metadata['Graph']
        self.filtered_queries_priv= \
            list(set(Graph.filtered_queries_test_private) & set(Graph.filtered_queries_dev_private))
        
        self.dplp_scores={}
        self.dplp_scores['test']={} 
        self.dplp_scores['dev']={}
        
        self.dplp_noise ={}
#         self.data_for_training_test ={}
        
        self.noise_done = 0
    
    def score_scaling_and_noise_gen(self,test_or_dev,score_def):
        score_sens =  self.score_sens
        
        score_all = score_sens['score_data'][test_or_dev]
        sens_all= score_sens['sens_data'][test_or_dev]
        
        score =score_all[score_def]
        sens = sens_all[score_def]
        
        Graph = self.metadata['Graph']
        dplp_scores ={}
        dplp_noise ={}
        power_vector =  self.power_vector
        score_trfmd, sens_trfmd  = nonlinear_transform_plus(Graph,score, sens,  power_vector)
            

        dplp_scores = score_trfmd

        
        if self.noise_done==0:
            dplp_noise['gumbel'] = gumbel_plus(Graph, sens_trfmd,score_trfmd) ###all query
            dplp_noise['laplace']= laplace_plus(Graph, sens_trfmd,score_trfmd)
            self.noise_done = 1
            self.dplp_noise[test_or_dev] = dplp_noise
        
        self.dplp_scores[test_or_dev][score_def] = dplp_scores
        
        return self

    def store_noise_score(self,test_or_dev,score_def,noise_type,device, score_or_noise):
        self.data_for_training_test ={}
        if score_or_noise == 'score':
            dplp = self.dplp_scores[test_or_dev][score_def]
        if score_or_noise == 'noise':
            dplp = self.dplp_noise[test_or_dev][noise_type]
            
        queries=self.filtered_queries_priv
        qsize = len(queries)
        power_vector = self.power_vector
        
        
        predTensor={'pub':{'pos': {}, 'neg':{}}, 
                    'priv':{'pos': {}, 'neg':{}}}

            
        for i in range(qsize):
            
            sz1 = int(round(dplp[queries[i]][:,2].sum()))
            sz2= dplp[queries[i]][:,2].size - sz1 
            sz = len(power_vector)
            
            row_n = dplp[queries[i]].shape[0]
            col_n = len(power_vector)
            
            predTensor['pub']['pos'][queries[i]] = torch.zeros((sz1,sz)).to(device)
            predTensor['pub']['neg'][queries[i]] = torch.zeros((sz2,sz)).to(device)


            labels = dplp[queries[i]][:,2] > 0.5
            labels = labels*1

            priv = dplp[queries[i]][:,4] > 0.5
            priv = priv*1

            sz1 = (labels*(1-priv)).sum()
            sz2 = ((1 - labels)*(1-priv)).sum()

            predTensor['priv']['pos'][queries[i]] = torch.zeros((sz1,sz)).to(device)
            predTensor['priv']['neg'][queries[i]] = torch.zeros((sz2,sz)).to(device)

 
            predictions = dplp[queries[i]][:,5:]
            labels = dplp[queries[i]][:,2] > 0.5
            labels = labels*1

            priv = dplp[queries[i]][:,4] > 0.5
            priv = priv*1
            
            predPos = predictions[labels==1,:]
            predNeg = predictions[labels==0,:]

            predPosPriv = predictions[labels*(1-priv)==1,:]
            predNegPriv = predictions[(1-labels)*(1-priv)==1,:]

            predTensor['pub']['pos'][queries[i]]= torch.FloatTensor(predPos) 
            predTensor['pub']['neg'][queries[i]]= torch.FloatTensor(predNeg) 

            predTensor['priv']['pos'][queries[i]]= torch.FloatTensor(predPosPriv) 
            predTensor['priv']['neg'][queries[i]]= torch.FloatTensor(predNegPriv) 

        Pred={}
        Pred[score_or_noise] = predTensor
        self.data_for_training_test = Pred
        return self

####################################################################################################################
####################################################################################################################
####################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
##New##

#######################################################