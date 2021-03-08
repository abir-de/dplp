import copy
# import timeit
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
from deep_package.pygcn.models import *
import deep_package.node2vec.src.node2vec as node2vec
from gensim.models import Word2Vec

import deep_package.PRUNE.src.PRUNE as prune_module
from deep_package.GraphEmbeddingUnified.ge.models.deepwalk import DeepWalk
from deep_package.GraphEmbeddingUnified.ge.models.line import LINE
from deep_package.GraphEmbeddingUnified.ge.models.struc2vec import Struc2Vec


from DPLP_plus import *
from sklearn.linear_model import LogisticRegression as logs

def compute_score_matrix(Graph_struct,embeddings):
    
    Graph=Graph_struct
    
    embd = embeddings
    embd = embd.detach().numpy()
    
    q_G = Graph. query_true_training_set
    NN = len(Graph.list_true_training_potential_edges)
    
    d = embd.shape[1]
    N = embd.shape[0]
    
    Feat = np.zeros((NN, d)) 
    labels = np.zeros(NN)
    queries  = Graph.queries
    
    i=0
    ii=0
    for ii in range(Graph.qsize):
        q_nlist = np.array(q_G[queries[ii]]['Nbr'], dtype=np.int32)
        
        Feat[i:i+q_nlist.shape[0], :] = np.abs(embd[q_nlist,:]-embd[queries[ii],:])
        labels[i:i+i+q_nlist.shape[0]] =1
        i= i+q_nlist.shape[0]
        
        
        q_nnlist = np.array(q_G[queries[ii]]['NonNbr'], dtype=np.int32)

        Feat[i:i+q_nnlist.shape[0], :] = np.abs(embd[q_nnlist,:]-embd[queries[ii],:])
        labels[i:i+q_nnlist.shape[0]]=-1
        i= i+q_nnlist.shape[0]

    llog = logs(tol=0.0001, max_iter=1000, solver='liblinear')
    llog.fit(Feat,labels)
    
    
    
    scoreMat = np.zeros((N,N))
    
    for i in range(Graph.qsize):
        FF = np.abs(embd - embd[i,:])
        vv= np.matmul(FF, llog.coef_.transpose()) + llog.intercept_
        scoreMat[:,i:i+1] = normalize_arr(vv) 

    return scoreMat
    

def compute_feature_only_GCN(GG):

    degrees = np.array([GG.degree(n) for n in GG.nodes()])
    triangles = np.array([nx.triangles(GG,n) for n in GG.nodes()])
    pr=nx.pagerank(GG,alpha=0.9)
    pageranks = np.array([pr[n] for n in GG.nodes()])

    deg = normalize_arr(degrees)
    triangles=normalize_arr(triangles)
    pageranks=normalize_arr(pageranks)
    yy = np.zeros((deg.shape[0],4+len(degrees)))
    
    yy[:,0]=deg
    yy[:,1]=triangles
    yy[:,2]=pageranks
    yy[:,3]=1
    yy[:,4:]=nx.to_numpy_matrix(GG)
    
    return yy

def normzd_embeddings(embeddings,N):
    if isinstance(embeddings,np.ndarray) is False:
        x=embeddings['0']
        emb = np.zeros((N, x.shape[0])) 
        for i_str in  embeddings.keys():
            emb[int(i_str),:] = normalize_arr(embeddings[i_str])
        embeddings =  torch.FloatTensor(emb)
        return  embeddings
    
    else:
        for i in range(embeddings.shape[0]):
            embeddings[i,:] =normalize_arr(embeddings[i,:])

        embeddings =  torch.FloatTensor(embeddings)

        return  embeddings 


class GraphConv:
    def __init__(self,dataset_name,Graph,specs,device):
        self.device = "cpu"
        self.G = Graph
        self.specs =specs
        self.N = Graph.number_of_nodes()
        self.hidden=specs['hidden']
        self.nout=specs['nout']
        self.dropout=specs['dropout']
        self.lr=specs['lr']
        self.wd=specs['wt_decay']
        self.nb_epoch=specs['nb_epochs']
        self.features = torch.FloatTensor(compute_feature_only_GCN(self.G))
        features = self.features


        self.adj = torch.FloatTensor(nx.to_numpy_matrix(self.G))


    
    def train(self):
        
        
        self.model = GCN(nfeat=self.features.shape[1],
                             nhid=self.hidden,
                             nout=self.nout,
                             dropout=self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.lr, weight_decay=self.wd)

        
        
        model = self.model
        optimizer=self.optimizer
        adj=self.adj
        features =  self.features
        
        for epoch in range(self.nb_epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train, per_edge_loss,ell = laplacian(output, adj)
            loss_train.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()/ self.G.number_of_edges()),
                    'time: {:.4f}s'.format(time.time() - t))
        embeddings = output.detach().numpy()
        
        return normzd_embeddings(embeddings,self.N)

class Node2Vec:
    def __init__(self,dataset_name,Graph,specs,device):
        self.G = Graph
        self.specs =specs
        self.N = Graph.number_of_nodes()

#         self.hidden=specs['hidden']
#         self.nout=specs['nout']
#         self.dropout=specs['dropout']
#         self.lr=specs['lr']
#         self.wd=specs['wt_decay']
#         self.nb_epoch=specs['nb_epochs']
        self.p = specs['p']
        self.q = specs['q']
        self.walk_length = specs['walk_length']
        self.num_walks = specs['num_walks']
        self.dimensions =  specs['dimensions']
        self.window_size=  specs['window_size']
        self.workers=  specs['workers']
        self.iters=  specs['iters']
        for edge in Graph.edges():
            Graph[edge[0]][edge[1]]['weight'] = 1
        Graph = Graph.to_undirected()
        
        self.GN2V= node2vec.Graph(Graph, False, self.p, self.q)

    def train(self):
        
        G = self.GN2V
        
        G.preprocess_transition_probs()
        walks = G.simulate_walks(self.num_walks, self.walk_length)

        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=self.dimensions, 
                         window=self.window_size,
                         min_count=0,
                         sg=1, workers=self.workers,
                         iter=self.iters)
        
        emb = np.zeros((self.G.number_of_nodes(), self.dimensions)) 
        for i in range(self.G.number_of_nodes()):
            emb[i,:] = normalize_arr(model[str(i)])
        embeddings =  torch.FloatTensor(emb)
        return  embeddings

class PRUNE:
    def __init__(self,dataset_name,Graph,specs,device):

        dump_path="deep_dumps/" + dataset_name + ".elist"
        self.out = "deep_dumps/" + dataset_name + ".emb"
        nx.write_edgelist(Graph,dump_path,delimiter='\t',data=False)
        self.graph = np.loadtxt(dump_path).astype(np.int32)
        self.nodeCount = Graph.number_of_nodes()
        self.N = Graph.number_of_nodes()

        self.lamb = specs['lamb']
        self.dimension = specs['dimensions']
        self.learning_rate =specs['lr']             
        self.epoch=specs['nb_epochs']
        self.gpu_fraction=specs['gpu']
        self.batchsize=specs['batch_size']
        self.print_every_epoch=1
        self.save_checkpoints=False

    def train(self):
        output = prune_module.run_PRUNE(self.lamb, self.graph, self.nodeCount,\
                               self.dimension, self.learning_rate,\
                               self.epoch, self.gpu_fraction,\
                               self.batchsize, self.print_every_epoch,\
                               save_cp=False)
#         np.savetxt(self.out, embeddings, delimiter=',')
#         embeddings = np.loadtxt(self.out).astype(np.float32)
        embeddings=output
        return normzd_embeddings(embeddings,self.N)

    
class DeepWalk_:
    def __init__(self,dataset_name,Graph,specs,device):
            dump_path="deep_dumps/" + dataset_name + ".elist"
            nx.write_edgelist(Graph,dump_path,delimiter='\t',data=False)
            G = nx.read_edgelist(dump_path,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
            self.graph = Graph
            model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
            self.model=model
            
            self.specs = specs
            self.N = Graph.number_of_nodes()

            
    def train(self):
        model = self.model
        model.train(embed_size = self.specs['dimensions'], window_size=5, iter=5)
        embeddings = model.get_embeddings()
 
        return  normzd_embeddings(embeddings,self.N)#embeddings
        
class LINE_:
    def __init__(self,dataset_name,Graph,specs,device):
            dump_path="deep_dumps/" + dataset_name + ".elist"
            nx.write_edgelist(Graph,dump_path,delimiter='\t',data=False)
            G = nx.read_edgelist(dump_path,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
            self.graph = Graph
            self.specs = specs

            
            self.N = Graph.number_of_nodes()
            model = LINE(G, embedding_size= self.specs['dimensions'], order='second')
            self.model=model

            
            
    def train(self):
        model = self.model        
        model.train(batch_size=200, epochs=self.specs['nb_epochs'], verbose=2)
        embeddings = model.get_embeddings()
 
        return  normzd_embeddings(embeddings,self.N)

        
class Struc2Vec_:
    def __init__(self,dataset_name,Graph,specs,device):
            dump_path="deep_dumps/" + dataset_name + ".elist"
            nx.write_edgelist(Graph,dump_path,delimiter='\t',data=False)
            G = nx.read_edgelist(dump_path,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
            self.graph = Graph
            self.specs = specs

            self.N = Graph.number_of_nodes()
            model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )

            self.model=model

            
            
    def train(self):
        model = self.model        
        model.train(embed_size = self.specs['dimensions'], window_size=5, iter=5)
        embeddings = model.get_embeddings()
 
        return  normzd_embeddings(embeddings,self.N)

#     ['GCN','Node2Vec','PRUNE','DeepWalk','LINE','Struc2Vec']    
class Deep_embedding:
    def __init__(self,score_def,dataset_name,graph_data,specs):
        nrl={'GCN':GraphConv,
            'Node2Vec':Node2Vec,
            'PRUNE':PRUNE,
             'LINE':LINE_,
             'DeepWalk':DeepWalk_,
             'Struc2Vec':Struc2Vec_}
        self.score_def = score_def
        self.graph_struct = graph_data['Graph']
        Graph = graph_data['Graph'].G_sample
        device=specs['device']
        self.method = nrl[score_def](dataset_name,Graph,specs,device)
        self.specs=specs
        self.dataset_name=dataset_name
        
    def train(self):
        embeddings = self.method.train()
        self.score_mat = compute_score_matrix(self.graph_struct,embeddings)
        return self
    
    def save(self):
        
        dataset_name = self.dataset_name
        machine = self.specs['machine']
        config = configure(machine)
        score_path = config['deep_score_path'] 
        
        score_def = self.score_def
        
        
        qf_test_dev_prot_dict = self.specs['qf_test_dev_prot_dict']
        
        qfrac = qf_test_dev_prot_dict['qfrac']
        test_frac = qf_test_dev_prot_dict['test_frac']
        dev_frac =  qf_test_dev_prot_dict['dev_frac']
        prot_frac =  qf_test_dev_prot_dict['prot_frac']
        
        q_str = format(qfrac, '.2f')

        test_str = format(test_frac, '.2f')

        dev_str = format(dev_frac, '.2f')

        prot_str = format(prot_frac, '.2f')
        
        mat_name = 'matrix_score_'+ dataset_name  + "_qfrac_" + q_str\
                                                        + "_test_frac_" + test_str \
                                                        + "_dev_frac_" + dev_str \
                                                        + "_prot_frac_" + prot_str\
                                                        + "_score_def_" + score_def\

        
        mat_file =  score_path + mat_name
        stf = self.score_mat
        save_into_pickle(stf,mat_file)
        return self