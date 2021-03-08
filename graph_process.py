import networkx as nx
import operator
from itertools import islice
import numpy as np
from random import sample
import copy
import timeit
import matplotlib.pyplot as plt
 

def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs

def ifknbrs(G, start, sink, k):
    nbrs = knbrs(G, start, k)
    if sink in nbrs:
        return 1
    else:
        return 0
    
def ifknbrX(st, sink,ll):
    if sink in ll[st]:
        return 1
    else:
        return 0
        
# read graph, select query set
class create_graph:
    def __init__(self, read_file):
        G = nx.read_edgelist(read_file)
        G = G.to_undirected()
        self.G = nx.convert_node_labels_to_integers(G)
        self.N = G.number_of_nodes()
        self.E = G.number_of_edges()

    def create_query_set(self, qfrac):
        loe=[] # list of edges
        lone=[] #list of non edges
        lope=[] #list of potential edges
        G = self.G
        all_two_len = {}
        N = self.N
        qsize = int(qfrac * N)
        two_len={}
        for i in range(N):
            two_len[i] = knbrs(G, i, 2)
            all_two_len[i] = len(two_len[i])

        d = sorted(all_two_len.items(), key=operator.itemgetter(1), reverse=True)
        self.queries = [qq for qq, nn in d][0:qsize]
        queries = self.queries
        query_full_set = {}

        for i in range(0, qsize):
            query_full_set[queries[i]] = {}
            query_full_set[queries[i]]['Nbr'] = [nbr for nbr in G[queries[i]]]
            query_full_set[queries[i]]['NonNbr'] = list(set(G.nodes()) - set(query_full_set[queries[i]]['Nbr'] + [queries[i]]))
            
            e = [(queries[i], x) for x in query_full_set[queries[i]]['Nbr']]
            ne = [(queries[i], x) for x in query_full_set[queries[i]]['NonNbr']]

            loe.extend(e)
            lone.extend(ne)
            
            lope.extend(e)
            lope.extend(ne)
            
        self.query_full_set = query_full_set;
        self.qsize = qsize
        self.qfrac = qfrac
        
        self.list_edges=loe
        self.list_non_edges=lone
        self.list_potential_edges=lope
        self.two_len=two_len
        return self


        
    def split_graph_for_training(self, test_fraction):

        Tr = test_fraction
        filtered_queries = []
        query_training_set = {}
        query_test_set = {} # test_set
        
        list_test_edges = []
        list_test_non_edges = []
        list_test_potential_edges = []
        
        list_training_edges = []
        list_training_non_edges = []
        list_training_potential_edges= []
        
        listx = self.two_len
        qsize = self.qsize
        queries = self.queries
        query_full_set = self.query_full_set
        
        query_test_two_len = {}
        query_training_two_len = {}
        
        query_test_two_len_arr = {}
        query_training_two_len_arr = {}
        
        for i in range(0, qsize):
            
            query_training_set[queries[i]] = {}
            query_test_set[queries[i]] = {}
            
            query_training_two_len[queries[i]]={}
            query_test_two_len[queries[i]]={}
            
            query_training_two_len_arr[queries[i]]={}
            query_test_two_len_arr[queries[i]]={}
            
            q_full = query_full_set[queries[i]]['Nbr']
            x = int(Tr * len(query_full_set[queries[i]]['Nbr']))
            yx = x
            
            query_test_set[queries[i]]['Nbr'] = sample(q_full, x)
            query_training_set[queries[i]]['Nbr'] = list(set(q_full)\
                                                          - set([queries[i]]\
                                                                + query_test_set[queries[i]]['Nbr']))
            
            query_training_two_len[queries[i]]['Nbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_training_set[queries[i]]['Nbr']}       
            
            
            query_test_two_len[queries[i]]['Nbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_test_set[queries[i]]['Nbr']}
            

            
            q_full = query_full_set[queries[i]]['NonNbr']
            x = int(Tr * len(query_full_set[queries[i]]['NonNbr']))
            
            if x>0 and yx >0:
                filtered_queries.append(queries[i])
                
            query_test_set[queries[i]]['NonNbr'] = sample(q_full, x)
            query_training_set[queries[i]]['NonNbr'] = list(set(q_full) \
                                                           - set([queries[i]]\
                                                                 + query_test_set[queries[i]]['NonNbr']))

            query_test_two_len[queries[i]]['NonNbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_test_set[queries[i]]['NonNbr']}
                
                
            query_training_two_len[queries[i]]['NonNbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_training_set[queries[i]]['NonNbr']}   

            
            ##########################matrix storage#########################################
            
            query_test_two_len_arr[queries[i]]['Nbr'] = np.array([[x, ifknbrX(queries[i],x,listx)]\
                                                         for x in query_test_set[queries[i]]['Nbr']])    
                
            
                        
            query_training_two_len_arr[queries[i]]['Nbr'] = np.array([[x,ifknbrX(queries[i],x,listx)]\
                                                         for x in query_training_set[queries[i]]['Nbr']])  
            
            
            query_test_two_len_arr[queries[i]]['NonNbr'] = np.array([[x,ifknbrX(queries[i],x,listx)]\
                                                         for x in query_test_set[queries[i]]['NonNbr']])                  

            query_training_two_len_arr[queries[i]]['NonNbr'] = np.array([[x,ifknbrX(queries[i],x,listx)]\
                                                         for x in query_training_set[queries[i]]['NonNbr']])    

            
            ##########################matrix storage#########################################

            
            
            e = [(queries[i], x) for x in query_test_set[queries[i]]['Nbr']]
            ne = [(queries[i], x) for x in query_test_set[queries[i]]['NonNbr']]

            e1 = [(queries[i], x) for x in query_training_set[queries[i]]['Nbr']]
            ne1 = [(queries[i], x) for x in query_training_set[queries[i]]['NonNbr']]


            list_test_edges.extend(e)
            list_test_non_edges.extend(ne)

            list_training_edges.extend(e1)
            list_training_non_edges.extend(ne1)

            
        for x,y in list_training_edges:
            if (y,x) in list_test_edges:
                list_test_edges.remove((y,x))
            
#         for x,y in list_training_non_edges:
#             if (y,x) in list_test_non_edges:
#                 list_test_non_edges.remove((y,x))

        list_potential_training_edges = list_training_edges + list_training_non_edges
        list_potential_test_edges = list_test_edges + list_test_non_edges

        G_sample = copy.deepcopy(self.G)
        G_sample.remove_edges_from(list_test_edges)
        

        self.query_training_set = query_training_set
        self.query_test_set = query_test_set
        
        self.query_training_two_len = query_training_two_len
        self.query_test_two_len = query_test_two_len

        self.query_training_two_len_arr = query_training_two_len_arr
        self.query_test_two_len_arr = query_test_two_len_arr
        
        self.list_test_edges = list_test_edges
        self.list_test_non_edges = list_test_non_edges
        self.list_test_potential_edges = list_potential_test_edges
        
        self.list_training_edges = list_training_edges
        self.list_training_non_edges = list_training_non_edges
        self.list_training_potential_edges = list_potential_training_edges

        
        self.G_sample = G_sample
        self.filtered_queries_test = filtered_queries
        
        return self
    
    

    def split_graph_for_validation(self, dev_fraction):

        Tr = dev_fraction
        filtered_queries = []

        query_true_training_set = {}
        query_dev_set = {} # test_set
        
        query_true_training_two_len = {}
        query_dev_two_len = {} # test_set
        
        query_true_training_two_len_arr = {}
        query_dev_two_len_arr = {} # test_set
        
        
        list_dev_edges = []
        list_dev_non_edges = []
        list_dev_potential_edges = []
        
        list_true_training_edges = []
        list_true_training_non_edges = []
        list_true_training_potential_edges= []
        
        
        qsize = self.qsize
        queries = self.queries
        query_training_set = self.query_training_set
        listx = self.two_len
        for i in range(0, qsize):
            
            query_true_training_set[queries[i]] = {}
            query_dev_set[queries[i]] = {}
            
            query_true_training_two_len[queries[i]] = {}
            query_dev_two_len[queries[i]] = {}
            
            query_true_training_two_len_arr[queries[i]] = {}
            query_dev_two_len_arr[queries[i]] = {}
            
            q_full = query_training_set[queries[i]]['Nbr']
            x = int(Tr * len(query_training_set[queries[i]]['Nbr']))
            
            yx =x
            
            query_dev_set[queries[i]]['Nbr'] = sample(q_full, x)
            query_true_training_set[queries[i]]['Nbr'] = list(set(q_full) \
                                                           - set([queries[i]]\
                                                                 + query_dev_set[queries[i]]['Nbr']))
            
            
            query_dev_two_len[queries[i]]['Nbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_dev_set[queries[i]]['Nbr']}
            query_true_training_two_len[queries[i]]['Nbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_true_training_set[queries[i]]['Nbr']}   

            
            
            
            q_full = query_training_set[queries[i]]['NonNbr']
            x = int(Tr * len(query_training_set[queries[i]]['NonNbr']))
            
            if x>0 and yx >0:
                filtered_queries.append(queries[i])
            
            
            query_dev_set[queries[i]]['NonNbr'] = sample(q_full, x)
            query_true_training_set[queries[i]]['NonNbr'] = list(set(q_full) \
                                                           - set([queries[i]]\
                                                                 + query_dev_set[queries[i]]['NonNbr']))
            
            
            query_dev_two_len[queries[i]]['NonNbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_dev_set[queries[i]]['NonNbr']}
            query_true_training_two_len[queries[i]]['NonNbr'] = {x: ifknbrX(queries[i],x,listx)\
                                                         for x in query_true_training_set[queries[i]]['NonNbr']} 
            

            #################################matrix-storage################################################
            
            query_dev_two_len_arr[queries[i]]['Nbr'] = np.array([[x, ifknbrX(queries[i],x,listx)]\
                                                         for x in query_dev_set[queries[i]]['Nbr']])
            query_true_training_two_len_arr[queries[i]]['Nbr'] = np.array([[x, ifknbrX(queries[i],x,listx)]\
                                                         for x in query_true_training_set[queries[i]]['Nbr']])  
            
            
            
            query_dev_two_len_arr[queries[i]]['NonNbr'] = np.array([[x, ifknbrX(queries[i],x,listx)]\
                                                         for x in query_dev_set[queries[i]]['NonNbr']])
            query_true_training_two_len_arr[queries[i]]['NonNbr'] = np.array([[x, ifknbrX(queries[i],x,listx)]\
                                                         for x in query_true_training_set[queries[i]]['NonNbr']])  
            

            e = [(queries[i], x) for x in query_dev_set[queries[i]]['Nbr']]
            ne = [(queries[i], x) for x in query_dev_set[queries[i]]['NonNbr']]
            
            e1 = [(queries[i], x) for x in query_true_training_set[queries[i]]['Nbr']]
            ne1 = [(queries[i], x) for x in query_true_training_set[queries[i]]['NonNbr']]

            list_dev_edges.extend(e)
            list_dev_non_edges.extend(ne)
            
            list_true_training_edges.extend(e1)
            list_true_training_non_edges.extend(ne1)
            
        for x,y in list_true_training_edges:
            if (y,x) in list_dev_edges:
                list_dev_edges.remove((y,x))
            
#         for x,y in list_true_training_non_edges:
#             if (y,x) in list_dev_non_edges:
#                 list_dev_non_edges.remove((y,x))

        list_potential_true_training_edges = list_true_training_edges + list_true_training_non_edges
        list_potential_dev_edges = list_dev_edges + list_dev_non_edges
        
            
        G_true_train = copy.deepcopy(self.G_sample)
        
        G_true_train.remove_edges_from(list_dev_edges)
        
        
        self.query_true_training_set = query_true_training_set
        self.query_dev_set = query_dev_set
        
        
        self.query_true_training_two_len = query_true_training_two_len
        self.query_dev_two_len = query_dev_two_len

        self.query_true_training_two_len_arr = query_true_training_two_len_arr
        self.query_dev_two_len_arr = query_dev_two_len_arr
        
        self.list_dev_edges = list_dev_edges
        self.list_dev_non_edges = list_dev_non_edges
        self.list_dev_potential_edges = list_dev_potential_edges
        self.list_true_training_potential_edges = list_potential_true_training_edges
        
        self.G_true_train = G_true_train
        
        self.filtered_queries_dev = filtered_queries

        
        return self    
    
    def assign_score(self,score):
        self.score = score
        return self
    

            
    def create_protected_graph(self,fr):
        Gs = copy.deepcopy(self.G_sample)
        x = int((1-fr) * len(Gs.edges()))
        tt = sample(Gs.edges(), x)
        Gs.remove_edges_from(tt)
        self.protected_graph = Gs
        return self
    
    def filter_query_private(self):

        query_test_set = self.query_test_set
        query_dev_set = self.query_dev_set
        
        fil_q_test=[]
        fil_q_dev=[]
        
        Gp=self.protected_graph
        
        queries = self.filtered_queries_test
        for i in range(len(queries)):
            qqN = sum([1-int(Gp.has_edge(queries[i],u))  for u in query_test_set[queries[i]]['Nbr']] )
            qqNN =sum([1-int(Gp.has_edge(queries[i],u))  for u in query_test_set[queries[i]]['NonNbr']])

            if qqN >0 and qqNN>0:
                fil_q_test.append(queries[i])

        queries = self.filtered_queries_dev

        for i in range(len(queries)):
            qqN = sum([1-int(Gp.has_edge(queries[i],u))  for u in query_dev_set[queries[i]]['Nbr']] )
            qqNN =sum([1-int(Gp.has_edge(queries[i],u))  for u in query_dev_set[queries[i]]['NonNbr']])

            if qqN >0 and qqNN>0:
                fil_q_dev.append(queries[i])
                       
                       
        self.filtered_queries_test_private = fil_q_test
        self.filtered_queries_dev_private = fil_q_dev
        return self
                       
        
def cn_priv(Gp,G,qid):
    a = set((nbr for nbr in Gp[qid]))
    b = set((nbr for nbr in G[qid]))
    return list(a & b)

def cn_comp(G,edge_list):
    a = ((x,y, len(list(nx.common_neighbors(G, x, y)))) for x,y in edge_list)
    return ((u,v,p) for u,v,p in a)



def look_up_from_matrix(u,v,score_mat):
    return [score_mat[u,v]]

def compute_raw_scores(G,Gp, queries, query_test_set, score_def,score_mat=None):
    

        
    sdict = {'AA':nx.adamic_adar_index,
          'PA':nx.preferential_attachment,
          'JC':nx.jaccard_coefficient,
          'CN':cn_comp
         }

    if score_def == 'AA' or score_def == 'PA' or score_def == 'JC' or score_def == 'CN': 
        Score = {}
        qsize = len(queries)
        for i in range(0,qsize):


            predsNbr = [(queries[i],u,1,ss, int(Gp.has_edge(queries[i],u))) for u in query_test_set[queries[i]]['Nbr']\
                                                for g1,g2,ss in sdict[score_def](G, [(queries[i],u)])]

            predsNonNbr =  [(queries[i],u,0,ss, int(Gp.has_edge(queries[i],u))) for u in query_test_set[queries[i]]['NonNbr']\
                                                    for g1,g2,ss in sdict[score_def](G, [(queries[i],u)])]


            preds= predsNbr + predsNonNbr
            preds.sort(key = operator.itemgetter(3), reverse = True)
            Score[queries[i]]=preds

        return Score
    elif score_mat is not None:
        Score = {}
        qsize = len(queries)
        for i in range(0,qsize):


            predsNbr = [(queries[i],u,1,ss, int(Gp.has_edge(queries[i],u))) for u in query_test_set[queries[i]]['Nbr']\
                                                for ss in look_up_from_matrix(queries[i],u,score_mat)]

            predsNonNbr =  [(queries[i],u,0,ss, int(Gp.has_edge(queries[i],u))) for u in query_test_set[queries[i]]['NonNbr']\
                                                    for ss in look_up_from_matrix(queries[i],u,score_mat)]


            preds= predsNbr + predsNonNbr
            preds.sort(key = operator.itemgetter(3), reverse = True)
            Score[queries[i]]=preds

        return Score        
    else:
        return None

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()
# ,'DeepWalk','LINE','Struc2Vec'
    
#### taken from diffPriv
def sens_dict(a, maxD, x):
        return {'AA': a / np.log(2),
                'PA': a * maxD,
                'CN': a,
                'JC':1,
                'GCN':1.0*np.sign(a),
                'Node2Vec':1.0*np.sign(a),
                'PRUNE':1.0*np.sign(a),
                'DeepWalk':1.0*np.sign(a),
                'LINE':1.0*np.sign(a),
                'Struc2Vec':1.0*np.sign(a)                 
                 }[x]
    
def compute_sensitivity(G, queries, score_def, protected_graph = None):
    sens={}
    qsize = len(queries)
    N=G.number_of_nodes()
    if protected_graph==None:
        for i in range(qsize):    
            maxD = max(list(dict(G.degree(range(N))).values()))
            a = G.degree[queries[i]]

            sens[queries[i]] = sens_dict(a, maxD, score_def)
            
    if isinstance(protected_graph,float) == True:
        for i in range(qsize):    
            
            maxD = max(list(dict(G.degree(range(N))).values()))
            fr = protected_graph
            a = fr*G.degree[queries[i]]
            
            sens[queries[i]] = sens_dict(a,maxD,score_def)

    else:
        for i in range(qsize):    
            
            maxD = max(list(dict(G.degree(range(N))).values()))
            Gp = protected_graph
            a = len(cn_priv(Gp, G, queries[i]))
            sens[queries[i]] = sens_dict(a,maxD, score_def)
 
    return sens

        
 # for i in range(qsize):
#     y=0
#     x=0
#     labels = [y for a,b,y,c in Score[queries[i]]['AA']]
#     predictions = [x for a,b,c,x in Score[queries[i]]['AA']]
#     prec.append(average_precision(labels, predictions))
    
