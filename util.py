import configparser
import pickle
import DPLP_plus


def make_dict(qfrac,test_frac,dev_frac,prot_frac):
    qf_test_dev_prot_dict={}

    qf_test_dev_prot_dict['qfrac'] = qfrac
    qf_test_dev_prot_dict['test_frac'] =  test_frac
    qf_test_dev_prot_dict['dev_frac'] = dev_frac
    qf_test_dev_prot_dict['prot_frac'] = prot_frac
    return qf_test_dev_prot_dict

def if_deep(score_def):

    if score_def in ['GCN','Node2Vec','PRUNE','DeepWalk','LINE','Struc2Vec']:
        deep_flag = True
    elif score_def  in ['AA','PA','JC','CN']:
        deep_flag = False
    return deep_flag

def configure(machine):
    x = configparser.ConfigParser()
    x.read('config.ini')
    #machine = x['MACHINE']['machine']
    config = x[machine]
    return config    

def save_into_pickle(data, filename):
    filename = filename + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_from_pickle(filename):
    filename = filename + '.pickle'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        return x
    
    
    
#######################################################


def read_graph_file(dataset_name, qf_test_dev_prot_dict,machine='CHERRY'):
    
    config = configure(machine)

    #raw_data_path = config['raw_data_path']
    out_path = config['out_graph_path']
    #score_path = config['score_path']
    
    
    qfrac = qf_test_dev_prot_dict['qfrac']
    test_frac = qf_test_dev_prot_dict['test_frac']
    dev_frac =  qf_test_dev_prot_dict['dev_frac']
    prot_frac =  qf_test_dev_prot_dict['prot_frac']
    
    
                
    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')
 
    
    graph_file = out_path + dataset_name + "_qfrac_" + q_str\
                                    + "_test_frac_" + test_str \
                                    + "_dev_frac_" + dev_str \
                                    + "_prot_frac_" + prot_str
    
     
    graphdata = read_from_pickle(graph_file)
    return graphdata


def gen_files_for_deep(dataset_name, qf_test_dev_prot_dict,machine='CHERRY'):
    
    config = configure(machine)

    #raw_data_path = config['raw_data_path']
    deep_path = config['deep_path']
    #score_path = config['score_path']
    
    
    qfrac = qf_test_dev_prot_dict['qfrac']
    test_frac = qf_test_dev_prot_dict['test_frac']
    dev_frac =  qf_test_dev_prot_dict['dev_frac']
    prot_frac =  qf_test_dev_prot_dict['prot_frac']
    
    
                
    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')
 
    
    file_prefix = deep_path + "deep_source_txt" + dataset_name + "_qfrac_" + q_str\
                                    + "_test_frac_" + test_str \
                                    + "_dev_frac_" + dev_str \
                                    + "_prot_frac_" + prot_str
    
     
    edge_file = file_prefix + ".cites"
    feat_file = file_prefix + ".content"
    return edge_file,feat_file


def read_score_file(dataset_name, qf_test_dev_prot_dict,machine='CHERRY',deep=False):
    
    config = configure(machine)

    #raw_data_path = config['raw_data_path']
    #out_path = config['out_graph_path']
    
    if deep is True:
        score_path = config['deep_score_path']
    else:
        score_path = config['score_path']
        
        
    qfrac = qf_test_dev_prot_dict['qfrac']
    test_frac = qf_test_dev_prot_dict['test_frac']
    dev_frac =  qf_test_dev_prot_dict['dev_frac']
    prot_frac =  qf_test_dev_prot_dict['prot_frac']
    
    
                
    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')
 
    

    
    if deep is True:
        score_file = score_path + 'deep_score_and_sens_'+ dataset_name + "_qfrac_" + q_str\
                                    + "_test_frac_" + test_str \
                                    + "_dev_frac_" + dev_str \
                                    + "_prot_frac_" + prot_str
    
    else:
        score_file = score_path + 'score_and_sens_'+ dataset_name + "_qfrac_" + q_str\
                                + "_test_frac_" + test_str \
                                + "_dev_frac_" + dev_str \
                                + "_prot_frac_" + prot_str
     
    scoredata = read_from_pickle(score_file)
    return scoredata
 

    
def read_dplp_plus(dataset_name, qf_test_dev_prot_dict, \
                   test_or_dev, score_or_noise, score_def='--', noise_def='--',machine='CHERRY',deep=False):
    
    config = configure(machine)



    
    qfrac = qf_test_dev_prot_dict['qfrac']
    test_frac = qf_test_dev_prot_dict['test_frac']
    dev_frac =  qf_test_dev_prot_dict['dev_frac']
    prot_frac =  qf_test_dev_prot_dict['prot_frac']
    
    
                
    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')
    
    if score_def == '--':
        score_or_noise_def =  noise_def
        
    if noise_def == '--':
        score_or_noise_def =  score_def
    
    
    
    if deep==True:
        dplp_path = config['embd_dplp_path'] + dataset_name + "_plus/"
        dplp_file_name = 'deep_dplp_plus_training_input_'+ dataset_name + "_qfrac_" + q_str\
                + "_test_frac_" + test_str \
                + "_dev_frac_" + dev_str \
                + "_prot_frac_" + prot_str\
                + "_test_or_dev_" + test_or_dev\
                + "_score_or_noise_" + score_or_noise\
                + "_" + score_or_noise_def
        
        
    else:
        dplp_path = config['dplp_path'] + dataset_name + "_plus/"
        dplp_file_name = 'dplp_plus_training_input_'+ dataset_name + "_qfrac_" + q_str\
                                + "_test_frac_" + test_str \
                                + "_dev_frac_" + dev_str \
                                + "_prot_frac_" + prot_str\
                                + "_test_or_dev_" + test_or_dev\
                                + "_score_or_noise_" + score_or_noise\
                                + "_" + score_or_noise_def        
    
    

    
 
    
    
    
    dplp_file =  dplp_path + dplp_file_name
    
    ppred = read_from_pickle(dplp_file)
    return ppred

def load_embeddings(dataset_name,qfrac,test_frac, dev_frac, prot_frac, score_def,score_path):
 

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
    stf = read_from_pickle(mat_file)
    return stf
    

def load_linear_optim_output(dataset_name,qfrac,test_frac, dev_frac, prot_frac,e_diff, score_def, 
                                                 machine="CHERRY",noise_type='gumbel'):

    deep_flag = if_deep(score_def)
    config = configure(machine)




    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')

    e_str = format(e_diff, '.2f')
    if e_diff<0.01:
        e_str = format(e_diff, '.6f')

    if deep_flag==True:
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
    stf = read_from_pickle(train_file)
    return stf

def load_linear_umnn_optim_output(dataset_name,qfrac,test_frac, dev_frac, prot_frac,e_diff, score_def, 
                                                 machine="CHERRY",noise_type='gumbel'):

    deep_flag = if_deep(score_def)
    config = configure(machine)




    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')

    e_str = format(e_diff, '.2f')
    if e_diff<0.01:
        e_str = format(e_diff, '.6f')

    if deep_flag==True:
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


    train_file =  train_path + train_file_name
    stf = read_from_pickle(train_file)
    return stf

def load_only_umnn_optim_output(dataset_name,qfrac,test_frac, dev_frac, prot_frac,e_diff, score_def, 
                                                 machine="CHERRY",noise_type='gumbel'):

    deep_flag = if_deep(score_def)
    config = configure(machine)




    q_str = format(qfrac, '.2f')

    test_str = format(test_frac, '.2f')

    dev_str = format(dev_frac, '.2f')

    prot_str = format(prot_frac, '.2f')

    e_str = format(e_diff, '.2f')
    if e_diff<0.01:
        e_str = format(e_diff, '.6f')

    if deep_flag==True:
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
    stf = read_from_pickle(train_file)
    return stf