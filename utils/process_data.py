import numpy as np
import pandas as pd
import scipy.io as sio 
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import itertools
from scipy.spatial.distance import cdist
from scipy import sparse
import h5py
import matplotlib.pyplot as plt
ROOTDIR  = "/scratch/ykaloga/"
#-------------------------- General --------------------------
def available_tasks():
    """ Return list of available tasks. """
    return ["uci7", "uci10", "uci7robustinf", "uci10robustinf", "uci7robustclassif", "uci10robustclassif", "uci10robustclassifv2",'uci7robustclassifv2',"mnist2views","tfr"]

def load_dataset(dataset):
    """ Return the specified dataset. """
    if dataset == "uci7" :
        data = load_uci(7)
    elif dataset == "uci10" :
        data = load_uci(10)
    elif  dataset == "uci7robustinf" : 
        data = load_uci(7, add_labels_as_views=True)
    elif dataset == "uci10robustinf" : 
        data = load_uci(10, add_labels_as_views=True)
    elif  dataset == "uci7robustclassif" : 
        data = load_uci(7)
    elif dataset == "uci10robustclassif" : 
        data = load_uci(10)
    elif dataset == "uci7robustclassifv2" : 
        data = load_uci(10)
    elif dataset == "uci10robustclassifv2" : 
        data = load_uci(7)
    elif dataset == "mnist2views":
        data = load_mnist2views("mnist2views")
    elif "tfr" in dataset:
        X, _, W, _ = load_twitter_fr(baseName = "twitterfr", sample_id = int(dataset.split("r")[-1]))
        data = {}
        data['X'] = X
        data['W'] = W
    return data

def split_data_intwo(data, cut = 0.9, shuffle = True):
    """ Split data in two : data_train & data_test """
    X = data["X"]
    W = data["W"]

    n = X[0].shape[0]
    s = np.arange(n)
    if shuffle :
        np.random.shuffle(s)

    cut = int(n*cut)
    W = W[s,:]
    W = W[:,s]
    Wtrain = W[:cut,:]
    Wtrain = Wtrain[:,:cut]
    Wtest = W[cut:,:]
    Wtest = Wtest[:,cut:]
    X = [x[s] for x in X]
    Xtrain = [x[:cut] for x in X]
    Xtest = [x[cut:] for x in X]
    labels_train =  list(np.array(data['labels'])[s])[:cut]
    labels_test =  list(np.array(data['labels'])[s])[cut:]

    data_train = {}
    data_train['X'] = Xtrain
    data_train['W'] = Wtrain
    data_train['nb_clusters'] = data['nb_clusters']
    data_train['labels'] = labels_train

    data_test = {}
    data_test['X'] = Xtest
    data_test['W'] = Wtest
    data_test['nb_clusters'] = data['nb_clusters']
    data_test['labels'] = labels_test

    return data_train, data_test

def get_labels(dataset):
    """ Return the label of specified dataset (When possible). """
    return load_dataset(dataset)['labels']

def normalize_features(x):
    """ x - MEAN(x) / STD(x) """
    mean = np.mean(x, axis=0) 
    std = np.std(x, axis = 0) 
    std[ np.where(std == 0) ] = 1
    x =  (x - mean)/std
    return x

def preprocess_data(data):
    """ Normalize features and weighted adjacency matrix. """
    X = data["X"]
    W = data["W"]
    X = [  normalize_features(x)  for x in X]
    X = [x.astype('float32')  for x in X]
    W = W.astype('float32')
    if sparse.issparse(W):
        W = W/np.max(W) + sparse.eye(W.shape[0])
    else:
        W = W/np.max(W) + np.eye(W.shape[0]).astype('float32')
    data["X"] = X
    data["W"] = W
    return data

def create_save_rootfolder(task, NOW):
    """ Create root folder results/task for save informations about MVGCCA training. """
    if not os.path.isdir(ROOTDIR+'results'):
        os.system('mkdir '+ROOTDIR+'results')
    if not os.path.isdir(ROOTDIR+'results/'+task):
        os.system('mkdir '+ROOTDIR+'results/'+task)
    if not os.path.isdir(ROOTDIR+'results/'+task+'/'+NOW):
        os.system('mkdir '+ROOTDIR+'results/'+task+'/'+NOW) 
   
def update_path(parameters, task, NOW, parameters_id, run_id):
    """ Change the path where we save information about current run. """
    any_write = parameters["write_weights"] or parameters["write_loss"] or parameters["write_latent_space"] or parameters["evaluation"]
    if any_write:
        create_save_rootfolder(task, NOW)
        if  not os.path.isdir(ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)):
            os.system('mkdir '+ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id))
        if  not os.path.isdir(ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)):
            os.system('mkdir '+ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id))
        sio.savemat(ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/parameters_dict',parameters)
    parameters["parameters_main_path"] = ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)
    parameters["weights_path"] = ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/weights/' if parameters["write_weights"] else ''
    parameters["write_loss_path"] = ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/logs/' if parameters["write_loss"] else ''
    parameters["latent_space_path"] = ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/embeddings/' if parameters["write_latent_space"] else ''
    parameters["evaluation_path"] = ROOTDIR+'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/evaluation/' if parameters["evaluation"] else ''
    os.system('mkdir '+ parameters["weights_path"] +' '+parameters["write_loss_path"]+' '+parameters["latent_space_path"]+' '+parameters["evaluation_path"] )
    return parameters

def get_all_sublist( set, size_of_subset = 1):
    """ Return list of sublist of specified size. 
    Example : 
                get_all_sublist( set = [1, 2, 3], size_of_subset = 2)
                ---> [[1,2],[1,3],[2,3]]
    """
    return [list(l) for l in list(itertools.combinations(set,  size_of_subset)) ]
#-------------------------- Create Graph --------------------------
def compute_gk(X):
    """ Compute gaussian kernel. Quicker version. """
    K =  cdist(X,X)**2
    K =  np.exp(- K/np.mean(K))
    return K

def compute_nk(K,k):
    """ Compute weight matrix from gaussian kernel. k is the number of shortest neighbor to consider. """
    A = 0*K
    D =  cdist(K,K)
    for i in range(K.shape[0]):
        d = D[i,:]
        indices = np.argsort(d)[0:k+1]
        indices = indices[np.where(indices != i)]
        A[indices,i] = 1
        A[i,indices] = 1
    A = A + np.transpose(A)
    A = np.array(A > 0,dtype=np.float32)
    return A*K
#-------------------------- UCI --------------------------
def load_uci(nb_clusters = 7, add_labels_as_views = False):
    """ Load uci7 or uci10 dataset. nb_clusters = 7 or 10. """
    Q = sio.loadmat("./datasets/uci"+str(nb_clusters)+"/"+"uci"+str(nb_clusters)+".mat")   
    X = [Q["X1"],Q["X2"],Q["X3"],Q["X4"],Q["X5"],Q["X6"]]
    W = Q["W"]
    data = {}
    data["X"] = X
    data["W"] = W
    data["labels"] = get_uci_labels(nb_clusters = nb_clusters)
    data["nb_clusters"] = nb_clusters
    if add_labels_as_views : 
        X.append(np.eye(nb_clusters)[np.array(data["labels"])])
    return data

def get_uci_labels(nb_clusters = 7):
    """ Return the label of uci7 or uci10. nb_clusters = 7 or 10. """
    labels = []
    for k in range(nb_clusters):
        for l in range(200):
            labels.append(k)
    return labels
#-------------------------- MNIST2Views --------------------------
def load_mnist2views(filename = "mnist2views"):
    Q = h5py.File(ROOTDIR+"datasets/mnist2views/"+filename+".mat", 'r+')
    data = {}
    data["X"] = [np.transpose(Q["feat_rotate"][()]), np.transpose(Q["feat_noisy"][()])]
    data['labels'] = list(Q["labels_flat"][()][0])
    data["nb_clusters"] = 10
    data["W"] = sparse.csc_matrix( (Q['W']['data'][()], Q['W']['ir'][()], Q['W']['jc'][()]) )
    return data
#-------------------------- Twitter Friend Recomandation --------------------------
def load_twitter_fr(n_sampling = 4*5*2506, n_top = 20, sample_id = 0,loadIfExist = True, saveIfnotExist = False , baseName = "twitterfr"):
    """ Load sampled version of twitter friend recomandation. """
    name = baseName+str(sample_id)
    if loadIfExist and  os.path.isfile(ROOTDIR+'datasets/tfr/'+name+".mat"):
        Q = sio.loadmat(ROOTDIR+"datasets/tfr/"+name+".mat")
        X, X_id, W = np.transpose(Q["X"],[2,0,1]), Q["X_id"], Q["W"]
    else : 
        saveIfnotExist = True
        X, X_id, W = get_twitter_sampling_fr(n_sampling)
    if saveIfnotExist :
        sio.savemat(ROOTDIR+"datasets/tfr/"+name+".mat", mdict={
        'X' : X,
        'X_id' : X_id,
        'W' : W })    
    top_celebrities, number_top_celebrities = get_top_celebrities_among_users(n_top = n_top, users_id =X_id)
    dict_hashtag_users = get_users_per_celebrities(top_celebrities,X_id)
    return list(X), np.squeeze(X_id), W, dict_hashtag_users 

def get_twitter_sampling_fr(n_sampling):
    """ Return one sample of twitter friend recommendation dataset. n_sampling is the number of users of this sample. """
    X, X_id = sampling_twitter_users_features(n_sampling = n_sampling)
    W = compute_nk( K = compute_gk(X[:,:,2]) , k = 50 )
    X = X[:,:,[0,3,5]]  
    return X, X_id, W

def sampling_twitter_users_features(n_sampling):
    twitter_dataset_name = 'datasets/twitter/user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv'
    twitter_dataset = pd.read_csv(twitter_dataset_name,sep=" ",delimiter='\t')
    # sampling
    n_total = 102327
    S = np.arange(n_total)
    np.random.shuffle(S)
    index = []
    X_id = []
    k = 0
    p = 0
    while k < n_sampling:
        if np.sum(twitter_dataset.iloc[S[p],1:7].values) == 6:
            index.append(S[p])
            X_id.append(twitter_dataset.iloc[S[p],:1].values[0])
            k = k + 1
        p = p + 1
    twitter_sample = twitter_dataset.iloc[index,7:].values
    X = np.concatenate( np.array( [ np.concatenate( [ np.array(twitter_sample[j,k].split(' '),dtype=np.float32)[:,np.newaxis] for k in np.arange(6) ] , axis = 1)[:,:,np.newaxis] for j in np.arange(n_sampling) ] ), axis = 2)
    X = np.transpose(X,(2,0,1))
    X_id = np.array(X_id,dtype=np.int64)
    return X, X_id

def get_users_per_celebrities(celebrities_list,users_id):
    users_id = np.squeeze(users_id)
    dict_cu = {}
    f = open('datasets/twitter/friend_and_hashtag_prediction_userids/uidPerFriend_fuse_all.txt')
    for line in f:
        A = line.split("\t")
        A[-1] = A[-1][:-1]
        celebrities_in_file = A[0]
        if celebrities_in_file in celebrities_list: 
            dict_cu[celebrities_in_file] = np.intersect1d(users_id, np.array(A[1:],dtype=np.int64))
    return dict_cu

def get_top_celebrities_among_users(n_top,users_id):
    users_id = np.squeeze(users_id)
    top_celebrities = ["vide"]
    number_top_celebrities = [0]
    f = open('datasets/twitter/friend_and_hashtag_prediction_userids/uidPerFriend_fuse_all.txt')
    for line in f:
        A = line.split("\t")
        A[-1] = A[-1][:-1]
        Intersect = np.intersect1d(users_id, np.array(A[1:],dtype=np.int64))
        if len(top_celebrities) < n_top and Intersect.shape[0] > 0:
            top_celebrities.append(A[0])
            number_top_celebrities.append(Intersect.shape[0])
        else:
            if Intersect.shape[0] > min(number_top_celebrities) and Intersect.shape[0] > 0:
                indice_min = np.argmin(number_top_celebrities)
                if type(indice_min) is np.ndarray:
                    indice_min = indice_min[0]
                top_celebrities[indice_min] = A[0]
                number_top_celebrities[indice_min] = Intersect.shape[0]
    f.close()  
    return top_celebrities, number_top_celebrities
