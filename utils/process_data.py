import numpy as np
import pandas as pd
import scipy.io as sio 
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
#-------------------------- General --------------------------
def available_tasks():
    return ["uci7", "uci10"]

def load_dataset(dataset):
    if dataset == "uci7":
        data = load_uci(7)
    elif dataset == "uci10":
        data = load_uci(10)
    return data

def preprocess_data(data):
    X = data["X"]
    W = data["W"]
    X = [ (x - np.mean(x, axis=0) ) / np.std(x, axis = 0) for x in X]
    X = [x.astype('float32')  for x in X]
    W = W.astype('float32')
    W = W/np.max(W) + np.eye(W.shape[0]).astype('float32')
    data["X"] = X
    data["W"] = W
    return data

def create_save_rootfolder(task, NOW):
    if not os.path.isdir('results'):
        os.system('mkdir results')
    if not os.path.isdir('results/'+task):
        os.system('mkdir results/'+task)
    if not os.path.isdir('results/'+task+'/'+NOW):
        os.system('mkdir results/'+task+'/'+NOW) 
   
def update_path(parameters, task, NOW, parameters_id, run_id):
    any_write = parameters["write_weights"] or parameters["write_loss"] or parameters["write_latent_space"] or parameters["evaluation"]
    if any_write:
        create_save_rootfolder(task, NOW)
        if  not os.path.isdir('results/'+task+'/'+NOW+'/parameters'+str(parameters_id)):
            os.system('mkdir results/'+task+'/'+NOW+'/parameters'+str(parameters_id))
        if  not os.path.isdir('results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)):
            os.system('mkdir results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id))
        sio.savemat('results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/parameters_dict',parameters)
    parameters["parameters_main_path"] = 'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)
    parameters["weights_path"] = 'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/weights/' if parameters["write_weights"] else ''
    parameters["write_loss_path"] = 'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/logs/' if parameters["write_loss"] else ''
    parameters["latent_space_path"] = 'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/embeddings/' if parameters["write_latent_space"] else ''
    parameters["evaluation_path"] = 'results/'+task+'/'+NOW+'/parameters'+str(parameters_id)+'/run'+str(run_id)+'/evaluation/' if parameters["evaluation"] else ''
    os.system('mkdir '+ parameters["weights_path"] +' '+parameters["write_loss_path"]+' '+parameters["latent_space_path"]+' '+parameters["evaluation_path"] )
    return parameters
#-------------------------- Create Graph --------------------------
#compute gaussian kernel
def compute_gk(X):
    K = np.zeros((X.shape[0],X.shape[0]))
    for i in tqdm(range(X.shape[0]),unit="row_gk"):
        for j in range(i,X.shape[0]):
            K[i,j] = np.linalg.norm(X[i,:]-X[j,:]) ** 2
    K = K + np.transpose(K)
    K =  np.exp(- K/np.mean(K))
    return K
#compute weight matrix from gaussian kernelcd ..
def compute_nk(K,k):
    A = 0*K
    D = 0*K 
    for i in tqdm(range(K.shape[0]),unit="row_wm"):
        for j in range(i,K.shape[0]):
            D[i,j] = np.linalg.norm(K[:,i]-K[:,j])
    D = D + np.transpose(D)
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
def load_uci(nb_clusters = 7):
    Q = sio.loadmat("./datasets/uci"+str(nb_clusters)+"/"+"uci"+str(nb_clusters)+".mat")   
    X = [Q["X1"],Q["X2"],Q["X3"],Q["X4"],Q["X5"],Q["X6"]]
    W = Q["W"]
    data = {}
    data["X"] = X
    data["W"] = W
    data["labels"] = get_uci_labels(nb_clusters = nb_clusters)
    data["nb_clusters"] = nb_clusters
    return data

def get_uci_labels(nb_clusters = 7):
    labels = []
    for k in range(nb_clusters):
        for l in range(200):
            labels.append(k)
    return labels
#-------------------------- Twitter Friend Recomandation --------------------------
def load_twitter_fr(n_sampling = 2506, n_top = 20, sample_id = 0,loadIfExist = True, saveIfnotExist = True , baseName = "twitterfr"):
    name = baseName+str(sample_id)
    if loadIfExist and os.path.isdir('datasets/tfr/'+name+".mat"):
        Q = sio.loadmat("datasets/tfr/"+name+".mat")
        X, X_id, W = np.transpose(Q["X"],[2,0,1]), Q["X_id"], Q["W"]
        saveIfnotExist = False
    else : 
        X, X_id, W = get_twitter_sampling_fr(n_sampling)

    if saveIfnotExist :
        sio.savemat("datasets/tfr/"+name+".mat", mdict={
        'X' : X,
        'X_id' : X_id,
        'W' : W })    
    top_celebrities, number_top_celebrities = get_top_celebrities_among_users(n_top = n_top, users_id =X_id)
    dict_hashtag_users = get_users_per_celebrities(top_celebrities,X_id)
    data = {}
    data["X"] = X
    data["W"] = W
    data["X_id"] = np.squeeze(X_id)
    data["dict_hashtag_users"] = dict_hashtag_users
    return data  

def get_twitter_sampling_fr(n_sampling):
    X, X_id = sampling_twitter_users_features(n_sampling = n_sampling)
    W = compute_nk( compute_gk(X[:,:,2]) , 50 )
    X = X[:,:,[0,3,5]]  
    return X, X_id, W
    
def sampling_twitter_users_features(n_sampling):
    twitter_dataset_name = './datasets/twitter/user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv'
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
