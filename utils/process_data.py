import numpy as np
import scipy.io as sio 
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

#-------------------------- General --------------------------
def available_tasks():
    return ["uci7", "uci10"]

def load_dataset(dataset):
    if dataset == "uci7":
        X, W = load_uci(7)
    elif dataset == "uci10":
        X, W = load_uci(10)
    return X, W

def preprocess_data(X, W):
    X = [ (x - np.mean(x, axis=0) ) / np.std(x, axis = 0) for x in X]
    X = [x.astype('float32')  for x in X]
    W = W.astype('float32')
    W = W/np.max(W) + np.eye(W.shape[0]).astype('float32')
    return X, W

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
#-------------------------- UCI --------------------------
def load_uci(nb_clusters = 7):
    data = sio.loadmat("./datasets/uci"+str(nb_clusters)+"/"+"uci"+str(nb_clusters)+".mat")   
    Xuci = [data["X1"],data["X2"],data["X3"],data["X4"],data["X5"],data["X6"]]
    Wuci = data["W"]
    return Xuci, Wuci

def get_uci_labels(nb_clusters = 7):
    labels = []
    for k in range(nb_clusters):
        for l in range(200):
            labels.append(k)
    return labels
