import os  
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio
import logging
import traceback
import scipy.linalg as sl
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from utils import process
from utils import process_data
from utils import process_evaluation
from models.mvgcca import Mvgcca
from sklearn.model_selection import ParameterGrid
from scipy import sparse
import scipy
from timeit import default_timer as time

NOW = datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')
MAX_RESTART = 1

def str2bool(string):
    """ Convert string to corresponding boolean.
        -  string : str
    """
    if string in ["True","true","1"]:
        return True
    elif string in ["False","false","0"]:
        return False
    else :
        return False

def create_views_batch_size(X, W = [], batch_size = 32, shuffle = True, precomputed_batch = False):
    """ Creates a list of batch of multiviews data. 
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
    """
    views = [np.copy(original_view) for original_view in X]
    n = views[0].shape[0]
    s = np.arange(n) 
    if shuffle:
        np.random.shuffle(s)
        if not W == []:
            W = W[s,:]
            W = W[:,s]
        for i, view in enumerate(views) :
            views[i] = view[s]  
    q = n//batch_size
    block_end = q*batch_size    
    batch_indice = [ [k*batch_size,(k+1)*batch_size]  for k in range(q)] + [[q*batch_size,n]]
    batch_views = [ [ view[ind[0]:ind[1]] for view in views ] for ind in batch_indice]
    batch_adj = []
    isparse = scipy.sparse.issparse(W)
    if not  W == []:
        for ind in batch_indice:
            if isparse:
                batch_adj.append(W[ind[0]:ind[1],ind[0]:ind[1]].todense())
            else:
                batch_adj.append(W[ind[0]:ind[1],ind[0]:ind[1]])
    else :
        for ind in batch_indice:
            W = np.eye(ind[1]-ind[0])
            batch_adj.append(W)
    batch_s = [  s[ind[0]:ind[1]]  for ind in batch_indice]
    if batch_views[-1][0].shape[0] == 0 :
        return batch_views[:-1], batch_adj[:-1], batch_s[:-1]
    return batch_views, batch_adj, batch_s

def get_reconstruct_views_from_someviews(X, W, model, views_id = None):
    """ Return a list containing the reconstructed views from a limited numbers of available views, the available views are the same for all instances.
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
            - model : trained model MVGCCA  
    """
    batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = X[0].shape[0], shuffle=False)
    X_recon = model.get_reconstruct_views_from_someviews(batch_views[0],W, views_id)
    return X_recon

def get_mvgcca_latents_space(X, W, model):
    """ Return a list containing the common latent space Z and all the views latent space Z_m.
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
            - model : trained model MVGCCA    
    """
    if scipy.sparse.issparse(W) and W.shape[0] > 500:
        Z_list = [[] for views in X]
        Z_list.append([])
        batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = 512, shuffle=False)
        for views, adj in zip(batch_views, batch_adj):
            tampon = model.get_latents_space(views,adj)
            for i, t in enumerate(tampon):
                Z_list[i].append(t)
        Z_list = [np.concatenate(Z,axis = 0) for Z in Z_list]
    else :
        batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = X[0].shape[0], shuffle=False)
        Z_list = model.get_latents_space(batch_views[0],batch_adj[0])
    return Z_list

def get_mvgcca_common_latent_space_from_someviews(X, W, model,views_id):
    """ Return the common latent space Z reconstructed from a subset of views, the available views are the same for all instances.
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
            - model : trained model MVGCCA      
    """
    batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = X[0].shape[0], shuffle=False)
    Z = model.get_common_latent_space_from_someviews(batch_views[0],batch_adj[0],views_id)
    return Z

def get_mvgcca_common_latent_space_from_someviews2(X, W, model,views_id_tab):
    """ Return the common latent space Z reconstructed from a subset of views, the available views can be not the same for all instances.
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
            - model : trained model MVGCCA    
    """
    batch_adj_tab = []
    for w in W:
        batch_views, batch_adj, batch_s = create_views_batch_size(X, w, batch_size = X[0].shape[0], shuffle=False)
        batch_adj_tab.append(batch_adj[0])
    Z = model.get_common_latent_space_from_someviews2(batch_views[0],batch_adj_tab,views_id_tab)
    return Z

def save_mvgcca_latents_space(X, W, model, path, prefix, epochs):
    """Saves the list containing the common latent space Z and all the views latent space Z_m.
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
            - model : trained model MVGCCA    
            - path : str
            - epochs (which epochs is saved): str
    """
    if prefix != '' :
        prefix = "_"+prefix
    Z_list = get_mvgcca_latents_space(X, W, model)
    key = ['t'+str(s) for s in range(len(Z_list))]
    dictionary = dict(zip(key, Z_list))
    sio.savemat(path+"latent_space_"+str(epochs+1)+'epochs'+prefix+'.mat',dictionary)  
    return Z_list[0]

def save_multiple_mvgcca_latents_space(Xlist, Wlist, model, path, epochs):
    """ This function replaces the previous one when the matrix W is too large for the system memory. In this case we save the latent space from a set of batch size of the initial multiviews data. 
        Hence it saves the list containing the common latent space Z and all the views latent space Z_m.
            - Xlist : [ [np.array(b x d1),...,np.array(b x dM)] , ...] list of multivews features ; b batch size ; dm dimension of views m ; M number of views   
            - Wlist : [np.array(b x b), ... ] list of weighted adjacency matrix
            - model : trained model MVGCCA    
            - path : str
            - epochs (which epochs is saved): str
    """
    allZ_list = [[] for x in Xlist[0]]
    allZ_list.append([])
    for X, W in zip(Xlist,Wlist):
        tamponZ_list = get_mvgcca_latents_space(X, W, model)
        for i, Z in enumerate(tamponZ_list):
            allZ_list[i].append(Z)
    concatZ_list = [np.concatenate(Z_list,axis = 0) for Z_list in allZ_list]
    key = ['t'+str(s) for s in range(len(concatZ_list))]
    dictionary = dict(zip(key, concatZ_list))
    sio.savemat(path+"latent_space_"+str(epochs+1)+'epochs.mat',dictionary) 
    return concatZ_list[0]

def train(data,  parameters, data_test = {}):
    """ Train the model given multiviews data and specified parameters and return it. """
    X = data["X"]
    W = data["W"]
    # Model parameters
    latent_dim = parameters["latent_dim"]
    gamma  = parameters["gamma"] 
    num_of_layer = parameters["num_of_layer"]
    hidden_layer = parameters["hidden_layer"]
    keep_prob =  parameters["keep_prob"]
    views_dropout_max =  parameters["views_dropout_max"]
    encoder_nn_type = parameters["encoder_nn_type"]
    encoder_use_common_hidden_layer = parameters["encoder_use_common_hidden_layer"],
    decoder_scalar_std = parameters["decoder_scalar_std"]
    # Training parameters
    learning_rate = parameters["learning_rate"]
    decay_learning_rate = parameters["decay_learning_rate"]
    num_of_epochs = parameters["num_of_epochs"]
    batch_size = parameters["batch_size"]
    # Wite parameteers
    write_loss = parameters["write_loss"]
    write_loss_path = parameters["write_loss_path"]
    write_latent_space = parameters["write_latent_space"]
    write_latent_space_interval = parameters["write_latent_space_interval"]
    latent_space_path = parameters["latent_space_path"]
    # write_weights = parameters["write_weights"]
    weights_path = parameters["weights_path"] 
    model_mvgcca = Mvgcca(latent_dim = latent_dim,
                          gamma = gamma,
                          num_of_layer = num_of_layer,
                          hidden_layer = hidden_layer,
                          keep_prob = keep_prob,
                          encoder_nn_type = encoder_nn_type,
                          encoder_use_common_hidden_layer = encoder_use_common_hidden_layer,
                          decoder_scalar_std = decoder_scalar_std,
                          views_dropout_max = views_dropout_max)
    writer = tf.summary.create_file_writer(write_loss_path)
    with writer.as_default():
        for e in range(num_of_epochs):
            print( "epochs : " + str(e) + "/" + str(num_of_epochs))
            batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = batch_size, shuffle = True)
            optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
            acc_loss_kl, acc_loss_ce = 0, 0 # ELBO component -> kl : kullbackleibler , ce : cross entropy (features + graph)
            for views, adj, s in zip( tqdm( batch_views, unit = "batch", disable = False ), batch_adj, batch_s ):
                with tf.GradientTape() as tape:
                    loss_kl, loss_ce = model_mvgcca(views,s,adj)
                    loss = loss_kl + loss_ce
                gradients = tape.gradient(loss, model_mvgcca.trainable_variables)
                gradient_variables = zip(gradients, model_mvgcca.trainable_variables)
                optimizer.apply_gradients(gradient_variables)
                acc_loss_kl += loss_kl
                acc_loss_ce += loss_ce
            if decay_learning_rate :
                optimizer.learning_rate = learning_rate * np.math.pow(1.1, - 50.*(e / num_of_epochs))
            avg_loss = ( acc_loss_kl + acc_loss_ce ) / len(batch_views)
            print( "avg_loss : " + str( avg_loss.numpy() ) )
            if write_loss :
                tf.summary.scalar("loss_kl", acc_loss_kl / len(batch_views), step=e)
                tf.summary.scalar("loss_ce", acc_loss_ce / len(batch_views), step=e)
                tf.summary.scalar("loss", avg_loss , step=e)
                writer.flush()
            else :
                os.system("rm -r -f events*")
            if ( ( e + 1 ) % write_latent_space_interval == 0 or e == num_of_epochs - 1 ) and write_latent_space :
                if data_test == {}:
                    save_mvgcca_latents_space(X, W, model_mvgcca, path = latent_space_path, prefix = '', epochs = e)
                else:
                    Xtest = data_test["X"]
                    Wtest = data_test["W"]
                    save_multiple_mvgcca_latents_space([X, Xtest], [W,Wtest], model_mvgcca, path = latent_space_path, epochs = e)
    return model_mvgcca

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='uci10robustclassifv2', help='Task to execute. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--device', default='0', help='Index of the target GPU. Specify \'-1\' to disable gpu support.')
    parser.add_argument('--use_graph_decoder', type = str2bool, default= True, help='True or False. Decide whether or not to use graph reconstruction term in loss.')
    parser.add_argument('--decoder_scalar_std', type = str2bool, default = True, help='True or False. Decide whether or not to use scalar matrix as covariance matrix for gaussian decoder.')
    parser.add_argument('--encoder_nn_type', default='krylov-4', help='Encoders neural networks. Only Mlp and Krylov-(Deep) is available. Example : krylov-4. ')
    parser.add_argument('--encoder_use_common_hidden_layer', type = str2bool, default = True, help='True or False. Whether or not to use differents hidden layers to calculate the mean and variance of encoders')
    parser.add_argument('--num_of_layer', type = int, default= 4, help='Number of layer for encoders AND decoders.')
    parser.add_argument('--hidden_layer', type = int, default= 1024, help='Size of hidden layer for encoders AND decoders.')
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help = 'Learning rate.' )
    parser.add_argument('--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate.')
    parser.add_argument('--num_of_epochs', type = int, default = 600, help='Number of epochs.')
    parser.add_argument('--batch_size', type = int, default = 512, help='Batch size.')
    parser.add_argument('--dropout', type = float, default = 0.5, help='Dropout rate applied to every hidden layers.')
    parser.add_argument('--views_dropout_max', type = int, default = 5, help='Integer. Views dropout is disable if null. Otherwise, for each epoch the numbers of views ignored during encoding phase is sample between 1 and MIN(views_dropout_max,nb_of_views - 1)')
    parser.add_argument('--latent_dim', type = int, default = 3, help='Dimension of latent space.')
    parser.add_argument('--write_weights', type = str2bool, default = False, help='True or False. Decide whether or not to write model weights.')
    parser.add_argument('--write_loss', type = str2bool, default = False, help='True or False. Decide whether or not to write loss training of model.')
    parser.add_argument('--write_latent_space', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights.')
    parser.add_argument('--write_latent_space_interval', type = int, default = 100, help='If --write_latent_space True : epochs interval between two saves of latent space. The last epoch is always saved.')
    parser.add_argument('--grid_search', type = str2bool, default = False, help='True or False. Decide whether or not to process a grid search.')
    parser.add_argument('--num_of_run', type = int, default = 1, help='Number of times the algorithm is runned for each set of parameters.')
    parser.add_argument('--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation works only for uci dataset related task and depends on the task selected). If option --num_of_run > 1, average evaluation of these run is returned for task =[uci7,uci10] ; while each run evaluation is returned for task=[uci7robustinf, uci10robustinf, uci7robustclassif, uci10robustclassif, uci10robustclassifv2,uci7robustclassifv2].')
    args = parser.parse_args()
    device = '/cpu:0' if args.device == '-1' or args.device == '' else '/gpu:'+args.device
    parameters = {}
    parameters["task"] = [args.task] 
    parameters["gamma"] = [1] if args.use_graph_decoder else [0]
    parameters["decoder_scalar_std"] = [args.decoder_scalar_std]
    parameters["encoder_nn_type"] = [args.encoder_nn_type]
    parameters["encoder_use_common_hidden_layer"] = [args.encoder_use_common_hidden_layer]
    parameters["num_of_layer"] = [args.num_of_layer]
    parameters["hidden_layer"] = [args.hidden_layer]
    parameters["learning_rate"] = [args.learning_rate]
    parameters["decay_learning_rate"] = [args.decay_learning_rate]
    parameters["num_of_epochs"] = [args.num_of_epochs]
    parameters["batch_size"] = [args.batch_size]
    parameters["keep_prob"] = [1 - args.dropout]
    parameters["views_dropout_max"] = [args.views_dropout_max]
    parameters["latent_dim"] = [args.latent_dim]
    parameters["write_loss"] = [args.write_loss]
    parameters["write_latent_space"] = [True if args.write_latent_space or args.grid_search  else False ] 
    parameters["write_latent_space_interval"] = [args.write_latent_space_interval]
    parameters["write_weights"] =  [args.write_weights] 
    parameters["evaluation"] = [args.evaluation] 
    if args.grid_search :
        parameters["learning_rate"] = [1e-3,1e-4]
        parameters["num_of_layer"] = [3,4]
        parameters["decay_learning_rate"] = [True,False]
        parameters["hidden_layer"] = [512, 1024]
        parameters["decoder_scalar_std"] = [True]
    if args.task in  process_data.available_tasks():
        if "uci" in  args.task:
            data = process_data.load_dataset(args.task)
            data = process_data.preprocess_data(data)
            data_val = {}
            data_test = {}
        if "robust" in args.task :
            data, data_test = process_data.split_data_intwo(data, cut = 0.9)
            data = process_data.preprocess_data(data)
            data_test = process_data.preprocess_data(data_test)
        if "mnist" in  args.task:
            data = process_data.load_dataset(args.task)
            data, data_test = process_data.split_data_intwo(data, cut = 5.0/7, shuffle=False)
            data = process_data.preprocess_data(data)
            data_test = process_data.preprocess_data(data_test)
        with tf.device(device):
            for parameters_, parameters_id in zip(list(ParameterGrid(parameters)),range(len(list(ParameterGrid(parameters))))):
                success = True
                for run_id in range(args.num_of_run):
                    if args.task == "tfr" :
                        data = process_data.load_dataset(args.task+str(run_id))
                        data = process_data.preprocess_data(data)
                        data_val = {}
                        data_test = {}
                    parameters_ = process_data.update_path(parameters_, args.task, NOW, parameters_id, run_id)
                    restart = 0
                    while restart < MAX_RESTART and success == True:
                        restart += 1
                        try :
                            model_trained = train(data, parameters_, data_test)
                        except Exception as e:
                            print(e.__doc__)
                            try:
                                print(e)
                            except:
                                pass                                
                            logging.error(traceback.format_exc())
                            if restart == MAX_RESTART :
                                success = False
                    # model_trained.save_weights(parameters["weights_path"])
                    result_tab = []
                    if args.evaluation :
                        if args.task == "uci7" or args.task == "uci10":
                            X = data["X"]
                            W = data["W"]
                            Z_list = get_mvgcca_latents_space(X, W, model_trained)
                            Z = Z_list[0]
                            nb_clusters = data['nb_clusters']
                            labels = process_data.get_uci_labels(nb_clusters = nb_clusters)
                            print("Parameters: " +str(parameters_))
                            display_bool = run_id == args.num_of_run -1
                            result_tab.append(process_evaluation.evaluate_clustering(Z,labels=labels,nb_clusters = nb_clusters, display_graph_with_tsne = False, display_score = False, write_score = True, write_path = parameters_["evaluation_path"]))
                            if run_id == args.num_of_run - 1 :
                                print("Parameters: " +str(parameters_))
                                print("Average results")
                                average_result = np.mean(result_tab, axis = 0)
                                print("Kmeans Clustering Score on UCI7: "+str(average_result[0]))
                                print("Spectral Clustering Score on UCI7: "+str(average_result[1]))
                                print("Svm-rbf Accuracy on UCI7: "+str(average_result[2]))
                        if args.task == "uci7robustclassif" or args.task == "uci10robustclassif":
                            nb_clusters = int(args.task.split('uci')[-1].split('robust')[0])
                            Xtrain = data["X"]
                            Wtrain = data["W"]
                            Xtest = data_test["X"]
                            Wtest = data_test["W"]
                            nb_views = len(Xtest)
                            views_id = list(np.arange(nb_views))
                            results_svm = []
                            for i in tqdm( np.flip(np.arange(0,nb_views)),unit="evaluation") :
                                list_of_keeped_views = process_data.get_all_sublist( set = views_id , size_of_subset = i + 1)
                                if list_of_keeped_views == [[]]:
                                    list_of_keeped_views = [views_id]                                
                                res_svm = 0
                                for keeped_views in list_of_keeped_views:
                                    Ztrain = get_mvgcca_latents_space(Xtrain,Wtrain,model_trained)[0]
                                    Ztest = get_mvgcca_common_latent_space_from_someviews(Xtest,Wtest,model_trained,views_id = keeped_views)
                                    res_svm  += (  process_evaluation.svm_fit_rbf(Ztrain,data["labels"],Ztest,data_test["labels"], nbclass = nb_clusters ) ) / len(list_of_keeped_views)  
                                results_svm.append(res_svm)
                            for i in np.arange(0,nb_views):
                                print( "Svm classification results" )
                                print("average for "+str(i)+" missing views : "+str(results_svm[i]))
                            # Reconstruction's visualisation of views 5th from differents number of views
                            # all_pics = [np.reshape(Xtest[4][:10],[10,16,15])]
                            # for views_id in [[0,1,2,3,5],[0,1,2,5],[0,1,5],[0,1],[0]] :  
                            #     recon = get_reconstruct_views_from_someviews(Xtest, Wtest, model_trained, views_id)
                            #     all_pics.append(np.reshape(recon[4][:10],[10,16,15]))
                            # process.imshow_grid(np.concatenate(all_pics,axis = 0), shape = [6,10])  
                        if args.task == "uci7robustclassifv2" or args.task == "uci10robustclassifv2":
                            nb_clusters = int(args.task.split('uci')[-1].split('robust')[0])
                            Xtrain = data["X"]
                            Wtrain = data["W"]
                            Xtest = data_test["X"]
                            Wtest = data_test["W"]
                            nb_views = len(Xtest)
                            views_id = list(np.arange(nb_views))
                            results_svm = []
                            drop_percents = [1, 5, 10, 15 , 25, 50, 75 ]
                            for drop_percent in tqdm(drop_percents):
                                list_of_keeped_views = []
                                list_of_keeped_views_W = []
                                for a in range(10):
                                    tampon = [[i for i in range(nb_views)] for j in range(len(Xtest[0]))]
                                    [np.random.shuffle(t) for t in tampon]
                                    tampon = [[] for j in range(len(Xtest[0]))]
                                    for i in range(len(Xtest[0])) :
                                        for j in range(nb_views):
                                            if np.random.rand(1) > drop_percent/100.0 :
                                                tampon[i].append(j)
                                        if tampon[i] == []:
                                            tampon[i].append([np.random.randint(nb_views)])
                                    list_of_keeped_views.append( [ np.sort(t) for t in tampon] )
                                    list_of_keeped_views_W.append([ np.copy(Wtest), np.copy(Wtest) ,np.copy(Wtest) ,np.copy(Wtest) ,np.copy(Wtest) ,np.copy(Wtest) ])
                                    for i in range(len(list_of_keeped_views[-1])) :
                                        for j in range(nb_views):
                                            if j not in list_of_keeped_views[-1][i] :
                                                list_of_keeped_views_W[-1][j][i,:] = 0
                                                list_of_keeped_views_W[-1][j][:,i] = 0
                                                list_of_keeped_views_W[-1][j][i,i] = 1
                                res_svm = 0
                                for keeped_views in list_of_keeped_views:
                                    Ztrain = get_mvgcca_latents_space(Xtrain,Wtrain,model_trained)[0]
                                    Ztest = get_mvgcca_common_latent_space_from_someviews2(Xtest,list_of_keeped_views_W[-1],model_trained,views_id = keeped_views)
                                    res_svm  += (  process_evaluation.svm_fit_rbf(Ztrain,data["labels"],Ztest,data_test["labels"], nbclass = nb_clusters ) ) / len(list_of_keeped_views)  
                                results_svm.append(res_svm)
                            for i in range(len(drop_percents)):
                                print( "Svm classification results" )
                                print("average for "+str(drop_percents[i])+" percenage missing views : "+str(results_svm[i]))
                            print(results_svm)
                            # Reconstruction's visualisation of views 5th from differents number of views
                            # all_pics = [np.reshape(Xtest[4][:10],[10,16,15])]
                            # for views_id in [[0,1,2,3,5],[0,1,2,5],[0,1,5],[0,1],[0]] :  
                            #     recon = get_reconstruct_views_from_someviews(Xtest, Wtest, model_trained, views_id)
                            #     all_pics.append(np.reshape(recon[4][:10],[10,16,15]))
                            # process.imshow_grid(np.concatenate(all_pics,axis = 0), shape = [6,10])  
                        if args.task == "uci7robustinf" or args.task == "uci10robustinf":
                            nb_clusters = data['nb_clusters']
                            Xtrain = data["X"]
                            Wtrain = data["W"]
                            Xtest = data_test["X"]
                            Wtest = data_test["W"]
                            nb_views = len(Xtest)
                            views_id = list(np.arange(nb_views))
                            results_inference = []
                            for i in tqdm( np.flip(np.arange(0,nb_views)),unit="evaluation") :
                                list_of_keeped_views = process_data.get_all_sublist( set = views_id , size_of_subset = i + 1)
                                if list_of_keeped_views == [[]]:
                                    list_of_keeped_views = [views_id]
                                res_inf = 0
                                for keeped_views in list_of_keeped_views:
                                    recon = get_reconstruct_views_from_someviews(Xtest, Wtest, model_trained, keeped_views)
                                    res_inf += (   np.sum(np.array(data_test["labels"])-recon[-1].argmax(axis=1) == 0)/len(Xtest[-1])     ) / len(list_of_keeped_views)
                                results_inference.append(res_inf)
                            for i in np.arange(0,nb_views):
                                print( "Inference classification results")
                                print("average for "+str(i)+" missing views : "+str(results_inference[i]))     
                if args.write_latent_space or args.evaluation or args.write_loss or args.write_weights:
                    status = "_success" if success else "_fail"
                    os.system('mv '+parameters_["parameters_main_path"]+' '+parameters_["parameters_main_path"]+status)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
    print("Fin.")
