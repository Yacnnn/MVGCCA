#%%
import os  
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
from utils import process_data
from utils import process_evaluation
from models.mvgcca import Mvgcca
from sklearn.model_selection import ParameterGrid

NOW = datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')
MAX_RESTART = 1

def str2bool(string):
    if string in ["True","true","1"]:
        return True
    elif string in ["False","false","0"]:
        return False
    else :
        return False

def create_views_batch_size(original_views, w = None, batch_size = 32, shuffle = True):
    views = [np.copy(original_view) for original_view in original_views]
    n = views[0].shape[0]
    s = np.arange(n) 
    if shuffle:
        np.random.shuffle(s)
        if not np.any( w == None ) :
            w = w[s,:]
            w = w[:,s]
        for i, view in enumerate(views) :
            views[i] = view[s]  
    q = n//batch_size
    block_end = q*batch_size    
    batch_indice = [ [k*batch_size,(k+1)*batch_size]  for k in range(q)] + [[q*batch_size,n]]
    batch_views = [ [ view[ind[0]:ind[1]] for view in views ] for ind in batch_indice]
    batch_adj = []
    if not np.any(w == None):
        for ind in batch_indice:
            a = w[ind[0]:ind[1],:]
            a = a[:,ind[0]:ind[1]]
            batch_adj.append(a)
    else :
        for ind in batch_indice:
            w = np.eye(ind[1]-ind[0])
            batch_adj.append(w)
    batch_s = [  s[ind[0]:ind[1]]  for ind in batch_indice]
    if batch_views[-1][0].shape[0] == 0 :
        return batch_views[:-1], batch_adj[:-1], batch_s[:-1]
    return batch_views, batch_adj, batch_s

def get_mvgcca_latent_space(X, W, model):
        batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = X[0].shape[0], shuffle=False)
        U = model.get_latent_space(batch_views[0],W)
        return U

def save_mvgcca_latent_space(X, W, model, path, epochs):
        U = get_mvgcca_latent_space(X, W, model)
        key = ['t'+str(s) for s in range(len(U))]
        dictionary = dict(zip(key, U))
        sio.savemat(path+"latent_space_"+str(epochs+1)+'epochs.mat',dictionary)  
             
def train(X, W, parameters):
    # Model parameters
    latent_dim = parameters["latent_dim"]
    gamma  = parameters["gamma"] 
    num_of_layer = parameters["num_of_layer"]
    hidden_layer = parameters["hidden_layer"]
    keep_prob =  parameters["keep_prob"]
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
    latent_space_path = parameters["latent_space_path"]
    write_weights = parameters["write_weights"]
    weights_path = parameters["weights_path"] 
    model_mvgcca = Mvgcca(latent_dim = latent_dim,
                          gamma = gamma,
                          num_of_layer = num_of_layer,
                          hidden_layer = hidden_layer,
                          keep_prob = keep_prob,
                          encoder_nn_type = encoder_nn_type,
                          encoder_use_common_hidden_layer = encoder_use_common_hidden_layer,
                          decoder_scalar_std = decoder_scalar_std)
    writer = tf.summary.create_file_writer(write_loss_path)
    with writer.as_default():
        for e in range(num_of_epochs):
            print( "epochs : " + str(e) + "/" + str(num_of_epochs))
            batch_views, batch_adj, batch_s = create_views_batch_size(X, W, batch_size = batch_size, shuffle = True)
            optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
            acc_loss_kl, acc_loss_ce = 0, 0
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
            if ( ( e + 1 ) % 100 == 0 or e == num_of_epochs - 1 ) and write_latent_space :
                save_mvgcca_latent_space(X, W, model_mvgcca, path = latent_space_path, epochs = e)
    return model_mvgcca

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='uci7', help='Task to execute. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--device', default='', help='Index of the target GPU. Specify \'-1\' to disable gpu support.')
    parser.add_argument('--use_graph_decoder', type = str2bool, default= True, help='True or False. Decide whether or not to use graph reconstruction term in loss.')
    parser.add_argument('--decoder_scalar_std', type = str2bool, default = True, help='True or False. Decide whether or not to use scalar matrix as covariance matrix for gaussian decoder.')
    parser.add_argument('--encoder_nn_type', default='krylov-4', help='Encoders neural networks. Only Krylov-(Deep) is available. Example :krylov-4. ')
    parser.add_argument('--encoder_use_common_hidden_layer', type = str2bool, default = True, help='True or False. Whether or not to use different hidden layers to calculate the mean and variance of encoders')
    parser.add_argument('--num_of_layer', type = int, default= 4, help='Number of layer for encoders AND decoders.')
    parser.add_argument('--hidden_layer', type = int, default= 1024, help='Size of hidden layer for encoders AND decoders.')
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help = 'Learning rate.' )
    parser.add_argument('--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate.')
    parser.add_argument('--num_of_epochs', type = int, default = 1, help='Number of epochs.')
    parser.add_argument('--batch_size', type = int, default = 512, help='Batch size.')
    parser.add_argument('--dropout', type = float, default = 0.3, help='Dropout rate applied to every hidden layers.')
    parser.add_argument('--latent_dim', type = int, default = 20, help='Dimension of latent space.')
    parser.add_argument('--write_weights', type = str2bool, default = False, help='True or False. Decide whether or not to write model weights.')
    parser.add_argument('--write_loss', type = str2bool, default = False, help='True or False. Decide whether or not to write loss training of model.')
    parser.add_argument('--write_latent_space', type = str2bool, default = False, help='True or False. Decide whether or not to write model weights.')
    parser.add_argument('--grid_search', type = str2bool, default = False, help='True or False. Decide whether or not to process a grid search.')
    parser.add_argument('--num_of_run', type = int, default = 1, help='Number of times the algorithm is runned.')
    parser.add_argument('--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation function depend on the task selected). If option --num_of_run > 1 average evaluation of these run is returned.')
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
    parameters["latent_dim"] = [args.latent_dim]
    parameters["write_loss"] = [args.write_loss]
    parameters["write_latent_space"] = [args.write_latent_space]
    parameters["write_weights"] = [args.write_weights]
    parameters["evaluation"] = [args.evaluation] 
    if args.grid_search :
        parameters["learning_rate"] = [1e-3,1e-4]
        parameters["num_of_layer"] = [3,4]
        parameters["decay_learning_rate"] = [True,False]
        parameters["hidden_layer"] = [512, 1024]
        parameters["decoder_scalar_std"] = [True, False]
    if args.task in  process_data.available_tasks():
        if not args.task == "tfr" and not args.task == "NCI1" and not args.task == "PROTEINS_full" and not args.task == "PROTEINS" and not args.task == "ENZYMES":
            X, W = process_data.load_dataset(args.task)
            X, W = process_data.preprocess_data(X, W)
        with tf.device(device): 
            for parameters_, parameters_id in zip(list(ParameterGrid(parameters)),range(len(list(ParameterGrid(parameters))))):
                success = True
                for run_id in range(args.num_of_run):
                    parameters_ = process_data.update_path(parameters_, args.task, NOW, parameters_id, run_id)
                    restart = 0
                    while restart < MAX_RESTART and success == True:
                        restart += 1
                        try :
                            model_trained = train(X, W, parameters_)
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
                        if args.task == "uci7":
                            Z_tab = get_mvgcca_latent_space(X, W, model_trained)
                            Z_common = Z_tab[0]
                            labels = process_data.get_uci_labels(nb_clusters=7)
                            print("Parameters: " +str(parameters_))
                            display_bool = run_id == args.num_of_run -1
                            result_tab.append(process_evaluation.evaluate_clustering(Z_common,labels=labels,nb_clusters = 7, display_graph_with_tsne = display_bool, display_score = False, write_score = True, write_path = parameters_["evaluation_path"]))
                            if run_id == args.num_of_run - 1 :
                                print("Parameters: " +str(parameters_))
                                print("Average results")
                                average_result = np.mean(result_tab, axis = 0)
                                print("Kmeans Clustering Score on UCI7: "+str(average_result[0]))
                                print("Spectral Clustering Score on UCI7: "+str(average_result[1]))
                                print("Svm-rbf Accuracy on UCI7: "+str(average_result[2]))    
                        elif args.task == "uci10":
                            Z_tab = get_mvgcca_latent_space(X, W, model_trained)
                            Z_common = Z_tab[0]
                            labels = process_data.get_uci_labels(nb_clusters=10)
                            print("Parameters: " +str(parameters_))
                            display_bool = run_id == args.num_of_run -1
                            result_tab.append(process_evaluation.evaluate_clustering(Z_common,labels=labels,nb_clusters = 10, display_graph_with_tsne = display_bool, display_score = False, write_score = True, write_path = parameters_["evaluation_path"]))
                            if run_id == args.num_of_run - 1 :
                                print("Parameters: " +str(parameters_))
                                print("Average results")
                                average_result = np.mean(result_tab, axis = 0)
                                print("Kmeans Clustering Score on UCI10: "+str(average_result[0]))
                                print("Spectral Clustering Score on UCI10: "+str(average_result[1]))
                                print("Svm-rbf Accuracy on UCI10: "+str(average_result[2])) 
                if args.write_latent_space or args.write_weights or args.evaluation or args.write_loss:
                    status = "_success" if success else "_fail"
                    os.system('mv '+parameters_["parameters_main_path"]+' '+parameters_["parameters_main_path"]+status)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
    print("Fin.")
