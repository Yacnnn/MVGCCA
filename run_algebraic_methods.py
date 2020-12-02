
import argparse
import numpy as np
from utils import process_data
from utils import process_evaluation

from models import algebraic
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV

def available_tasks():
    return ["uci7",'uci10','tfr']

def available_methods():
    return ["pca",'gpca','mcca','gmcca']

def evaluate_clustering_gpca(data, latent_dim = 3, beta_list = [0], nb_run = 1):
    Xconcat = np.concatenate(data["X"],axis = 1)
    W = data["W"]
    labels = data["labels"]
    nb_clusters = data["nb_clusters"]
    Xpca_tab = []
    labels = np.array(labels)
    for beta in tqdm( beta_list, unit = 'beta' ):
        Xpca, _ =  algebraic.gpca(Xconcat, W, beta, latent_dim)
        Xpca_tab.append(Xpca)
    process_evaluation.evaluate_clustering_cross_validation(Xpca_tab, labels, nb_clusters = nb_clusters, nb_run = nb_run)

def evaluate_clustering_gmcca(data, latent_dim = 3, gamma_list = [0], nb_run = 1):
    X = np.concatenate(data["X"],axis = 1)
    W = data["W"]
    labels = data["labels"]
    nb_clusters = data["nb_clusters"]
    Stmcca_tab = []
    labels = np.array(labels)
    for gamma in tqdm(gamma_list):
        Stmcca, _, _ = algebraic.gmmca(X,W,gamma,n_components = latent_dim)  
        Stmcca_tab.append(Stmcca)
    process_evaluation.evaluate_clustering_cross_validation(Stmcca_tab, labels, nb_clusters = nb_clusters, nb_run = nb_run)

def evaluate_uci(method):
    if method == "pca":
        beta_list = [0]
        evaluate_clustering_gpca(data, latent_dim = 3, beta_list = beta_list, nb_run = 10)
    if method == "gpca":
        beta_list = list(np.logspace(-2,0,20))
        evaluate_clustering_gpca(data, latent_dim = 3, beta_list = beta_list, nb_run = 100)
    elif method == "mcca":
        gamma_list = [0]
        evaluate_clustering_gmcca(data, latent_dim = 3, gamma_list = gamma_list, nb_run = 10)
    elif method == "gmcca":
        gamma_list = list(np.logspace(-2,0,20))
        evaluate_clustering_gmcca(data, latent_dim = 3, gamma_list = gamma_list, nb_run = 100)

def evaluate_tfr(method):
    print("TO DO.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='pca', help='Method to execute. Only %s are currently available.'%str(available_methods()))
    parser.add_argument('--task', default='uci7', help='Task to execute. Only %s are currently available.'%str(available_tasks()))
    args = parser.parse_args()

    if args.task in available_tasks():
        data = process_data.load_dataset(args.task)
        if args.method in available_methods():
            if "uci" in args.task:
                evaluate_uci(args.method)
            elif "tfr" in args.task:
                evaluate_tfr(args.method)
        else:
            print('Unknown method %s'%args.method)
            parser.print_help()
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
print("Fin.")
