import argparse
import sys
import time
import glob
import numpy as np

from utils import process_data
from utils import process_evaluation

from models import algebraic
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score


def available_tasks():
    return ["uci7",'uci10']

def available_methods():
    return ["pca",'gpca','mcca','gmcca']

def evaluate_clustering_cross_validation_algebraic(Zo_tab, labels, nb_clusters = 7, averaging_times = 10, display_score = True, display_name = ''):
    svm_accuracy_acc =  0
    clustering_score_acc = 0
    spect_clustering_score_acc = 0

    labels_original = labels
    for run in range(averaging_times):
        labels = labels_original
        num_of_instance = Zo_tab[0].shape[0]
        s = np.arange(num_of_instance)
        np.random.shuffle(s)

        Z_tab = []
        for Z in Zo_tab:
            scaler = StandardScaler()
            scaler.fit(Z)
            Z_tab.append(scaler.transform(Z)[s,:])

        tuned_parameters = {'kernel': ['rbf'], 
                            'gamma': ["auto"]+list(np.logspace(-3,3,7)),
                                'C': [1]+list(np.logspace(-3,3,7))}

        train_cut = int(0.9*num_of_instance)
        labels = labels[s]
        labelsTrain = labels[train_cut:]
        labelsTest = labels[:train_cut]
        ZTrain_tab = [Z[train_cut:] for Z in Z_tab]
        ZTest_tab = [Z[:train_cut] for Z in Z_tab]

        svm_acc_tab = []
        parameters_tab = []
        for Z in ZTrain_tab:
            clf = GridSearchCV( SVC(), tuned_parameters, refit = True, scoring='accuracy',)
            clf.fit(Z, labelsTrain)
            svm_acc_tab.append(clf.best_score_)
            parameters_tab.append(clf.best_params_)

        ibest = np.argmax(svm_acc_tab)
        best_parameters = parameters_tab[ibest]

        bestbeta_ZTrain = ZTrain_tab[ibest]
        bestbeta_ZTest = ZTest_tab[ibest]
        bestbeta_Z = Z_tab[ibest]

        final_clf =  SVC(gamma = best_parameters["gamma"], kernel = best_parameters["kernel"], C = best_parameters["C"])
        final_clf.fit(bestbeta_ZTrain,labelsTrain)
        svm_accuracy = accuracy_score(final_clf.predict(bestbeta_ZTest),labelsTest)

        kmeansgpca = KMeans(n_clusters=nb_clusters, random_state=0).fit(bestbeta_Z)
        clustering_score = adjusted_rand_score(kmeansgpca.labels_,labels)

        spectral = SpectralClustering(n_clusters=nb_clusters,assign_labels="discretize", random_state=0, gamma=5).fit(bestbeta_Z)
        spect_clustering_score = adjusted_rand_score(spectral.labels_,labels)

        svm_accuracy_acc +=  svm_accuracy/averaging_times
        clustering_score_acc += clustering_score/averaging_times
        spect_clustering_score_acc += spect_clustering_score/averaging_times

    if display_score :
        print("Svm-rbf Accuracy "+display_name+" on UCI: "+str(svm_accuracy_acc))
        print("Clustering Score "+display_name+" on UCI: "+str(clustering_score_acc))
        print("Spec Clustering Score "+display_name+" on UCI: "+str(spect_clustering_score_acc))

    return svm_accuracy_acc, clustering_score_acc, spect_clustering_score_acc

def evaluate_clustering_gpca(data, latent_dim = 3, beta_list = [0], averaging_times = 1):
    Xconcat = np.concatenate(data["X"],axis = 1)
    W = data["W"]
    labels = data["labels"]
    nb_clusters = data["nb_clusters"]
    Xpca_tab = []
    labels = np.array(labels)
    for beta in tqdm( beta_list, unit = 'beta' ):
        Xpca, _ =  algebraic.gpca(Xconcat, W, beta, latent_dim)
        Xpca_tab.append(Xpca)
    evaluate_clustering_cross_validation_algebraic(Xpca_tab, labels, nb_clusters = nb_clusters, averaging_times = averaging_times)

def evaluate_clustering_gmcca(data, latent_dim = 3, gamma_list = [0], averaging_times = 1):
    X = data["X"]
    W = data["W"]
    labels = data["labels"]
    nb_clusters = data["nb_clusters"]
    Stmcca_tab = []
    labels = np.array(labels)
    for gamma in tqdm(gamma_list):
        Stmcca, _, _ = algebraic.gmmca(X,W,gamma,n_components = latent_dim)  
        Stmcca_tab.append(Stmcca)
    evaluate_clustering_cross_validation_algebraic(Stmcca_tab, labels, nb_clusters = nb_clusters, averaging_times = averaging_times)

def evaluate_uci(data, method):
    if method == "pca":
        beta_list = [0]
        evaluate_clustering_gpca(data, latent_dim = 3, beta_list = beta_list, averaging_times = 100)
    if method == "gpca":
        beta_list = list(np.logspace(-2,0,20))
        evaluate_clustering_gpca(data, latent_dim = 3, beta_list = beta_list, averaging_times = 100)
    elif method == "mcca":
        gamma_list = [0]
        evaluate_clustering_gmcca(data, latent_dim = 3, gamma_list = gamma_list, averaging_times = 100)
    elif method == "gmcca":
        gamma_list = list(np.logspace(-2,0,20))
        evaluate_clustering_gmcca(data, latent_dim = 3, gamma_list = gamma_list, averaging_times = 100)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='gmcca', help='Method to execute. Only %s are currently available.'%str(available_methods()))
    parser.add_argument('--task', default='uci7', help='Task to execute. Only %s are currently available.'%str(available_tasks()))
    args = parser.parse_args()

    if args.task in available_tasks():
        data = process_data.load_dataset(args.task)
        if args.method in available_methods():
            if "uci" in args.task:
                evaluate_uci(data, args.method)
        else:
            print('Unknown method %s'%args.method)
            parser.print_help()
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
print("Fin.")
