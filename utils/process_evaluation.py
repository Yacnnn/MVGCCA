import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from utils import process_data
import scipy.io as sio

from grakel.kernels import  WeisfeilerLehmanOptimalAssignment
from grakel.kernels import  WeisfeilerLehman
from grakel.kernels import PropagationAttr

from sklearn.model_selection import train_test_split


def svm_fit_rbf(ZTrain,labelsTrain,ZTest,labelsTest,nbclass = 2 ):
    clf = SVC(gamma='auto',kernel="rbf", C = 1)
    clf.fit(ZTrain,labelsTrain)
    result = clf.score(ZTest,labelsTest)
    return result

def evaluate_clustering(Zo, labels, nb_clusters = 7, display_score = True, display_graph_with_tsne = True, display_name = '',  write_score = False, write_path = ''):
    Z = np.copy(Zo)
    scaler = StandardScaler()
    scaler.fit(Z)
    Z = scaler.transform(Z)
    
    kmeans_ = KMeans(n_clusters=nb_clusters, random_state=0).fit(Z)
    kmeans_clustering_score = adjusted_rand_score(kmeans_.labels_,labels)
    spectral_ = SpectralClustering(n_clusters=nb_clusters,assign_labels="discretize", random_state=0, gamma=5).fit(Z)
    spectral_clustering_score = adjusted_rand_score(spectral_.labels_,labels)
    clf = SVC(kernel = "rbf")
    svm_accuracy = np.mean(cross_val_score(clf, Z, np.array(labels), cv=5))
    
    if display_score :
        print("Kmeans Clustering Score "+display_name+" on UCI: "+str(kmeans_clustering_score))
        print("Spectral Clustering Score "+display_name+" on UCI: "+str(spectral_clustering_score))
        print("Svm-rbf Accuracy "+display_name+" on UCI: "+str(svm_accuracy))
        
    if display_graph_with_tsne :
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
        Z_tsne = tsne.fit_transform(Z)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#be4922']
        color_list =  []
        for k in range(nb_clusters):
            for l in range(200):
                color_list.append(colors[k])  
        f1 = plt.figure()
        plt.title(display_name)
        plt.scatter(Z_tsne[:, 0],  Z_tsne[:, 1], s=10,c=color_list);
        plt.show()

    if write_score :
        sio.savemat(write_path+"clustering_results.mat",mdict = {
            "Kmeans Clustering Score" : kmeans_clustering_score,
            "Spectral Clustering Score" : spectral_clustering_score,
            "Svm-rbf Accuracy" : svm_accuracy
        })
    return [kmeans_clustering_score, spectral_clustering_score, svm_accuracy]

def evaluate_clustering_cross_validation(Zo_tab, labels, nb_clusters = 7, nb_run = 10, display_score = True, display_name = ''):#, display_graph_with_tsne = True, display_name = "Title",  write_score = False, write_path = ''):

    svm_accuracy_acc =  0
    clustering_score_acc = 0
    spect_clustering_score_acc = 0
    labels_original = labels
    for run in range(nb_run):
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

        ZTrain_tab = [Z[train_cut:] for Z in Z_tab]#Z_tab
        ZTest_tab = [Z[:train_cut] for Z in Z_tab]#Z_tab

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

        # final_clf =  SVC(gamma = best_parameters["gamma"], kernel = best_parameters["kernel"], C = best_parameters["C"])
        # svm_accuracy = np.mean(cross_val_score(final_clf, bestbeta_Z,labels, cv=10))

        kmeansgpca = KMeans(n_clusters=nb_clusters, random_state=0).fit(bestbeta_Z)
        clustering_score = adjusted_rand_score(kmeansgpca.labels_,labels)

        spectral = SpectralClustering(n_clusters=nb_clusters,assign_labels="discretize", random_state=0, gamma=5).fit(bestbeta_Z)
        spect_clustering_score = adjusted_rand_score(spectral.labels_,labels)

        # if display_score :
        #     print("Svm-rbf Accuracy "+display_name+" on UCI: "+str(svm_accuracy))
        #     print("Clustering Score "+display_name+" on UCI: "+str(clustering_score))
        #     print("Spec Clustering Score "+display_name+" on UCI: "+str(spect_clustering_score))
        #     print("best_parameters "+display_name+" on UCI: "+str(best_parameters))

        svm_accuracy_acc +=  svm_accuracy/nb_run
        clustering_score_acc += clustering_score/nb_run
        spect_clustering_score_acc += spect_clustering_score/nb_run

    if display_score :
        print("Svm-rbf Accuracy "+display_name+" on UCI: "+str(svm_accuracy_acc))
        print("Clustering Score "+display_name+" on UCI: "+str(clustering_score_acc))
        print("Spec Clustering Score "+display_name+" on UCI: "+str(spect_clustering_score_acc))

    return svm_accuracy_acc, clustering_score_acc, spect_clustering_score_acc
