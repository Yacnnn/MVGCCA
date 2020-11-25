import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from utils import process_data
import scipy.io as sio

from grakel.kernels import  WeisfeilerLehmanOptimalAssignment
from grakel.kernels import  WeisfeilerLehman
from grakel.kernels import PropagationAttr

from sklearn.model_selection import train_test_split


def svm_fit_rbf(ZXtrain,ZYtrain,ZXtest,ZYtest,nbclass = 2 ):
    clf = SVC(gamma='auto',kernel="rbf", C = 1)
    clf.fit(ZXtrain,ZYtrain)
    result = clf.score(ZXtest,ZYtest)
    return result

def evaluate_clustering(Zo, index, nb_clusters = 7, display_score = True, display_graph_with_tsne = True, display_name = "Title",  write_score = False, write_path = ''):
    Z = np.copy(Zo)
    scaler = StandardScaler()
    scaler.fit(Z)
    Z = scaler.transform(Z)
    
    kmeans_ = KMeans(n_clusters=nb_clusters, random_state=0).fit(Z)
    kmeans_clustering_score = adjusted_rand_score(kmeans_.labels_,index)
    spectral_ = SpectralClustering(n_clusters=nb_clusters,assign_labels="discretize", random_state=0, gamma=5).fit(Z)
    spectral_clustering_score = adjusted_rand_score(spectral_.labels_,index)
    clf = SVC(kernel = "rbf")
    svm_accuracy = np.mean(cross_val_score(clf, Z, np.array(index), cv=5))
    
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

