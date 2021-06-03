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

from scipy.spatial.distance import cosine


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

def evaluate_friend_recommendation(X, X_id, dict_hashtag_users_, display_score = True, display_name =  "Method", ref_size = 5,  l_value = 35):
    dict_hashtag_users = dict_hashtag_users_.copy()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)   
    users_ref = []
    users_ref2 = []
    for key in dict_hashtag_users.keys():
        np.random.shuffle(dict_hashtag_users[key])
        users_ref.append(dict_hashtag_users[key][:ref_size])
        users_ref2 = users_ref2 + list(dict_hashtag_users[key][:ref_size])
        dict_hashtag_users[key] = dict_hashtag_users[key][ref_size:]
    # Xnew = []
    # Xnew_id = []
    # for k in range(len(X_id)) : 
    #     if not X_id[k] in users_ref2:
    #         Xnew.append(X[k,:])
    #         Xnew_id.append(X_id[k])
    Xref = []
    for tab in users_ref:
        tampon = []
        for id_ in tab :
            index = np.where(id_ == X_id)[0][0]
            tampon.append(X[index,:])
        Xref.append(np.mean(tampon,axis=0))
    distances = []
    for ref in Xref:
        distance = []
        for k in range(len(X)):
            distance.append(cosine(ref,X[k][:]))
        distances.append(distance)
    distances_order = [np.argsort(distance) for distance in distances]
    k = 0
    dict_classif = {}
    for key in dict_hashtag_users.keys():
        dict_classif.update({key : []})
        for l in distances_order[k]:
            dict_classif[key].append(X_id[l])
        k = k + 1
    L = l_value
    precision = 0
    recall = 0
    mrr = 0
    integer = 0
    for key in dict_hashtag_users.keys() :
        if not len(dict_hashtag_users[key]) == 0 :
            precision += len(np.intersect1d(dict_hashtag_users_[key],dict_classif[key][:L]))/L
            recall += len(np.intersect1d(dict_hashtag_users_[key],dict_classif[key][:L]))/len(dict_hashtag_users_[key])
            _, _, first_indices =  np.intersect1d(dict_hashtag_users_[key],dict_classif[key],return_indices = True)
            if len(first_indices) > 0 :
                mrr += 1./(min(first_indices)+1)#[0]
            integer += 1
    precision = precision/float(integer)
    recall = recall/float(integer)
    mrr = mrr/integer
    if display_score :
        print("Precison "+display_name+" on TEP: "+str(precision))
        print("Recall "+display_name+" on TEP: "+str(recall))
        print("Mrr "+display_name+" on TEP: "+str(mrr))
    return precision, recall, mrr
