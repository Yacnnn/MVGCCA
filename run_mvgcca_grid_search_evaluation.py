import argparse
import numpy as np
import glob
import os
import scipy.io as sio
from utils import process_data
from utils import process_evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score ,normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans, SpectralClustering
from tqdm import tqdm

def available_tasks():
    """ Return list of available tasks. """
    return ["uci7",'uci10','tfr','mnist2views']

def get_parameters_folder_path(search_path):
    return glob.glob(search_path+'/*success')

def get_run_folder_path(parameter_path):
    run_folder_path = glob.glob(parameter_path+'/*run*')
    parameter_file_path = glob.glob(parameter_path+'/*dict*')[0]
    return run_folder_path, parameter_file_path

def rotate(list_, n):
    return list_[n:] + list_[:n]

def get_embedding_path(run_folder_path):
    return [rotate(sorted(glob.glob(r+'/embeddings/*.mat')),1) for r in run_folder_path]

def embeddings_path2embeddings(embeddings_path):
    return [[sio.loadmat(path)['t0'] for path in ep] for ep in embeddings_path]

def run_folder_path_to_data(run_folder_path):
    data = process_data.load_twitter_fr(loadIfExist=True,sample_id=int(run_folder_path.split("run")[-1]))
    return data

def evaluate_clustandclassif_cross_validation_uci(embeddings_list, labels, nb_clusters, averaging_times = 10):
    "Return accuracy following the method described in article."
    nb_run =   np.shape(embeddings_list)[1] 
    nb_early_stop = np.shape(embeddings_list)[2]
    svm_valid_acc_results = np.zeros((nb_early_stop)) 
    svm_test_acc_results = np.zeros((nb_early_stop)) 
    kmeans_clustering_results = np.zeros((nb_early_stop)) 
    spectral_clutering_reults = np.zeros((nb_early_stop)) 
    labels_original = labels
    for avgt in range(averaging_times):
        for e in range(nb_early_stop):
            svm_test_accuracy_acc =  0
            clustering_score_acc = 0
            spect_clustering_score_acc = 0
            for r in range(nb_run):
                Zo_tab  = list(np.array(embeddings_list)[:,r,e])
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
                labelsTrain = labels[:train_cut]
                labelsTest = labels[train_cut:]
                ZTrain_tab = [Z[:train_cut] for Z in Z_tab]
                ZTest_tab = [Z[train_cut:] for Z in Z_tab]
                svm_acc_tab = []
                parameters_tab = []
                for Z in ZTrain_tab:
                    clf = GridSearchCV( SVC(), tuned_parameters, refit = False, scoring='accuracy',)
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

                svm_valid_acc_results[e] += clf.best_score_/(nb_run*averaging_times)
                svm_test_acc_results[e]  += svm_accuracy/(nb_run*averaging_times)
                kmeans_clustering_results[e] += clustering_score/(nb_run*averaging_times)
                spectral_clutering_reults[e] += spect_clustering_score/(nb_run*averaging_times)

        ifinalbest = np.argmax(svm_valid_acc_results)
        final_svm_accuracy_acc = svm_test_acc_results[ifinalbest]
        final_clustering_score_acc = kmeans_clustering_results[ifinalbest]
        final_spect_clustering_score_acc = spectral_clutering_reults[ifinalbest]
        print("Svm-rbf Accuracy on UCI: "+str(final_svm_accuracy_acc))
        print("Clustering Score on UCI: "+str(final_clustering_score_acc))
        print("Spec Clustering Score on UCI: "+str(final_spect_clustering_score_acc))
    return final_svm_accuracy_acc, final_clustering_score_acc, final_spect_clustering_score_acc

def evaluate_twitter_friends_recommandation(embeddings_list,run_folder_path):
    "Return precison, recall, mrr for each parameters and each epochs saved and average it over the different ru (for each parameters)."
    run_folder_path = run_folder_path[:]#[:10]
    embeddings_list = embeddings_list#[:10]
    averaging_times = len(run_folder_path)
    precision = np.zeros(np.shape(embeddings_list)[:2])
    recall = np.zeros(np.shape(embeddings_list)[:2])
    mrr = np.zeros(np.shape(embeddings_list)[:2])
    for path, i in zip(run_folder_path,tqdm(range(len(run_folder_path)),unit="run_folders_processed")):
        _, X_id, _, dict_hashtag_users  = run_folder_path_to_data(path)
        for j in range( len(    embeddings_list[i]   )  ): 
            p, r, m = process_evaluation.evaluate_friend_recommendation( embeddings_list[i][j] ,X_id, dict_hashtag_users, display_score = False, ref_size = 50,  l_value = 500 )
            precision[i,j] = p
            recall[i,j] = r
            mrr[i,j] = m
    print("precision mean by epochs saved : ",np.mean(precision,axis = 0))
    print("recall mean by epochs saved: ",np.mean(recall,axis = 0))
    print("mrr mean by epochs saved: ",np.mean(mrr,axis = 0))
    return precision, recall, mrr

def evalute_classif_cross_validation_mnist2views(embeddings_list,labels,nb_clusters,run_folder_path):  
    "Return accuracy following the method described in article."
    nb_parameters_gridsearch = np.shape(embeddings_list)[0]
    nb_run = np.shape(embeddings_list)[1] 
    nb_early_stop = np.shape(embeddings_list)[2]
    svm_valid_acc_results = np.zeros((nb_run,nb_early_stop)) 
    svm_test_acc_results = np.zeros((nb_run,nb_early_stop)) 

    labels = labels - 1
    labels = [int(l) for l in labels]
    train_val_list  = [ [ [process_data.normalize_features(latent[:60000]) for latent in run] for run in parameters] for parameters in embeddings_list]
    test_list = [ [ [process_data.normalize_features(latent[60000:]) for latent in run] for run in parameters] for parameters in embeddings_list]
    labelsTrainVal = labels[:60000]
    labelsTest = labels[60000:]
    classif_tuned_parameters = {'kernel': ['linear'], 'C': [1]+list(np.logspace(-3,3,8))}
    for e in range(nb_early_stop):
        for r in range(nb_run):
            Ztrainval_tab  = list(np.array(train_val_list)[:,r,e])
            Ztest_tab  = list(np.array(test_list)[:,r,e])

            svm_acc_tab = []
            parameters_svm_acc_tab = []
            for Z in Ztrainval_tab:
                clf = GridSearchCV( SVC(), classif_tuned_parameters, refit = False, scoring='accuracy',cv = [([i for i in range(50000)],[i for i in range(50000,60000)])],n_jobs=8,verbose = 10)
                clf.fit(Z, labelsTrainVal)
                svm_acc_tab.append(clf.best_score_)
                parameters_svm_acc_tab.append(clf.best_params_)

            ibest = np.argmax(svm_acc_tab)
            best_parameters = parameters_svm_acc_tab[ibest]

            bestbeta_Ztrainval = Ztrainval_tab[ibest][:60000]
            bestbeta_ZTest = Ztest_tab[ibest]

            final_clf =  SVC(kernel = best_parameters["kernel"], C = best_parameters["C"])
            final_clf.fit(bestbeta_Ztrainval,labelsTrainVal)
            svm_accuracy = accuracy_score(final_clf.predict(bestbeta_ZTest),labelsTest)
            svm_valid_acc_results[r,e] += clf.best_score_
            svm_test_acc_results[r,e]  += svm_accuracy

            # sio.savemat(run_folder_path[r]+"/classiflinv2t1"+str(e)+"x"+str(r)+".mat", mdict = {
            #     "val_res" : svm_valid_acc_results,
            #     "test_res" : svm_test_acc_results
            # })

    ifinalbest = np.argmax(np.mean(svm_valid_acc_results, axis =  0))
    final_svm_acc = np.mean(svm_test_acc_results, axis =  0)[ifinalbest]
    print("Svm-rbf Accuracy on Mnist2views: "+str(final_svm_acc))
    return final_svm_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='mnist2views', help='Task to execute. Only %s are currently available.'%str(available_tasks()))
    parser.add_argument('--date', default='April_23_2021_01h05m53s', help='[MONTH]_[DAY]_[YEAR]_[HOUR]h[MINUTES]m[SECONDES]s')
    args = parser.parse_args()
    if args.task in available_tasks():
        search_path = '/scratch/ykaloga/'+'results/'+args.task+'/'+args.date+"/"
        if os.path.isdir(search_path):
            list_of_parameters = [] 
            embeddings_list  = []
            results = []
            parameters_folder_path = get_parameters_folder_path(search_path)
            for k in range(len(parameters_folder_path)):
                run_folder_path, parameter_file_path = get_run_folder_path(parameters_folder_path[k])
                embeddings_list.append( embeddings_path2embeddings(get_embedding_path(run_folder_path)) )
                list_of_parameters.append( sio.loadmat(parameter_file_path) )
            if "uci" in args.task :     
                data = process_data.load_dataset(args.task)
                labels = np.array(data['labels'])
                nb_clusters = data['nb_clusters']
                evaluate_clustandclassif_cross_validation_uci(embeddings_list,labels,nb_clusters)
            if "tfr" in args.task : 
                embeddings_list = embeddings_list[0]
                evaluate_twitter_friends_recommandation(embeddings_list,run_folder_path)
            if "mnist" in args.task:
                data = process_data.load_dataset(args.task)
                labels = np.array(data['labels'])
                nb_clusters = data['nb_clusters']
                evalute_classif_cross_validation_mnist2views(embeddings_list,labels, nb_clusters,run_folder_path)
        else:
            print('Unknown date %s'%args.date)
            parser.print_help()
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
