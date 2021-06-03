# MVGCCA

## Requirements

- python 3.6
- numpy
- tqdm
- tensorflow 2.3
- sklearn
- scipy
- matplotlib

## Multiviews Dataset with graph prior
### Example 
![multiview_dataset](https://user-images.githubusercontent.com/73817884/120652163-c9c5b480-c47f-11eb-8513-82213542c9e1.png)
### Multiviews Dataset with missing views
![multiview_dataset_corruptedv2](https://user-images.githubusercontent.com/73817884/120651931-9551f880-c47f-11eb-8f5d-f277b995d716.png)

## Datasets

All datasets are described in the article (~ will come)

- __UCI Handwritten digits__ multiviews dataset, available here : https://archive.ics.uci.edu/ml/datasets/Multiple+Features           

  The files uci7.mat and uci10.mat used in the code are provided in this repository in folder "datasets/uci/".
- __Twitter Friend Recommendation__, available here : http://www.cs.jhu.edu/~mdredze/datasets/multiview_embeddings/

  Because of the size of the file, it is not provided on this repository. Hence, you have to download user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv.gz (1.4 GB) file. Extract it. Then concatenate file uidPerFriend_test_all.txt, uidPerFriend_dev_all.txt to uidPerFriend_fuse_all.txt. And concatenate uidPerHashtag_test_all.txt, uidPerHashtag_dev_all.txt to uidPerHashtag_fuse_all.txt.
  
*uidPerFriend_fuse_all.txt and uidPerHashtag_fuse_all.txt must be in path "datasets/twitter/friend_and_hashtag_prediction_userids/".*
- __Mnist2views__ multiviews dataset based on Mnist, available here : https://drive.google.com/drive/folders/1WCfjQRQ79B68YvZG-wQNAIHj0Fxzqw0j?usp=sharing
The downloaded file "mnist2views_example.mat" must be in folder "datasets/mnist2views/".

NB : A short sample of this dataset is provided in "datasets/mnist2views/mnist2views_example.mat" to help you to build your own, if the upper link is death.

## Tasks

All tasks are described in article (~ link will come)

Task available : ["uci7", "uci10", "uci7robustinf", "uci10robustinf", "uci7robustclassif", "uci10robustclassif", "uci10robustclassifv2",'uci7robustclassifv2',"mnist2views","tfr"]

- __uci7/uci10__ : Evaluate clustering and classification on uci7/uci10 latent space.


- __uci7robustinf/uci10robustinf__ : We split the dataset in train and test set. We remove some views in test set (following scenario 1). In the train set, we add the label of each instances as the 7th views. We train the model on this train set. Then we infer the labels (the 7th views) of test set instances.
- __uci7robustclassif/uci10robustclassif__ : We split the dataset in train and test set. We remove some views in test set (following scenario 1). Then we train the model on this train set. Finally a classifier trained on train set latent space is evaluated on test set latent space.
- __uci10robustclassifv2/uci7robustclassifv2__ : Same experiment as previous, but this time we follow scenario 2 when we remove some test set views.
- __mnist2views__ : Evaluate classification on mnist2views latent space.
- __tfr__ : Evaluate twitter recommendation task.

## Usage 

### Command

    '--task', default='uci7', help='Task to execute. Only ["uci7", "uci10", "uci7robustinf", "uci10robustinf", "uci7robustclassif", "uci10robustclassif", "uci10robustclassifv2",'uci7robustclassifv2',"mnist2views","tfr"] are currently available.' 
    '--device', default='', help='Index of the target GPU. Specify '-1' to disable gpu support.'
    '--use_graph_decoder', type = str2bool, default= True, help='True or False. Decide whether or not to use graph reconstruction term in loss.'
    '--decoder_scalar_std', type = str2bool, default = True, help='True or False. Decide whether or not to use scalar matrix as covariance matrix for gaussian decoder.'
    '--encoder_nn_type', default='krylov-4', help='Encoders neural networks. Only Mlp and Krylov-(Deep) is available. Example :krylov-4. '
    '--encoder_use_common_hidden_layer', type = str2bool, default = True, help='True or False. Whether or not to use different hidden layers to calculate the mean and variance of encoders'
    '--num_of_layer', type = int, default= 4, help='Number of layer for encoders AND decoders.'
    '--hidden_layer', type = int, default= 1024, help='Size of hidden layer for encoders AND decoders.'
    '--learning_rate', type = float, default = 1e-4, help = 'Learning rate.' 
    '--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate.'
    '--num_of_epochs', type = int, default = 1, help='Number of epochs.'
    '--batch_size', type = int, default = 512, help='Batch size.'
    '--dropout', type = float, default = 0.3, help='Dropout rate applied to every hidden layers.'
    '--views_dropout_max', type = int, default = 0, help='Integer. Views dropout is disable if null. Otherwise, for each epoch the numbers of views 
     ignored during encoding phase is sample between 1 and MIN(views_dropout_max,nb_of_views - 1)')
    '--latent_dim', type = int, default = 3, help='Dimension of latent space.'
    '--write_loss', type = str2bool, default = False, help='True or False. Decide whether or not to write loss training of model.'
    '--write_latent_space', type = str2bool, default = True, help='True or False. Decide whether or not to write latent space.'
    '--write_latent_space_interval', type = int, default = 100, help='If --write_latent_space True : epochs interval between two saves of latent space.
     The last epoch is always saved.'
    '--grid_search', type = str2bool, default = False, help='True or False. Decide whether or not to process a grid search.'
    '--num_of_run', type = int, default = 3, help='Number of times the algorithm is runned.'
    '--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation works only for uci 
    dataset related task and depends on the task selected). If option --num_of_run > 1, average evaluation of these run is returned for task =[uci7,uci10] ;
    while each run evaluation is returned for task=[uci7robustinf, uci10robustinf, uci7robustclassif, uci10robustclassif,  
    uci10robustclassifv2,uci7robustclassifv2].'
    
### Example

#### Input

`python3 run_mvgcca.py --task uci7 --device -1 --use_graph_decoder true --decoder_scalar_std true --encoder_nn_type krylov-4 --encoder_use_common_hidden_layer true --num_of_layer 4 --hidden_layer 1024--learning_rate 1e-4 --decay_learning_rate true --num_of_epochs 600 --batch_size 512 --dropout 0.5 --latent_dim 3 --write_loss true --write_latent_space true --num_of_run 2 --evaluation true`

This command will train the models two times and evalute the quality of latent space on uci7 task. It will create "results" folder where loss value during training, latent space and clustering accuracy for all run will be saved. The average results of these runs will be printed at the end of computation and a 2 dimensional tsne-projection of latent space of the last run will be displayed.

  
#### Output

```
Parameters: {'batch_size': 512, 'decay_learning_rate': True, 'decoder_scalar_std': True, 'encoder_nn_type': 'krylov-4', 'encoder_use_common_hidden_layer': True, 'evaluation': True, 'gamma': 1, 'hidden_layer': 1024, 'keep_prob': 0.5, 'latent_dim': 3, 'learning_rate': 0.0001, 'num_of_epochs': 600, 'num_of_layer': 4, 'task': 'uci7', 'write_latent_space': True, 'write_loss': True, 'write_weights': False, 'parameters_main_path': 'results/uci7/November_25_2020_10h24m54s/parameters0', 'weights_path': '', 'write_loss_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/logs/', 'latent_space_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/embeddings/', 'evaluation_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/evaluation/'}
Average results
Kmeans Clustering Score on UCI7: 0.8362873574164587
Spectral Clustering Score on UCI7: 0.8488895835087565
Svm-rbf Accuracy on UCI7: 0.9707142857142858
```

### Reproduce article (to come) experiments

#### UCI Clustering and Classification 

*Perform the grid search :* 
``` 
python3 run_mvgcca.py --task uci7 --grid_search True 
```
It will create a folder (for example) April_23_2021_01h05m53s in folder "results/uci7/" with all the information about the gridsearch. 

*Evaluate the grid search :*
``` 
python3 run_mvgcca_grid_search_evaluation.py --task uci7 '--date' April_23_2021_01h05m53s --write_latent_space_interval 100
```
This command will evaluate the gridsearch performed on uci7 at April_23_2021_01h05m53s and print it.
(SVM-rbf accuracy, Kmeans adjusted rand index, spectral clustering rand index)

#### UCI Robust Classification with Inference
``` 
python3 run_mvgcca.py --task uci7robustinf --views_dropout_max 0 (or 5)
``` 
It will print the acccuracy for different levels of views removed in test set.
We remove between 0 and 5 views.
#### UCI Robust Classification 
``` 
python3 run_mvgcca.py --task uci7robustclassif --views_dropout_max 0 (or 5)
```
It will print the acccuracy for different levels of views removed in test set.
We remove between 0 and 5 views.
#### UCI Robust Classificationv2
``` 
python3 run_mvgcca.py --task uci7robustclassifv2 --views_dropout_max 0 (or 3)
``` 
It will print the acccuracy for different levels of views removed in test set.
We remove [1, 5, 10, 15 , 25, 50, 75 ] percent of views.
#### Twitter Friends Recommendation

In previous experiments, the number of run corresponded to the number of times we trained the model for each parameter (in order to compute average performance for each parameter across the different run). For twitter friends recommendation, it is the same but each run corresponds to a different sampling of the (huge) inital datasets. We trained the method for one set of parameters for 100 runs (i.e 100 different sampling) and then computed the performance for each of these runs (every 100 epochs). 

*Perform runs :* 
``` 
python3 run_mvgcca.py --task tfr --latent_dim 5 --num_of_run 100 --write_latent_space_interval 100
```
When you launch this command, for each run $i$, if the file "datasets/tfr/twitter$i$.mat" exists we load it. Otherwise we sample 2506 users from database and create the associated weighted graph as specified in the paper. Then we save it in "datasets/tfr/twitter$i$.mat".
However, this command will also create a folder (for example) April_23_2021_01h05m53s in folder "results/tfr/" with all runs information.

*Evaluate runs :*
``` 
python3 run_mvgcca_grid_search_evaluation.py --task tfr '--date' April_23_2021_01h05m53s --write_latent_space_interval 100
```
This command will evaluate the different run on April_23_2021_01h05m53s. It will print the precision, recall and mrr metrics for each epochs saved and this for all run.

#### Mnist2views Classsification

A grid search was used to decide latent_dim = 60. (30 vs 60)

*Perform runs :* 

``` 
python3 run_mvgcca.py --task mnist2views --latent_dim 60  --num_of_epochs 200 --write_latent_space_interval 10

```
It will create a folder (for example) April_23_2021_01h05m53s in folder "results/mnist2views/" with all the information about the gridsearch. 

*Evaluate runs :*
``` 
python3 run_mvgcca_grid_search_evaluation.py --task uci7 '--date' April_23_2021_01h05m53s
```
This command will evaluate the gridsearch performed on mnist2views at April_23_2021_01h05m53s and print it.
(SVM-Rbf accuracy)

