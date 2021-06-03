# MVGCCA

## Requirements

- python 3.6
- numpy
- tqdm
- tensorflow 2.3
- sklearn
- scipy
- matplotlib

## Datasets

All the datasets are decribed in details on article (~ will come)

- Uci Handwritten digits multiviews datasets, available here : https://archive.ics.uci.edu/ml/datasets/Multiple+Features           

  The file uci7.mat and uci10.mat used in the code is provided on this repository in folder "datasets".
- Twitter Friend recommendation, available here : http://www.cs.jhu.edu/~mdredze/datasets/multiview_embeddings/

  Because of the size of the file, it is not provided on this repository. Hence, you have to download user_6views_tfidf_pcaEmbeddings_userTweets+networks.tsv.gz (1.4 GB) file. Extract it. Then concatenate file uidPerFriend_test_all.txt, uidPerFriend_dev_all.txt to uidPerFriend_fuse_all.txt. And concatenate uidPerHashtag_test_all.txt, uidPerHashtag_dev_all.txt to uidPerHashtag_fuse_all.txt.

uidPerFriend_fuse_all.txt and uidPerHashtag_fuse_all.txt must be in path datasets/twitter/friend_and_hashtag_prediction_userids/.
- Mnist2views (~ will come)

## Tasks

All the tasks are decribed in details on article (~ will come)

Task available : ["uci7", "uci10", "uci7robustinf", "uci10robustinf", "uci7robustclassif", "uci10robustclassif", "uci10robustclassifv2",'uci7robustclassifv2',"mnist2views","tfr"]

- uci7/uci10 : Evaluate clustering and classification on uci7/uci10 latent space.
- uci7robustinf/uci10robustinf : We split the dataset in train and test set. We eventually remove some views is test set (same views removed for all instances). In the train set, we add the label of each instances as the 7th views. We train the model on this train set. Then we infer the labels (the 7th views) of the test sets.
- uci7robustclassif/uci10robustclassif : We split the dataset in train and test set. We train the model on this train set. We then train a classifier on train latent space and evaluate it on test latent space.
- uci10robustclassifv2/uci7robustclassifv2 : Same experiment as previous, but this time removed views in test set are not necessary the same for differents instances.
- mnist2views : Evaluate classification on mnist2views latent space.
- tfr : Evaluate twitter recommendation task.

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
    '--views_dropout_max', type = int, default = 5, help='Integer. Views dropout is disable if null. Otherwise, for each epoch the numbers of views 
     ignored during encoding phase is sample between 1 and MIN(views_dropout_max,nb_of_views - 1)')

    '--latent_dim', type = int, default = 3, help='Dimension of latent space.'
    '--write_loss', type = str2bool, default = False, help='True or False. Decide whether or not to write loss training of model.'
    '--write_latent_space', type = str2bool, default = False, help='True or False. Decide whether or not to write model weights.'
    '--write_latent_space_interval', type = int, default = 100, help='If --write_latent_space True : epochs interval between two saves of latent space.
     The last epoch is always saved.'
    '--grid_search', type = str2bool, default = False, help='True or False. Decide whether or not to process a grid search.'
    '--num_of_run', type = int, default = 1, help='Number of times the algorithm is runned.'
    '--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation works only for uci 
    dataset related task and depends on the task selected). If option --num_of_run > 1, average evaluation of these run is returned for task =[uci7,uci10] ;
    while each run evaluation is returned for task=[uci7robustinf, uci10robustinf, uci7robustclassif, uci10robustclassif,  
    uci10robustclassifv2,uci7robustclassifv2].'
    
### Example

#### Input

`python3 run_mvgcca.py --task uci7 --device -1 --use_graph_decoder true --decoder_scalar_std true --encoder_nn_type krylov-4 --encoder_use_common_hidden_layer true --num_of_layer 4 --hidden_layer 1024--learning_rate 1e-4 --decay_learning_rate true --num_of_epochs 600 --batch_size 512 --dropout 0.5 --latent_dim 3 --write_loss true --write_latent_space true --num_of_run 2 --evaluation true`

This command will train the models two times and evalute the quality of latent space on uci7 task. it will create a results folder where loss value during training, latent space and clustering accuracy for all run will be saved The average results of these runs will be print at the end of computation and a 2 dimensional tsne-projection of latent space of the last run will be display.

  
#### Output

```
Parameters: {'batch_size': 512, 'decay_learning_rate': True, 'decoder_scalar_std': True, 'encoder_nn_type': 'krylov-4', 'encoder_use_common_hidden_layer': True, 'evaluation': True, 'gamma': 1, 'hidden_layer': 1024, 'keep_prob': 0.5, 'latent_dim': 3, 'learning_rate': 0.0001, 'num_of_epochs': 600, 'num_of_layer': 4, 'task': 'uci7', 'write_latent_space': True, 'write_loss': True, 'write_weights': False, 'parameters_main_path': 'results/uci7/November_25_2020_10h24m54s/parameters0', 'weights_path': '', 'write_loss_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/logs/', 'latent_space_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/embeddings/', 'evaluation_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/evaluation/'}
Average results
Kmeans Clustering Score on UCI7: 0.8362873574164587
Spectral Clustering Score on UCI7: 0.8488895835087565
Svm-rbf Accuracy on UCI7: 0.9707142857142858
```

### Reproduce article (to come) experiments
 
