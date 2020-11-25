# MVGCCA

## Requirements

- python 3.6
- numpy
- tqdm
- tensorflow 2.3
- sklearn
- scipy
- matplotlib

## Tasks

Task available : ['uci7', 'uci10']

Others tasks will be soonly available.

## Usage 

### Command

`python3 run_mvgcca.py --task uci7 --device -1 --use_graph_decoder true --decoder_scalar_std true --encoder_nn_type krylov-4 --encoder_use_common_hidden_layer true --num_of_layer 4 --hidden_layer 1024--learning_rate 1e-4 --decay_learning_rate true --num_of_epochs 600 --batch_size 512 --dropout 0.5 --latent_dim 3 --write_loss true --write_latent_space true --num_of_run 2 --evaluation true`

This command will train the models two times and evalute the quality of latent space on uci7 task. it will create a results folder where loss value during training, latent space and clustering accuracy for all run will be saved The average results of these runs will be print at the end of computation and a 2 dimensional tsne-projection of latent space of the last run will be display.

    '--task', default='uci7', help='Task to execute. Only ['uci7', 'uci10'] are currently available.' 
    '--device', default='', help='Index of the target GPU. Specify '-1' to disable gpu support.'
    '--use_graph_decoder', type = str2bool, default= True, help='True or False. Decide whether or not to use graph reconstruction term in loss.'
    '--decoder_scalar_std', type = str2bool, default = True, help='True or False. Decide whether or not to use scalar matrix as covariance matrix for gaussian decoder.'
    '--encoder_nn_type', default='krylov-4', help='Encoders neural networks. Only Krylov-(Deep) is available. Example :krylov-4. '
    '--encoder_use_common_hidden_layer', type = str2bool, default = True, help='True or False. Whether or not to use different hidden layers to calculate the mean and variance of encoders'
    '--num_of_layer', type = int, default= 4, help='Number of layer for encoders AND decoders.'
    '--hidden_layer', type = int, default= 1024, help='Size of hidden layer for encoders AND decoders.'
    '--learning_rate', type = float, default = 1e-4, help = 'Learning rate.' 
    '--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate.'
    '--num_of_epochs', type = int, default = 1, help='Number of epochs.'
    '--batch_size', type = int, default = 512, help='Batch size.'
    '--dropout', type = float, default = 0.3, help='Dropout rate applied to every hidden layers.'
    '--latent_dim', type = int, default = 20, help='Dimension of latent space.'
    '--write_loss', type = str2bool, default = False, help='True or False. Decide whether or not to write loss training of model.'
    '--write_latent_space', type = str2bool, default = False, help='True or False. Decide whether or not to write model weights.'
    '--grid_search', type = str2bool, default = False, help='True or False. Decide whether or not to process a grid search.'
    '--num_of_run', type = int, default = 1, help='Number of times the algorithm is runned.'
    '--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation function depend on the task selected). If option --num_of_run > 1 average evaluation of all run is returned. Results of evaluation will be written in results folder.'    
### Output

```
Parameters: {'batch_size': 512, 'decay_learning_rate': True, 'decoder_scalar_std': True, 'encoder_nn_type': 'krylov-4', 'encoder_use_common_hidden_layer': True, 'evaluation': True, 'gamma': 1, 'hidden_layer': 1024, 'keep_prob': 0.5, 'latent_dim': 3, 'learning_rate': 0.0001, 'num_of_epochs': 600, 'num_of_layer': 4, 'task': 'uci7', 'write_latent_space': True, 'write_loss': True, 'write_weights': False, 'parameters_main_path': 'results/uci7/November_25_2020_10h24m54s/parameters0', 'weights_path': '', 'write_loss_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/logs/', 'latent_space_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/embeddings/', 'evaluation_path': 'results/uci7/November_25_2020_10h24m54s/parameters0/run1/evaluation/'}
Average results
Kmeans Clustering Score on UCI7: 0.8362873574164587
Spectral Clustering Score on UCI7: 0.8488895835087565
Svm-rbf Accuracy on UCI7: 0.9707142857142858
```
 
