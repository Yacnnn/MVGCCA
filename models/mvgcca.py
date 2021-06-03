import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from layers.mlp import Mlp
from layers.cdgn_mlp import Cdgn_Mlp
from layers.cdgn_krylov import Cdgn_Krylov


class Mvgcca(tf.keras.Model):
    def __init__(self, 
                 latent_dim = 3,
                 gamma = 1,
                 num_of_z_sampled = 1, 
                 num_of_layer = 3,
                 hidden_layer = 1024,
                 keep_prob = 0.5, 
                 encoder_nn_type = "krylov-4",
                 encoder_use_common_hidden_layer = True,
                 decoder_scalar_std = True,
                 use_gcn = False,
                 l2_reg = 0,
                 views_dropout_max = 0
                ):
        super(Mvgcca, self).__init__() 
        #Parameters
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.num_of_z_sampled = num_of_z_sampled
        self.num_of_layer = num_of_layer
        self.hidden_layer = hidden_layer
        self.keep_prob = keep_prob
        self.encoder_nn_type = encoder_nn_type
        self.encoder_use_common_hidden_layer = encoder_use_common_hidden_layer
        self.decoder_scalar_std = decoder_scalar_std
        self.use_gcn = use_gcn
        self.l2_reg = l2_reg,
        self.views_dropout_max = views_dropout_max
        self.first_run = True
        
    def build(self,shape):
        self.num_of_views = len(shape)
        self.views_features_dimension = [s[1] for s in shape]
        self.coeffnormalisation = [ self.views_features_dimension[s] * tf.math.log( 2 * np.pi ) for s in range(self.num_of_views)]
        # Decoders mean: conditional gaussian distribution with MLP
        self.cdgn_decoders_mean = [ Mlp(output_dim = self.views_features_dimension[s],
                                   num_of_layer = self.num_of_layer,
                                   hidden_layer_size = self.hidden_layer,
                                   keep_prob = self.keep_prob,
                                   l2_reg = self.l2_reg)
                              for s in range(self.num_of_views) ] 
        # Decoders std: weights matrix
        if self.decoder_scalar_std:
            self.rPhi = [ self.add_weight( shape = ( 1, 1 ), initializer = tf.initializers.Ones() ) for s in range(self.num_of_views) ]
        else:
            self.rPhi = [ self.add_weight( shape = ( self.views_features_dimension[s], self.views_features_dimension[s] ), initializer = tf.initializers.Identity() ) for s in range(self.num_of_views) ]
        # Encoders
        if "krylov" in self.encoder_nn_type :
            #Conditional gaussian distribution with Krylov
            self.cdgn_encoders = [ Cdgn_Krylov(output_dim = self.latent_dim,
                                       num_of_layer = self.num_of_layer,
                                       hidden_layer_size = self.hidden_layer,
                                       krylov_deep = int(self.encoder_nn_type.split('-')[-1]),
                                       keep_prob = self.keep_prob,
                                       use_common_hidden_layer = self.encoder_use_common_hidden_layer,
                                       l2_reg = self.l2_reg )
                               for s in range(self.num_of_views)]
        else : 
            #Conditional gaussian distribution with MLP
            self.cdgn_encoders = [  Cdgn_Mlp(
                    hidden_layer_size = self.hidden_layer,
                    num_of_layer = self.num_of_layer,
                    output_dim = self.latent_dim,
                    l2_reg = self.l2_reg,
                    keep_prob = self.keep_prob,
                    use_common_hidden_layer = self.encoder_use_common_hidden_layer
                    )  for s in range(self.num_of_views)]

    def call(self,views,indice,w):
        views_id = self.views_id_to_keep()
        mean, var, logvar = self.average_encoders(views, w, views_id = views_id)
        #Kullback-Leibler 
        kl_qp =  self.latent_dim + tf.reduce_sum( logvar, axis = 1 )  -  tf.reduce_sum( var, axis = 1 ) - tf.reduce_sum( mean ** 2, axis = 1 )
        kl_qp = - 0.5 * kl_qp
        #Cross Entropy Data + Graph
        ce_qp_sum = 0
        for i in range(self.num_of_z_sampled):
            z_sample = mean + tf.multiply( tf.math.sqrt(var), tf.random.normal(mean.shape))
            views_sample = [ ( self.cdgn_decoders_mean[s](z_sample)) for s in range(self.num_of_views) ]    
            if self.decoder_scalar_std :
                n_dist2 = [ tfp.distributions.MultivariateNormalTriL(loc=views[s], scale_tril=(tf.nn.relu(self.rPhi[s])+1e-6)*tf.eye(self.views_features_dimension[s])[np.newaxis,:]) for s in range(self.num_of_views) ]
            else:
                n_dist2 = [ tfp.distributions.MultivariateNormalTriL(loc=views[s], scale_tril= (self.rPhi[s][np.newaxis,:])  + 1e-6*tf.eye(self.rPhi[s].shape[0])   ) for s in range(self.num_of_views) ]
            ce_qp = 0
            for s in range(self.num_of_views) : 
                ce_qp += tf.reduce_mean(n_dist2[s].log_prob(views_sample[s]))
            ce_qp_sum += ce_qp
            if self.gamma > 0 :
                logits_w =  tf.clip_by_value( tf.nn.sigmoid( ( z_sample @ tf.transpose(z_sample) ) ), 1e-6, 1-1e-6, name="clip")
                mask = ( tf.ones(w.shape) + tf.eye(w.shape[0]) )/2.0
                logpA = tf.reduce_mean(tf.reduce_sum(   mask*( w*tf.math.log(logits_w) + (1-w)*tf.math.log(1- logits_w) )   ,axis=1)) 
                ce_qp_sum += self.gamma*logpA       
        ce_qp = ce_qp_sum / self.num_of_z_sampled
        return tf.reduce_mean( kl_qp  , axis = 0 ), - ce_qp
    
    # Encodes views from a limited numbers of views (the same for all instances) or from all views
    def average_encoders(self, views, w, views_id = []):
        if views_id == [] :
            views_id = np.arange(self.num_of_views)
        if "krylov" in self.encoder_nn_type:
            packs = [self.cdgn_encoders[s]([ views[s:s+1] , [w] ]) for s in views_id]
        else:
            packs = [self.cdgns[s](views[s:s+1]) for s in views_id] 
        means = [pack[0] for pack in packs]
        logvars = [pack[1] for pack in packs]
        vars_ = [tf.math.exp(logvar) for logvar in logvars]
        inv_vars = [1/var for var in vars_]
        meanXinv_var  =  [ mean * inv_var for mean, inv_var in zip(means,inv_vars)]
        mean = 0
        var = 0
        for k in range(len(views_id)):
            mean += meanXinv_var[k] 
            var += inv_vars[k]
        mean = mean / var
        var = 1/ var
        logvar = tf.math.log(var)
        return mean, var, logvar

    # Select views to keep when views dropout is activate
    def views_id_to_keep(self):
        if self.first_run == True:
            self.first_run = False
            return list(np.arange(self.num_of_views))
        if self.views_dropout_max > 0:
            views_id = np.arange(self.num_of_views)
            np.random.shuffle(views_id)
            return views_id[:np.random.randint(max(self.num_of_views - self.views_dropout_max,1),self.num_of_views+1 )]
            # return views_id[:np.random.randint(1,self.num_of_views+1 )]
        return list(np.arange(self.num_of_views))

    # Reconstruct views from a subset of views (the same for all instances)                  
    def get_reconstruct_views_from_someviews(self, views, w, views_id = []):
        mean, var, logvar = self.average_encoders(views, w, views_id = views_id)
        z_sample = mean + tf.multiply( tf.math.sqrt(var), tf.random.normal(mean.shape))
        views_sample = [ ( self.cdgn_decoders_mean[s](z_sample)).numpy() for s in range(self.num_of_views) ]    
        return views_sample 

    # Get all latent space [Z,Z1,...,ZM] from all views
    def get_latents_space(self,views, w):
        all_latent_space = [self.average_encoders(views, w, views_id = [id])[0].numpy() for id in range(self.num_of_views)]
        all_latent_space.insert(0, self.average_encoders(views, w)[0].numpy())
        return all_latent_space

    # Get common latent space Z from a subset of views (the same for all instances)     
    def get_common_latent_space_from_someviews(self,views,w,views_id):
        return self.average_encoders(views, w, views_id )[0].numpy()

    # Encode views from a limited numbers of views (can be not the same for all instance) or from all views
    def average_encoders2(self, views, wtab, views_id_tab ):
        views_id0 = np.arange(self.num_of_views)
        if "krylov" in self.encoder_nn_type:
            packs = [self.cdgn_encoders[s]([ views[s:s+1] , wtab[s:s+1] ]) for s in views_id0]
        else:
            packs = [self.cdgns[s](views[s:s+1]) for s in views_id0] 
        means = [pack[0] for pack in packs]
        logvars = [pack[1] for pack in packs]
        vars_ = [tf.math.exp(logvar) for logvar in logvars]
        inv_vars = [1/var for var in vars_]
        existing_views = np.transpose([ np.array([int(id_ in tab) for tab in views_id_tab]) for id_ in range(len(views)) ])
        meanXinv_var  =  [ mean * inv_var for mean, inv_var in zip(means,inv_vars)]
        mean = 0
        var = 0
        for k in range(self.num_of_views):
            mean += existing_views[:,k:k+1]*meanXinv_var[k] 
            var += existing_views[:,k:k+1]*inv_vars[k]
        mean = mean / var
        var = 1/ var
        logvar = tf.math.log(var)
        return mean, var, logvar

    # Get all latent space Z from a subset of views (not the same for all instance) or from all views  
    def get_common_latent_space_from_someviews2(self,views,wtab,views_id):
        return self.average_encoders2(views, wtab, views_id )[0].numpy()
