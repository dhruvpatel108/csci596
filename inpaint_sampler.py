# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:31:10 2019

@author: pogo
"""
import os
import utils
import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import tensorflow_probability as tfp

tfd= tfp.distributions

tf.reset_default_graph()
N = 64000
burn = int(0.5*N)
z_dim = 100
batch_size = 1 # should always be one
noise_lvl = 0.8
dim_like = 28*28*1
seed_no = 1008
np.random.seed(seed_no)

parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int, default=1)
parser.add_argument('--noise_var', type=float, default=noise_lvl)
parser.add_argument('--start_row', type=int)
parser.add_argument('--end_row', type=int)
parser.add_argument('--start_col', type=int)
parser.add_argument('--end_col', type=int)
PARAMS = parser.parse_args()
print('------------- digit = {}     ---------------'.format(PARAMS.digit))
print('------------- noise_var = {} ---------------'.format(PARAMS.noise_var))
print('------------- start_row = {} ---------------'.format(PARAMS.start_row))
print('------------- end_row = {} ---------------'.format(PARAMS.end_row))
print('------------- start_col = {} ---------------'.format(PARAMS.start_col))
print('------------- end_col = {} ---------------'.format(PARAMS.end_col))
noise_var = PARAMS.noise_var
"""
noisy_mat3d = np.expand_dims(noisy_data, axis=2)
noisy_mat4d = np.tile(noisy_mat3d, (batch_size,1,1,1)).astype(np.float32)
"""

save_dir = './inpaint_exps/digit{}_var{}_N{}_strow{}_endrow{}_stcol{}_endcol{}'.format(PARAMS.digit, noise_var, N, 
                                                                                PARAMS.start_row, PARAMS.end_row, PARAMS.start_col, PARAMS.end_col)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
utils.mkdir(save_dir+'/')

''' data '''
nice_digit_idx = [10,2,1,32,4,15,11,0,128,12]
test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[nice_digit_idx[PARAMS.digit]], [28,28,1])
noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1)
noisy_mat3d = test_sample + np.reshape(noise_mat3d, [28,28,1])
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)
np.save(save_dir+'/noisy_mat4d.npy', noisy_mat4d)
#noisy_mat4d = np.load('./mcmc/likelihood_compare/digit1_var0.8_N64000.npy')
#noisy_mat4d = np.load('./mcmc/digit7_var0.8_N64000/noisy_mat4d.npy')

mask = np.ones((batch_size, 28,28,1))
mask[:, PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 0.
dim_inpaint = int(np.sum(mask))
np.save(save_dir+'/mask.npy', mask)

with tf.Graph().as_default() as g:
    def joint_log_prob(z):              
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = gen_out - tf.constant(noisy_mat4d)
        visible_img = tf.boolean_mask(diff_img, mask)
        
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), scale_diag=np.ones(z_dim, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_inpaint, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_inpaint, dtype=np.float32))
        
        return (prior.log_prob(z) + like.log_prob(visible_img))
                                          

    def unnormalized_posterior(z):
        return joint_log_prob(z) 
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                    tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_posterior, step_size=np.float32(1.), num_leapfrog_steps=3),
                   num_adaptation_steps=int(0.8*burn))
    

    #initial_state = tf.constant(np.zeros((batch_size, z_dim)).astype(np.float32))
    initial_state = tf.constant(np.random.normal(size=[batch_size, z_dim]).astype(np.float32))
    samples, [st_size, log_accept_ratio] = tfp.mcmc.sample_chain(
      num_results=N,
      num_burnin_steps=burn,
      current_state=initial_state,
      kernel=adaptive_hmc,
      trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                             pkr.inner_results.log_accept_ratio])
    p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))

     
    zz = tf.placeholder(tf.float32, shape=[N-burn, z_dim])    
    gen_out1 = gen(zz, reuse=tf.AUTO_REUSE, training=False)
    
    model_path = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'    
    variables_to_restore = slim.get_variables_to_restore()   
    #variables_to_restore = [v for v in variables if v.name.split(':')[0]!='dummy'] 
    saver = tf.train.Saver(variables_to_restore)
    

with tf.Session(graph=g) as sess:
    
    saver.restore(sess, model_path)
 
    
    
    samples_ = sess.run(samples)
    np.save(save_dir+'/samples.npy', samples_)
    #np.save(save_dir+'/noisy_mat4d.npy', noisy_mat4d)
    #print('acceptance ratio = {}'.format(sess.run(p_accept)))
