# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:40:46 2019

@author: pogo
"""
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import matplotlib.pyplot as plt



seed_no = 1008
N = 64000  #np.size(mcmc_samps, 0)
burn = int(0.5*N)
n_eff = N-burn
batch_size = 640
z_dim = 100
n_iter = int(n_eff/batch_size)
dim_like = 28*28*1
np.random.seed(seed_no)

parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int)
parser.add_argument('--noise_var', type=float)
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

sample_dir = './inpaint_exps/digit{}_var{}_N{}_strow{}_endrow{}_stcol{}_endcol{}'.format(PARAMS.digit, noise_var, N, 
                                                                                PARAMS.start_row, PARAMS.end_row, PARAMS.start_col, PARAMS.end_col)
mcmc_samps = np.load(sample_dir + '/samples.npy')
eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

''' data '''
nice_digit_idx = [10,2,1,32,4,15,11,0,128,12]
test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[nice_digit_idx[PARAMS.digit]], [28,28,1])
x_true = test_sample[:,:,0]
noisy_mat4d = np.load(sample_dir+'/noisy_mat4d.npy')

mask = np.ones((batch_size, 28,28,1))
mask[:, PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 0.
dim_inpaint = int(np.sum(mask))


plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.hist(eff_samps[:, ii], 50, density=True);
    plt.xlabel(r'z_{} '.format(ii))
plt.tight_layout()
plt.savefig('./{}/hist_eff_samples'.format(sample_dir))
plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.plot(mcmc_samps[:, 0, ii])
    plt.ylabel(r'z_{}'.format(ii))
plt.tight_layout()
plt.savefig('./{}/eff_samples'.format(sample_dir))
#plt.show()


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(noisy_mat4d)
    visible_img = tf.boolean_mask(diff_img, mask)


    model_path = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'    
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)

with tf.Session(graph=g) as sess:
    saver.restore(sess, model_path)
    
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((28,28,1))
    x2_mean = np.zeros((28,28,1))    
    for ii in range(n_iter):
        g_z, diff = sess.run([gen_out, visible_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):
            loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff)**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2

    
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter    
    var = x2_mean - (x_mean)**2
    map_ind = np.argmin(loss)
    g_z_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})
 
    rec_error = np.linalg.norm(g_z_map[0,:,:,0]-x_true)/dim_like
   
    mask_ = np.zeros((28,28,1))
    mask_[PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 1.
    mx = np.ma.masked_array(noisy_mat4d[0,:,:,:],mask=mask_) 
 
    fig, axs = plt.subplots(3,2, figsize=(20,20))
    im1 = axs[0][0].imshow(x_true)    
    fig.colorbar(im1, ax=axs[0][0])
    axs[0][0].set_title('x')
    im2 = axs[0][1].imshow(mx[:,:,0])    
    fig.colorbar(im2, ax=axs[0][1])
    axs[0][1].set_title('y')
    
    im3 = axs[1][0].imshow(g_z_map[0,:,:,0])    
    fig.colorbar(im3, ax=axs[1][0])    
    axs[1][0].set_title('g(z_map)')
    im4 = axs[1][1].imshow(g_z_map[0,:,:,0]-x_true)    
    fig.colorbar(im4, ax=axs[1][1])
    axs[1][1].set_title('g(z_map) - x | Reconstruction error(per pixel)={}'.format(rec_error))    
    
    im5 = axs[2][0].imshow(x_mean[:,:,0])    
    fig.colorbar(im5, ax=axs[2][0])
    axs[2][0].set_title('x_mean')
    im6 = axs[2][1].imshow(var[:,:,0])    
    fig.colorbar(im6, ax=axs[2][1])
    axs[2][1].set_title('avg. var = {}'.format(np.mean(var)))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(sample_dir+'/stats')
    
    np.save(sample_dir+'/x_true.npy', x_true)
    np.save(sample_dir+'/x_var.npy', var)
    np.save(sample_dir+'/x_mean.npy', x_mean)
    np.save(sample_dir+'/x_map.npy', g_z_map)
    np.save(sample_dir+'/mask_.npy', mask_)
    #plt.show()
    
