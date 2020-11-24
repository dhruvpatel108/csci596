# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:32:42 2019

@author: pogo
"""

# Jay Swaminarayan
import os
import glob
import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as ckpt
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

graph_name = './checkpoints/celeba_wgan_gp/Epoch_(49)_(32of632).ckpt'
#ckpt.print_tensors_in_checkpoint_file(graph_name, tensor_name='', all_tensors=False)
N = 640000
z_dim = 100
batch_size = 64
noise_lvl = 0.02
noise_mat3d = noise_lvl*np.random.randn(64,64,3)

def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/test/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)


noisy_mat3d = data_pool.batch()[0,:,:,:] + noise_mat3d
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1))
"""
plt.figure()
plt.imshow(data_pool.batch()[0,:,:,:], interpolation='bilinear')
plt.colorbar()

plt.figure()
plt.imshow(noisy_mat4d[0,:,:,:], interpolation='bilinear')
plt.colorbar()
"""

like_stddev = noise_lvl
prior_stddev = 1.0
dim_prior = z_dim            

n_iter = int(N/batch_size)
numerator = np.zeros((batch_size, n_iter))
log_like = np.zeros((batch_size, n_iter))
log_prior = np.zeros((batch_size, n_iter))
loss_value = np.zeros((batch_size, n_iter))

sample_loss =np.zeros((batch_size, n_iter))
#x_rec = np.zeros((batch_size, 64, 64, 3, n_iter))
dim_like = 64*64*3  #h*w*ch
z_store = np.zeros((batch_size, z_dim, n_iter))

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    
    """
    saver = tf.train.import_meta_graph(graph_name+'.meta')
    saver.restore(sess,'./'+graph_name)
    print("Model Restored!")
    
    # Sample single z
    graph = tf.get_default_graph()
    print('********************************************')
    z_ = graph.get_tensor_by_name('Placeholder_1:0') 
    fake = graph.get_tensor_by_name('generator/Tanh:0') 
        
    for n in range(n_iter):
        z_sample = np.random.normal(size=[batch_size, z_dim])
        z_store[:,:,n] = z_sample
        fake_sample = sess.run(fake, feed_dict={z_:z_sample})
        diff_mat4d = fake_sample - noisy_mat4d
        #x_rec[:,:,:,:,n] = fake_sample
        print('iteration={}'.format(n))
        for k in range(batch_size):
            sample_norm2 = np.linalg.norm(diff_mat4d[k,:,:,:])**2
            #print('sample_norm2 = {}'.format(sample_norm2))
            numerator[k, n] = np.exp(-((sample_norm2)/(2*like_stddev*like_stddev)))
            #print('numerator = {:3E}'.format(numerator[k,n]))            
            loss_value[k,n] = ((0.5*sample_norm2)/(like_stddev**2)) + (0.5*np.linalg.norm(z_sample[k,:])**2)
            #print('loss_value = {}'.format(loss_value[k,n]))
            #log_like[k,n] = -(sample_norm2/(2*like_stddev*like_stddev)) - ((dim_like/2)*np.log(2*np.pi)) - (dim_like*np.log(like_stddev))
            #log_prior[k,n] = -(np.linalg.norm(z_sample[k,:])**2/(2*prior_stddev*prior_stddev)) - ((dim_prior/2)*np.log(2*np.pi)) - (dim_prior*np.log(prior_stddev))    

    print('max numerator = {:4E}'.format(np.max(numerator)))
    print('min numerator = {:4E}'.format(np.min(numerator)))
    print('mean numerator = {:3E}'.format(np.mean(numerator)))
    norm_prob = numerator/np.sum(numerator)
    log_post = log_prior + log_like            
    

    k_i, n_i = np.unravel_index(loss_value.argmin(), loss_value.shape)
    x_map4d = sess.run(fake, feed_dict={z_:z_store[:,:,n_i]})
    x_map = x_map4d[k_i,:,:,:]
    #x_map = x_rec[k_i, :, :, :, n_i]
    fig = plt.figure()
    axs = fig.add_subplot(111)
    cb = plt.imshow(x_map, cmap='viridis', interpolation='bilinear')
    plt.colorbar(cb, ax=axs)
    plt.title('x_map_MC - LikeStddev={}, N={}'.format(like_stddev, N))
    save_dir='data/test/celeba_mc_map/front_face_m_glass/epoch50model/run1'
    plt.savefig('./{}/xmap{}'.format(save_dir, N))
    plt.show()  
       
            

           
    x_mean = np.zeros((64, 64, 3)) 
    x2_mean = np.zeros((64, 64, 3))
    x2_rec = x_rec**2
    
    prob_reshape = np.reshape(norm_prob, [batch_size*n_iter, 1])
    xt = np.transpose(x_rec, (1,2,3,0,4))
    xrt = np.reshape(xt, [64, 64, 3, batch_size*n_iter])
    xt2 = np.transpose(x2_rec, (1,2,3,0,4))
    xrt2 = np.reshape(xt2, [64, 64, 3, batch_size*n_iter])

    x_mean = np.squeeze(np.dot(xrt, prob_reshape))
    x2_mean = np.squeeze(np.dot(xrt2, prob_reshape))            
    var = x2_mean - (x_mean**2)
    
    descending = np.sort(prob_reshape, axis=0)[::-1]
    cummulative = np.cumsum(descending)
    effective_samps = int(sum((cummulative<0.99).astype(int)))    
    print('sum of normailized prob. = {} and shape of it is {}'.format(np.sum(norm_prob), np.shape(norm_prob)))
    print('no. of non-zero norm_probs = {}'.format(np.size(np.nonzero(norm_prob))))
    print('max. norm-prob = {}'.format(np.max(norm_prob)))
    print('no. of effective samples = {}'.format(effective_samps))
    print('Total loss = {}'.format(np.sum(sample_loss)))   

    fig = plt.figure()
    axs = fig.add_subplot(111)
    cb = plt.imshow(x_mean, cmap='viridis', interpolation='bilinear')
    plt.colorbar(cb, ax=axs)
    plt.title('x_mean_MC - LikeStddev={}, N={}'.format(like_stddev, N))
    #plt.savefig('./{}/xmean_N{}_effsamples{}'.format(save_dir, N, effective_samps))
    #plt.show()  
    
    fig = plt.figure()
    axs = fig.add_subplot(111)
    cb = plt.imshow(np.sqrt(var), cmap='viridis', interpolation='bilinear')
    plt.colorbar(cb, ax=axs)
    plt.title('var_MC - LikeStddev={}, N={}'.format(like_stddev, N))
    #plt.savefig('./{}/stddev_N{}_effsamples{}'.format(save_dir, N, effective_samps))
    #plt.show()       
    """
    
    