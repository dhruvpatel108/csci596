# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:16:05 2019

@author: pogo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import tensorflow as tf
import numpy as np
from models_64x64 import generator as gen
import utils
from tensorflow.python.tools import inspect_checkpoint as ckpt

graph_name = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'
#ckpt.print_tensors_in_checkpoint_file(graph_name, tensor_name='', all_tensors=False)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    
    
    saver = tf.train.import_meta_graph(graph_name+'.meta')
    saver.restore(sess,'./'+graph_name)
    print("Model Restored!")
    # Sample single z
    graph = tf.get_default_graph()
    print('********************************************')
    z_ = graph.get_tensor_by_name('Placeholder_1:0') 
    fake = graph.get_tensor_by_name('generator/Tanh:0') 
    
"""
tf.reset_default_graph()
N = 6400
batch_size = 64
z_dim = 100
noise_lvl = 0.02
like_stddev = noise_lvl
dim_prior = z_dim
prior_stddev = 1.0


n_iter = int(N/batch_size)
numerator = np.zeros((batch_size, n_iter))
log_like = np.zeros((batch_size, n_iter))
log_prior = np.zeros((batch_size, n_iter))
loss_value = np.zeros((batch_size, n_iter))
sample_loss =np.zeros((batch_size, n_iter))

#x_rec = np.zeros((batch_size, 64, 64, 3, n_iter))
dim_like = 64*64*3  #h*w*ch
z_store = np.zeros((batch_size, z_dim, n_iter))



def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/test/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)

noise_mat3d = noise_lvl*np.random.randn(64,64,3)
noisy_mat3d = data_pool.batch()[0,:,:,:] + noise_mat3d
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)

noisy_mat4d[noisy_mat4d<-1]=-1
noisy_mat4d[noisy_mat4d>1]=1


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])    
    #z__ = tf.Variable(name='z__', initial_value=np.random.normal(size=[batch_size, z_dim]).astype(np.float32), trainable=False)
    #dummy = tf.Variable(name='dummy', trainable=False, initial_value=z1)   
    #assign_op1 = dummy.assign(z1)
    
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    #diff_img = gen_out - tf.constant(noisy_mat4d)
    
    #init_op = tf.initializers.global_variables()
    saver = tf.train.Saver()
    
z_sample = np.random.normal(size=[batch_size, z_dim]).astype(np.float32)
with tf.Session(graph=g) as sess:
    #sess.run(init_op, feed_dict={z__:z_sample})
    #_ = sess.run(assign_op1, feed_dict={z1:z_sample})    
    saver.restore(sess, './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt')
    print('Model restored!')
    
    gen_out_ = sess.run(gen_out, feed_dict={z1:z_sample})
    
    save_dir = './sample_test/mnist_wgan_gp{}'.format(N)
    utils.mkdir(save_dir + '/')
    #utils.imwrite(utils.immerge(noisy_mat4d, 10, 10), '{}/214_noisy.jpg'.format(save_dir))# % (save_dir, epoch, it_epoch, batch_epoch))
    utils.imwrite(utils.immerge(gen_out_, 8, 8), '{}/2215.jpg'.format(save_dir))
    
    #_ = sess.run(assign_op, feed_dict={z1:z_sample})
    #diff_img_ = sess.run(diff_img, feed_dict={z1:z_sample})
    
    for n in range(n_iter):
        z_sample = np.random.normal(size=[batch_size, z_dim])
        z_store[:,:,n] = z_sample
        print('iterations={}'.format(n))
        for k in range(batch_size):
            sample_norm2 = np.linalg.norm(diff_img_[k,:,:,:])**2
            numerator[k,n] = np.exp(-(sample_norm2/(2*like_stddev*like_stddev)))
            loss_value[k,n] = ((0.5*sample_norm2)/(like_stddev**2)) + (0.5*np.linalg.norm(z_sample[k,:])**2)   

    print(np.max(numerator))
    print(np.mean(numerator))
    norm_prob = numerator/np.sum(numerator)
    k_i, n_i = np.unravel_index(loss_value.argmin(), loss_value.shape)
    x_map4d = sess.run(gen_out, feed_dict={z__:z_store[:,:,n_i]})
    x_map = x_map4d[k_i,:,:,:]   
    print('pre-shape of x_map = {}'.format(np.shape(x_map))) 
    x_map = np.tile(x_map, (batch_size, 1, 1, 1)).astype(np.float32) 
    print('post-shape of x_map = {}'.format(np.shape(x_map)))
    """
