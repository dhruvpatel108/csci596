from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import gzip
import struct
import subprocess
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def mnist_load(data_dir, dataset='train'):
    """
    modified from https://gist.github.com/akesling/5358964

    return:
    1. [-1.0, 1.0] float64 images of shape (N * H * W)
    2. int labels of shape (N,)
    3. # of datas
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    y_train = mnist.train.labels    
    X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    y_test = mnist.test.labels
    
    if dataset is 'train':
        assert np.shape(X_train)==(55000, 784), 'Shape of X_train is not consistent'
        img = np.reshape(X_train, [55000, 28, 28])
        lbls = y_train        
    elif dataset is 'test':
        assert np.shape(X_test)==(10000, 784), 'Shape of X_test is not consistent'
        img = np.reshape(X_test, [10000, 28, 28])
        lbls = y_test
    else:
        raise ValueError("dataset must be 'test' or 'train'")
    img = (img * 2 - 1).astype(np.float64)
    
    return img, lbls, len(lbls)

"""
def unzip_gz(file_name):
    unzip_name = file_name.replace('.gz', '')
    gz_file = gzip.GzipFile(file_name)
    print(gz_file)
    open(unzip_name, 'w+').write(gz_file.read())
    gz_file.close()


def mnist_load(data_dir, dataset='train'):
    
    #modified from https://gist.github.com/akesling/5358964

    #return:
    #1. [-1.0, 1.0] float64 images of shape (N * H * W)
    #2. int labels of shape (N,)
    #3. # of datas
    

    if dataset is 'train':
        fname_img = os.path.join(data_dir, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    elif dataset is 'test':
        fname_img = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    if not os.path.exists(fname_img):
        unzip_gz(fname_img + '.gz')
    if not os.path.exists(fname_lbl):
        unzip_gz(fname_lbl + '.gz')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, n = struct.unpack('>II', flbl.read(8))
        lbls = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack('>IIII', fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), rows, cols) / 127.5 - 1

    return imgs, lbls, len(lbls)


def mnist_download(download_dir):
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = url_base + file_name
        save_path = os.path.join(download_dir, file_name)
        cmd = ['curl', url, '-o', save_path]
        print('Downloading ', file_name)
        if not os.path.exists(save_path):
            subprocess.call(cmd)
        else:
            print('%s exists, skip!' % file_name)
"""