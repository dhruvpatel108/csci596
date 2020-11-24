# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:22:09 2019

@author: pogo
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from keras.layers import Dense, Dropout, Activation, Flatten
#(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#train_images = train_images.reshape((60000, 28, 28, 1))
#test_images = test_images.reshape((10000, 28, 28, 1))

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
        img = np.reshape(X_train, [55000, 28, 28,1])
        lbls = y_train        
    elif dataset is 'test':
        assert np.shape(X_test)==(10000, 784), 'Shape of X_test is not consistent'
        img = np.reshape(X_test, [10000, 28, 28,1])
        lbls = y_test
    else:
        raise ValueError("dataset must be 'test' or 'train'")
    img = (img * 2 - 1).astype(np.float64)
    
    return img, lbls, len(lbls)


train_images, train_labels, _ = mnist_load('./data/mnist', dataset='train')
test_images, test_labels, _ = mnist_load('./data/mnist', dataset='test')
# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0
print(np.shape(train_images))
print(np.shape(test_images))
print(np.max(train_images))
print(np.min(train_images))
print(np.max(test_images))
print(np.min(test_images))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

model.save("./classifier_model.h5")
print("Saved model to disk")