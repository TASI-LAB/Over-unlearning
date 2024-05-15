## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

from attack_setup.vgg16 import VGG16
from attack_setup.resblock import ResBlock

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    

class CIFAR:
    def __init__(self):

        x_train, x_test = np.load('train_test_data/Cifar/X_train.npy'), np.load('train_test_data/Cifar/X_test.npy')
        x_train = np.array(x_train,dtype=np.float32)
        x_valid = np.load('train_test_data/Cifar/X_valid.npy')

        
        x_train = np.array(x_train,dtype=np.float32)
        x_train = ((x_train/255)-.5)
        x_valid = np.array(x_valid,dtype=np.float32)
        x_valid = ((x_valid/255)-.5)
        x_test = np.array(x_test,dtype=np.float32)
        x_test = ((x_test/255)-.5)
        

        y_train, y_test = np.load('train_test_data/Cifar/y_train.npy'), np.load('train_test_data/Cifar/y_test.npy')
        y_valid = np.load('train_test_data/Cifar/y_valid.npy')
        y_train = y_train.astype(int)
        y_train_one_hot = np.eye(10)[y_train]
        y_valid = y_valid.astype(int)
        y_valid_one_hot = np.eye(10)[y_valid]
        y_test = y_test.astype(int)
        y_test = np.eye(10)[y_test]
        self.test_data = x_test
        self.test_labels = y_test
        self.validation_data = x_valid
        self.validation_labels = y_valid_one_hot
        self.vl = y_valid
        self.train_data = x_train
        self.train_labels = y_train_one_hot
        self.tl = y_train

class CIFARModel:
    def __init__(self, restore=None, session=None, use_log=False, network='vgg16'):
        model = None
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        if network == 'vgg16':
            model = VGG16(restore, session, use_log, self.num_labels)
        elif network == 'vgg19':
            print("get vgg19 !!!!!!!!!!")
            model = VGG19(restore, session, use_log, self.num_labels)
        elif network == 'vgg10':
            print("get vgg10 !!!!!!!!!!")
            model = VGG10(restore, session, use_log, self.num_labels)
        elif network == 'vgg14':
            model = VGG14(restore, session, use_log, self.num_labels)
        elif network == 'resnet':
            print("get res block !!!!!!!!")
            model = ResBlock(restore, session, use_log, self.num_labels)

        self.model = model

    def predict(self, data):
        return self.model.predict(data)
        
    
