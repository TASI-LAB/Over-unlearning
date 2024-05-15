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
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

def read_labels(path_to_labels):

            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8)
                return labels

def read_all_images(path_to_data):

    with open(path_to_data, 'rb') as f:

        everything = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(everything, (-1, 3, 96, 96))

        images = np.transpose(images, (0, 3, 2, 1))

        images = images.astype(np.float32)

        images = ((images/255)-.5)

        return images
    

class STL10:
    
    def __init__(self):

        # train_data = read_all_images("train_test_data/stl10_binary/train_X.bin")
        # train_labels = read_labels("train_test_data/stl10_binary/train_y.bin")
        # test_data = read_all_images("train_test_data/stl10_binary/test_X.bin")
        # test_labels = read_labels("train_test_data/stl10_binary/test_y.bin")
        # train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.2)
        # train_labels = train_labels - 1
        # validation_labels = validation_labels - 1
        # test_labels = test_labels - 1
        train_data = np.load("train_test_data/stl10_binary/train_X_new.npy")
        train_labels = np.load("train_test_data/stl10_binary/train_y_new.npy")
        test_data = np.load("train_test_data/stl10_binary/test_X_new.npy")
        test_labels = np.load("train_test_data/stl10_binary/test_y_new.npy")
        validation_data = np.load("train_test_data/stl10_binary/valid_X_new.npy")
        validation_labels = np.load("train_test_data/stl10_binary/valid_y_new.npy")

        train_data = train_data.astype(np.float32)
        train_data = ((train_data/255)-.5)
        test_data = test_data.astype(np.float32)
        test_data = ((test_data/255)-.5)
        validation_data = validation_data.astype(np.float32)
        validation_data = ((validation_data/255)-.5)

        train_labels_one_hot = np.eye(10)[train_labels]
        validation_labels_one_hot = np.eye(10)[validation_labels]
        test_labels_onehot = np.eye(10)[test_labels]
        self.test_data = test_data
        self.test_labels = test_labels_onehot
        self.validation_data = validation_data
        self.validation_labels = validation_labels_one_hot
        self.vl = validation_labels
        self.train_data = train_data
        self.train_labels = train_labels_one_hot
        self.tl = train_labels
        
        

class STL10Model:
    def __init__(self, restore=None, session=None, use_log=False):
        self.num_channels = 3
        self.image_size = 96
        self.num_labels = 10

        n_filters = [128, 128, 128, 128, 128, 128]
        conv_params = dict(activation='relu', kernel_size=3,
                        kernel_initializer='he_uniform', padding='same')

        model = Sequential()
        # VGG block 1
        model.add(Conv2D(filters=n_filters[0], input_shape=(96,96,3), **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[1], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        # VGG block 2
        model.add(Conv2D(filters=n_filters[2], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[3], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        # VGG block 3
        model.add(Conv2D(filters=n_filters[4], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[5], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # dense and final layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        model.add(Dense(units=10, activation='softmax'))
        if restore:
            print('model path: ', restore)
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
        
    
