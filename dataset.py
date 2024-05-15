from conf import BASE_DIR

import numpy as np
import tensorflow as tf 

from sklearn.model_selection import train_test_split

import os,sys
from PIL import Image
import re
import gzip
import urllib.request


class Cifar10(object):
    dataset_dir = BASE_DIR / 'train_test_data/Cifar'

    def __init__(self, train=None,   test=None, validation=None):
        if train is not None:
            self.x_train = train[0]
            self.y_train = train[1]
        if test is not None:
            self.x_test = test[0]
            self.y_test = test[1]
        if validation is not None:
            self.x_valid = validation[0]
            self.y_valid = validation[1]

    @classmethod
    def load(cls):
        x_train, x_test = np.load(cls.dataset_dir/'X_train.npy'), np.load(cls.dataset_dir/'X_test.npy')
        x_valid = np.load(cls.dataset_dir/'X_valid.npy')
        y_train, y_test = np.load(cls.dataset_dir/'y_train.npy'), np.load(cls.dataset_dir/'y_test.npy')
        y_valid = np.load(cls.dataset_dir/'y_valid.npy')
        y_train = y_train.astype(int)
        y_train_one_hot = tf.one_hot(y_train, depth=10).numpy()
        y_valid = y_valid.astype(int)
        y_valid_one_hot = tf.one_hot(y_valid, depth=10).numpy()
        y_test = y_test.astype(int)
        y_test = tf.one_hot(y_test, depth=10).numpy()
        x_train = np.array(x_train,dtype=np.float32)
        x_train = ((x_train/255)-.5)
        x_valid = np.array(x_valid,dtype=np.float32)
        x_valid = ((x_valid/255)-.5)
        x_test = np.array(x_test,dtype=np.float32)
        x_test = ((x_test/255)-.5)
        return (x_train, y_train_one_hot), (x_test, y_test), (x_valid, y_valid_one_hot), y_train, y_valid
    
    
    
class Cifar100(object):
    dataset_dir = BASE_DIR / 'train_test_data/Cifar100'

    def __init__(self, train=None, test=None, validation=None):
        if train is not None:
            self.x_train = train[0]
            self.y_train = train[1]
            self.y_train_fine = train[2]
        if test is not None:
            self.x_test = test[0]
            self.y_test = test[1]
            self.y_test_fine = test[2]
        if validation is not None:
            self.x_valid = validation[0]
            self.y_valid = validation[1]
            self.y_valid_fine  = validation[2]

    @classmethod
    def load(cls):
        x_train, x_test = np.load(cls.dataset_dir/'X_train.npy'), np.load(cls.dataset_dir/'X_test.npy')
        x_valid = np.load(cls.dataset_dir/'X_valid.npy')
        y_train, y_test = np.load(cls.dataset_dir/'y_train_fine.npy'), np.load(cls.dataset_dir/'y_test_fine.npy')
        y_valid = np.load(cls.dataset_dir/'y_valid_fine.npy')
        y_train = y_train.astype(int)
        y_train_one_hot = tf.one_hot(y_train, depth=100).numpy()
        y_valid = y_valid.astype(int)
        
        y_valid_one_hot = tf.one_hot(y_valid, depth=100).numpy()
        y_test = y_test.astype(int)
        y_test = tf.one_hot(y_test, depth=100).numpy()

        x_train = np.array(x_train,dtype=np.float32)
        x_train = ((x_train/255)-.5)
        x_valid = np.array(x_valid,dtype=np.float32)
        x_valid = ((x_valid/255)-.5)
        x_test = np.array(x_test,dtype=np.float32)
        x_test = ((x_test/255)-.5)
        return (x_train, y_train_one_hot), (x_test, y_test), (x_valid, y_valid_one_hot), y_train, y_valid
    


class STL10(object):         

    @classmethod
    def load(cls):

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

        print(f'train_data: {train_data.shape}, train_labels: {train_labels.shape}')
        print(f'test_data: {test_data.shape}, test_labels: {test_labels.shape}')
        print(f'validation_data: {validation_data.shape}, validation_labels: {validation_labels.shape}')

        
        train_labels_one_hot = tf.one_hot(train_labels, depth=10).numpy()
        
        validation_labels_one_hot = tf.one_hot(validation_labels, depth=10).numpy()
        
        test_labels_onehot = tf.one_hot(test_labels, depth=10).numpy()
        return (train_data, train_labels_one_hot), (test_data, test_labels_onehot), (validation_data, validation_labels_one_hot), train_labels, validation_labels


class UnlearnData(object):
        
    @classmethod
    def load(self, path, orgclass=None, targclass=None, label_mode = None, model = None, dataset = None):
        img_list = []
        img_label_list = []
        img_prev_label_list = []
        img_id_list = []

        #adv_pattern = re.compile(r'adv[0-9]+_')
        prev_pattern = re.compile(r'prev[0-9]+_')
        id_pattern = re.compile(r'id[0-9]+_')
        
        for filename in os.listdir(path):
            #adv_result = adv_pattern.findall(filename)
            img_np = np.load(f'{path}/{filename}')
            img_list.append(img_np)
            if orgclass == None:
                prev_label = orgclass
            else:
                prev_result = prev_pattern.findall(filename)
                prev_label = int(prev_result[0][prev_result[0].index('v')+1: prev_result[0].index('_')])
                
            if label_mode == 0 and orgclass == None: 
                prev_result = prev_pattern.findall(filename)
                label = int(prev_result[0][prev_result[0].index('v')+1: prev_result[0].index('_')])
            elif label_mode == 0:
                label = orgclass

            if label_mode == 1 and targclass != None:
                print('meme')
                label = targclass
            elif label_mode == 1:
                return None
            
            if label_mode == 2:
                imagelist =[]
                imagelist.append(img_np)
                # y_pred = model.predict(tf.squeeze(imagelist))
                print("!!!!!!!!!!!!!!!!!!!")
                imagelist = tf.convert_to_tensor(imagelist)
                imagelist = tf.squeeze(imagelist, axis=0)
                print(imagelist.shape)
                y_pred = model.predict(imagelist)
                y_pred=np.argmax(y_pred,axis=1)
                
                label = y_pred[0]

            id_result = id_pattern.findall(filename)
            id_int = int(id_result[0][id_result[0].index('d')+1: id_result[0].index('_')])

            # img = Image.open(path+'/'+filename)
            # img_np = np.array(img)
            
            img_label_list.append(label)
            img_prev_label_list.append(prev_label)
            img_id_list.append(id_int)

        images = np.stack(img_list,axis=0)
        # images = np.array(images,dtype=np.float32)
        # images = np.array(img_list,dtype=np.float32)
        # images = ((images/255)-.5)
        #images = np.array(images)
        images = tf.squeeze(images)
        adv_labels = np.stack(img_label_list, axis=0)
        prev_labels = np.stack(img_prev_label_list, axis=0)
        ids = np.stack(img_id_list)
        if (dataset == 'cifar100'):
            y_test = tf.one_hot(adv_labels, depth=100).numpy()
        else:
            y_test = tf.one_hot(adv_labels, depth=10).numpy()
        print("prev_labels", prev_labels)
        
        return images, y_test, prev_labels, ids
            
        
    
    
    
