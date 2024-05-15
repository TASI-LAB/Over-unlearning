import os
import argparse

import numpy as np
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from util import TrainingResult, measure_time

from configs.config import Config
from dataset import Cifar10, Cifar100, STL10



from vgg16 import get_VGG
# from vgg19 import get_VGG19
# from vgg10 import get_VGG10
# from vgg14 import get_VGG14

# from resnetnew import get_Resnet50
# from resnet import get_Resnet18
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD

def train(model_init, model_folder, data, dataset, network, lr, epochs, batch_size, model_filename='best_model.hdf5', **kwargs):
    os.makedirs(model_folder, exist_ok=True)
    model_save_path = os.path.join(model_folder, model_filename)
    csv_save_path = os.path.join(model_folder, 'train_log.csv')
    result = TrainingResult(model_folder)
    (x_train, y_train), (x_test, y_test),(x_valid, y_valid), y_train_org, _= data
    
    if dataset == 'cifar10':
        model = model_init()
    elif dataset == 'cifar100':
        model = model_init(output=100, lr_init=lr)
    elif dataset == 'stl10':
        model = model_init(input_shape=(96, 96, 3))

    metric_for_min = 'loss'
    loss_ckpt = ModelCheckpoint(model_folder+"/{epoch:02d}.hdf5", monitor=metric_for_min, save_best_only=False, save_freq='epoch',period=1,
                                save_weights_only=True)
    csv_logger = CSVLogger(csv_save_path)
    callbacks = [loss_ckpt, csv_logger]
    # if network == 'resnet18':
    #     lr = LearningRateScheduler(lr_schedule)
    #     callbacks.append(lr)

    with measure_time() as t:
        hist = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(x_test, y_test), verbose=1,
                            callbacks=callbacks).history
        training_time = t()
    best_loss = np.min(hist[metric_for_min]) if metric_for_min in hist else np.inf
    best_loss_epoch = np.argmin(hist[metric_for_min]) + 1 if metric_for_min in hist else 0
    print('Best model has test loss {} after {} epochs'.format(best_loss, best_loss_epoch))
    return 


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str)
    parser.add_argument("--dataset",  type=str, default='cifar10', help="dataset name")
    parser.add_argument("--network",  type=str, default='vgg16', choices=["vgg16", "vgg19", "vgg10", "vgg14", "resnet", "resnet18", "resnet50"], help="network structure")
    parser.add_argument("--lr",  type=float, default=0.0001, help="learning rate")
    
    return parser


def main(model_folder, dataset, network, lr):
    data = None
    model_init = None
    if network == 'vgg16':
        model_init = get_VGG
    # elif network == 'resnet':
    #     model_init = get_Resnet
    # elif network == 'resnet18':
    #     model_init = get_Resnet18
    # elif network == 'resnet50':
    #     model_init = get_Resnet50
    # elif network == 'vgg19':
    #     model_init = get_VGG19
    # elif network == 'vgg10':
    #     model_init = get_VGG10
    # elif network == 'vgg14':
    #     model_init = get_VGG14
    

    if dataset == 'cifar10':
        data = Cifar10.load()
    elif dataset == 'cifar100':
        data = Cifar100.load()
    elif dataset == 'stl10':
        data = STL10.load()
    else: 
        return
        
    train_conf = os.path.join(model_folder, 'train_config.json')
    train_kwargs = Config.from_json(train_conf)
    
    train(model_init, model_folder, data, dataset, network, lr, **train_kwargs)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
