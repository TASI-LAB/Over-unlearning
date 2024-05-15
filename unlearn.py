import os
from os.path import dirname as parent
import json
import argparse
import sys
import numpy as np
from configs.config import Config
from vgg16 import get_VGG
from dataset import Cifar10, Cifar100, UnlearnData, STL10

from common import evaluate_unlearning, fine_tuning, train_retrain

import tensorflow as tf 
from util import UnlearningResult, reduce_dataset, measure_time
# from resnet import get_Resnet18, get_Resnet50
# from resblock import get_Resnet
# from vgg19 import get_VGG19

from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json',
                        help="config file with parameters for this experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable additional outputs")
    parser.add_argument("--unlearn_class", "--uc", nargs="+", type=str, default=[])
    parser.add_argument("--retrain", action="store_true", help="enable model retrain")
    parser.add_argument("--dataset",  type=str, default='cifar10', help="dataset name")
    parser.add_argument("--order", type=int, default=1, help="first/second order unlearn")
    parser.add_argument("--orgclass", type=str, default=None, help="the original class of the attack sample")
    parser.add_argument("--targclass", type=str, default=None, help="the class we want to impact")
    parser.add_argument("--label_mode", type=int, default=0, help="the label we want to give for unlearn sample, 0: origin class, 1: target class, 2: model prediction")
    parser.add_argument("--unlearn_attack_with_path",  type=str, default='', help="unlearn dataset path")
    parser.add_argument("--log_path", type=str, default='models/log', help="Path for save log")
    parser.add_argument("--random_seed", type=str, default='88', help="sample random choice seed")
    parser.add_argument("--unlearn_type", type=int, default=3, help="unlearn approximate type")
    parser.add_argument("--network",  type=str, default='vgg16', choices=["vgg16", "vgg19", "vgg10", "resnet"], help="network structure")
    return parser


def run_experiment(model_folder, unlearn_kwargs, verbose=False, unlearn_ratio=1.0, 
                   unlearn_class=[], retrain=False, order=2, unlearn_attack_with_path='', 
                   log_path='models/log', random_seed=42, unlearn_type=0, dataset='cifar10', network='vgg16'):
    
    certainmodel = None
    if network == 'vgg16':
        certainmodel = get_VGG
    elif network == 'vgg19':
        certainmodel = get_VGG19
    elif network == 'vgg10':
        certainmodel = get_VGG10
    elif network == 'resnet':
        certainmodel = get_Resnet

    if (dataset == 'cifar100'):
        label_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 
                                'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 
                                'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 
                                'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 
                                'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
                                'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        alllabels = [i for i in range(100)]
        data = Cifar100.load()
        model_init = None
        model_init = (certainmodel, {'output':100})

    elif (dataset == 'cifar10'):
        label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        alllabels = [i for i in range(10)]
        data = Cifar10.load()
        model_init = (certainmodel, {})
    elif (dataset == 'stl10'):
        label_name = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        alllabels = [i for i in range(10)]
        data = STL10.load()
        model_init = (certainmodel, {'input_shape':(96, 96, 3)})

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid), y_train_org, y_valid_org = data
    data = (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
    
    clear_filename = 'best_model.hdf5'
    unlearn_filename = 'unlearn_filename.hdf5'
    
    trainId = range(len(x_train))
    
    np.random.seed(random_seed)
    
    if len(unlearn_class) > 0:
        removedIds = []
        removed_y_train_random = []
        labels = [label_name.index(i) for i in unlearn_class]
        for i in range(len(y_train_org)):
            if y_train_org[i] in labels:
                potentialLabels = [item for item in alllabels if item != y_train_org[i]]
                removed_y_train_random.append(np.random.choice(potentialLabels, 1 ,replace = False)[0])
                removedIds.append(i)

        removedIds = np.array(removedIds)  
        validIds = []
        validremainIds = []
        for i in range(len(y_valid_org)):
            if y_valid_org[i] in labels:
                validIds.append(i)
            else:
                validremainIds.append(i)
            
        validIds = np.array(validIds)  
        rationRemoved = np.random.choice(range(len(removedIds)), int(unlearn_ratio*len(removedIds)),replace = False)

        removedIds = np.array(removedIds)[rationRemoved]
        removed_y_train_random = np.array(removed_y_train_random)[rationRemoved]
    else:
        validIds = []
        validremainIds = [i for i in range(len(y_valid_org))]
        removedIds = np.random.choice(trainId, int(unlearn_ratio*len(x_train)),replace = False)
        removed_y_train_random = np.random.choice(alllabels, int(unlearn_ratio*len(x_train)),replace = True)

    if (dataset == 'cifar100'):
        removed_y_train_random = tf.one_hot(removed_y_train_random, depth=100).numpy()
    else:
        removed_y_train_random = tf.one_hot(removed_y_train_random, depth=10).numpy()


    removedData = (x_train[removedIds], y_train[removedIds])
    validRemovedClass = (x_valid[validIds], y_valid[validIds])
    validRemainClass = (x_valid[validremainIds], y_valid[validremainIds])

    log_path = f'{log_path}-ratio-{unlearn_ratio}'
    order_unlearning(model_folder, clear_filename, unlearn_filename, model_init, 
                    data, removedIds, removedData, 
                    validRemovedClass, validRemainClass,removed_y_train_random, 
                    unlearn_kwargs, verbose=verbose, retrain=retrain, 
                    order=order, dataset=dataset, log_path=log_path, unlearn_type=unlearn_type)




def run_experiment_unlearn_with_image_input(model_folder, unlearn_kwargs, verbose=False, unlearn_ratio=1.0, 
                                            unlearn_class=[], retrain=False, order=2, orgclass=None, targclass=None, label_mode=0,
                                            unlearn_attack_with_path='', dataset='cifar10', 
                                            log_path='models/log', random_seed=42, unlearn_type=0, network='vgg16'):
    certainmodel = None
    if network == 'vgg16':
        certainmodel = get_VGG
    # elif network == 'vgg19':
    #     certainmodel = get_VGG19
    # elif network == 'vgg10':
    #     certainmodel = get_VGG10
    # elif network == 'resnet':
    #     certainmodel = get_Resnet
    


    if (dataset == 'cifar100'):
        label_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 
                      'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 
                      'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 
                      'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 
                      'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                      'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
                      'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        alllabels = [i for i in range(100)]
        model_init = None
        model_init = (certainmodel, {'output':100})

        data = Cifar100.load()
    elif (dataset == 'cifar10'):
        label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        alllabels = [i for i in range(10)]
        model_init = (certainmodel, {})
        data = Cifar10.load()
    elif (dataset == 'mnist'):
        label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alllabels = [i for i in range(10)]
        data = MNIST.load()
        model_init = (certainmodel, {'input_shape':(28, 28, 1)})
    elif (dataset == 'stl10'):
        label_name = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        alllabels = [i for i in range(10)]
        data = STL10.load()
        model_init = (certainmodel, {'input_shape':(96, 96, 3)})

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid), y_train_org, y_valid_org = data
    data = (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
    
    clear_filename = 'best_model.hdf5'
    unlearn_filename = 'unlearn_filename.hdf5'
    if label_mode == 2:
        (model_init_f, model_init_p) = model_init
        curmodel = model_init_f(**model_init_p)
        model_weights = os.path.join(model_folder, clear_filename)
        curmodel.load_weights(model_weights)
        unlearn_imgs, unlearn_labels, orginalclass, ids = UnlearnData.load(unlearn_attack_with_path, label_mode =label_mode, model = curmodel, dataset=dataset)
    elif orgclass is not None and targclass is not None:
        unlearn_imgs, unlearn_labels, orginalclass, ids = UnlearnData.load(unlearn_attack_with_path, orgclass=label_name.index(orgclass), targclass=label_name.index(targclass), label_mode =label_mode,dataset=dataset)
    elif orgclass is not None: 
        unlearn_imgs, unlearn_labels, orginalclass, ids = UnlearnData.load(unlearn_attack_with_path, orgclass=label_name.index(orgclass), label_mode =label_mode, dataset=dataset)
    else:
        unlearn_imgs, unlearn_labels, orginalclass, ids = UnlearnData.load(unlearn_attack_with_path, label_mode =label_mode, dataset=dataset)


    
    removed_y_train_random = []
    labels = set(orginalclass)
    if orgclass is not None:
        labels = set([label_name.index(orgclass)])
    np.random.seed(random_seed)
    for i in range(len(unlearn_imgs)):
        potentialLabels = [item for item in alllabels if item != orginalclass[i]]
        removed_y_train_random.append(np.random.choice(potentialLabels, 1 ,replace = False)[0])

    validIds = []
    validremainIds = []
    for i in range(len(y_valid_org)):
        if y_valid_org[i] in labels:
            validIds.append(i)
        else:
            validremainIds.append(i)
            
    validIds = np.array(validIds)  
    removedIds = range(len(unlearn_imgs))
    print("Unlearning attack samples: ", len(removedIds))
    rationRemoved = np.random.choice(range(len(removedIds)),int( (len(x_train) / len(label_name))*unlearn_ratio) ,replace = False)
    print("Unlearning attack samples: ", len(rationRemoved))
    removedIds = np.array(removedIds)[rationRemoved]
    removedtrainIds = np.array(ids)[removedIds]
    print("total sample",len(removedtrainIds))


    removedtrainImgs = np.array(x_train)[removedtrainIds]
    removedtrainAdvLabels = np.array(y_train)[removedtrainIds]
    
    removed_y_train_random = np.array(removed_y_train_random)[removedIds]
    if (dataset == 'cifar100'):
        removed_y_train_random = tf.one_hot(removed_y_train_random, depth=100).numpy()
    else:
        removed_y_train_random = tf.one_hot(removed_y_train_random, depth=10).numpy()

    removedData = (np.array(unlearn_imgs)[removedIds], np.array(unlearn_labels)[removedIds])
    removedOrigData = (removedtrainImgs, removedtrainAdvLabels)

    validRemovedClass = (x_valid[validIds], y_valid[validIds])
    validRemainClass = (x_valid[validremainIds], y_valid[validremainIds])

    valid_orgc_near_attc = None
    valid_attc_near_orgc = None
    valid_attc_set = None
    if targclass != None and orgclass != None:
        orgclass = label_name.index(orgclass) 
        targclass = label_name.index(targclass)

        valid_attc_Ids = []
        for i in range(len(y_valid_org)):
            if y_valid_org[i] == targclass:
                valid_attc_Ids.append(i)

        valid_attc = (x_valid[valid_attc_Ids], y_valid[valid_attc_Ids])
        valid_attc_set = (valid_attc_near_orgc, valid_attc)
    

    with open(f'{log_path}-ratio-{unlearn_ratio}_attack', 'a') as f:
        f.write("attack\n")
    order_unlearning_with_image_input(model_folder, clear_filename, unlearn_filename, 
                                    model_init, data, removedData, validRemovedClass, validRemainClass,
                                    removed_y_train_random, unlearn_kwargs, verbose=verbose, retrain=retrain, 
                                    order=order,dataset=dataset, 
                                    log_path=f'{log_path}-ratio-{unlearn_ratio}_attack', 
                                    valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set,
                                    unlearn_type=unlearn_type)

    with open(f'{log_path}-ratio-{unlearn_ratio}_original', 'a') as f:
        f.write("org\n")
    order_unlearning_with_image_input(model_folder, clear_filename, unlearn_filename, 
                                    model_init, data, removedOrigData, validRemovedClass, validRemainClass,
                                    removed_y_train_random, unlearn_kwargs, verbose=verbose, 
                                    retrain=retrain, order=order,dataset=dataset, 
                                    log_path=f'{log_path}-ratio-{unlearn_ratio}_original', 
                                    valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set,
                                    unlearn_type=unlearn_type)


def order_unlearning_with_image_input(model_folder, clear_filename, unlearn_filename, 
                                    model_init, data, removedData, validRemovedClass,validRemainClass,
                                    removed_y_train_random, unlearn_kwargs, verbose=False, retrain=False, 
                                    order=2, dataset='cifar10', log_path='models/log',  
                                    valid_orgc_near_attc = None, valid_attc_set=None,
                                    unlearn_type=0):
    unlearning_result = UnlearningResult(model_folder)
    log_dir = model_folder
    init_weight = os.path.join(model_folder, clear_filename)
    (model_init_f, model_init_p) = model_init
    curmodel = model_init_f(**model_init_p)
    curmodel.load_weights(init_weight)
    acc = curmodel.evaluate(removedData[0], removedData[1], verbose=0)[1]
    acc_finetune_before, acc_finetune_after, finetune_duration_s =fine_tuning(model_init, init_weight, data, 
                                                                            removed_y_train_random, removedData, 
                                                                            validRemovedClass, validRemainClass,
                                                                            repaired_filepath='finetune_model.hdf5', 
                                                                            log_dir=log_dir, log_path=log_path, 
                                                                            unlearn_kwargs=unlearn_kwargs,  
                                                                            valid_orgc_near_attc = valid_orgc_near_attc, 
                                                                            valid_attc_set=valid_attc_set)


    clean_acc = acc_finetune_before
    cm_dir = os.path.join(model_folder, 'cm')
    os.makedirs(cm_dir, exist_ok=True)


    
    if len(unlearn_kwargs['tau_list']) > 0:
        for i in range(len(unlearn_kwargs['tau_list'])):
            unlearn_kwargs['tau'] = unlearn_kwargs['tau_list'][i]
            acc_before, acc_after, diverged, logs, unlearning_duration_s = evaluate_unlearning(model_init, init_weight, data, 
                                                                                               removedData, validRemovedClass, validRemainClass,
                                                                                               removed_y_train_random, unlearn_kwargs, clean_acc=clean_acc,
                                                                                                unlearn_filename=unlearn_filename, verbose=verbose, cm_dir=cm_dir, 
                                                                                                log_dir=log_dir, order=order, log_path=log_path, 
                                                                                                valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set,
                                                                                                unlearn_type=unlearn_type)
    else:
        acc_before, acc_after, diverged, logs, unlearning_duration_s = evaluate_unlearning(model_init, init_weight, data, 
                                                                                           removedData, validRemovedClass, validRemainClass,
                                                                                           removed_y_train_random, unlearn_kwargs, clean_acc=clean_acc,
                                                                                            unlearn_filename=unlearn_filename, verbose=verbose, cm_dir=cm_dir, 
                                                                                            log_dir=log_dir, order=order, log_path=log_path, 
                                                                                            valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set,
                                                                                            unlearn_type=unlearn_type)
    acc_perc_restored = (clean_acc) - (acc_after)

    unlearning_result.update({
        'acc_clean': clean_acc,
        'acc_before': acc_before,
        'acc_after_finetune': acc_finetune_after,
        'acc_after_approx': acc_after,
        'acc_diff': acc_perc_restored,
        'diverged': diverged,
        'n_gradients': sum(logs),
        'unlearning_duration_s': unlearning_duration_s,
        'finetune_duration_s': finetune_duration_s
    })
        
    unlearning_result.save()


def get_edge_sample(validarray, label, nearclass):
    
    sample_with_label = list(filter(lambda x: x['predict label']==label and x['original label']==label, validarray))
    # sorted_by_class = sample_with_label.sortedby(x: x['predict probs'][nearclass])
    def get_element(element):
        return element['predict probs'][nearclass]
    sample_with_label.sort(key=get_element, reverse=True)
    

    return sample_with_label[: int(len(sample_with_label)*0.10)]

def grab_edge_samples(orgc, attc, record_path):
    validarray = []
    with open(record_path, "r") as f:
        validarray = json.load(f)
    orgc_near_acc = get_edge_sample(validarray, orgc, attc)
    orgc_near_acc_id = list(map(lambda x: x['id'], orgc_near_acc))
    attc_near_org = get_edge_sample(validarray, attc,orgc)
    attc_near_org_id = list(map(lambda x: x['id'], attc_near_org))

    return orgc_near_acc_id, attc_near_org_id


def order_unlearning(model_folder, clear_filename, unlearn_filename, model_init, 
                    data, removedIds, removedData, 
                    validRemovedClass, validRemainClass, removed_y_train_random, 
                    unlearn_kwargs, verbose=False, retrain=False, 
                    order=2,dataset='cifar10', log_path='models/log', unlearn_type=0):
    unlearning_result = UnlearningResult(model_folder)
    log_dir = model_folder
    
    (x_train, y_train), test, val = data
    init_weight = os.path.join(model_folder, clear_filename)
    trainId = range(len(x_train))
    if retrain:
        idAfterRemove = np.delete(trainId, removedIds)
        dataAfterRemoved = (x_train[idAfterRemove], y_train[idAfterRemove])
        (model_init_f, model_init_p) = model_init
        model = model_init_f(**model_init_p)
        model.load_weights(init_weight)
        with open(log_path, 'a') as f:
                f.write(f'''>> Retraining\n''') 
        best_model, best_loss, acc_unlearn = train_retrain(model, dataAfterRemoved, val, test, model_folder, 
                                                        data, removedData, validRemovedClass, 
                                                        validRemainClass, dataset, log_path=log_path)
    
    with open(log_path, 'a') as f:
                f.write(f'''>> Fine_tuning\n''') 
    acc_finetune_before, acc_finetune_after, finetune_duration_s =fine_tuning(model_init, init_weight, data, 
                                                                            removed_y_train_random, removedData, 
                                                                            validRemovedClass, validRemainClass,
                                                                            repaired_filepath='finetune_model.hdf5', 
                                                                            log_path=log_path, unlearn_kwargs=unlearn_kwargs)

    clean_acc = acc_finetune_before
    repaired_filepath = os.path.join(model_folder, clear_filename)
    cm_dir = os.path.join(model_folder, 'cm')
    os.makedirs(cm_dir, exist_ok=True)
    if len(unlearn_kwargs['tau_list']) > 0:
        for i in range(len(unlearn_kwargs['tau_list'])):
            unlearn_kwargs['tau'] = unlearn_kwargs['tau_list'][i]
            with open(log_path, 'a') as f:
                f.write(f'''>> tau: {unlearn_kwargs['tau']}\n''') 
            acc_before, acc_after, diverged, logs, unlearning_duration_s = evaluate_unlearning(model_init, init_weight, data, 
                                                                                                removedData, validRemovedClass, validRemainClass,
                                                                                               removed_y_train_random, unlearn_kwargs, clean_acc=clean_acc,
                                                                                                unlearn_filename=unlearn_filename, verbose=verbose, cm_dir=cm_dir, 
                                                                                                log_dir=log_dir, order=order, log_path=log_path, unlearn_type=unlearn_type)
    else:
        acc_before, acc_after, diverged, logs, unlearning_duration_s = evaluate_unlearning(model_init, init_weight, data, 
                                                                                        removedData, validRemovedClass, validRemainClass,
                                                                                        removed_y_train_random, unlearn_kwargs, clean_acc=clean_acc,
                                                                                        unlearn_filename=unlearn_filename, verbose=verbose, cm_dir=cm_dir, 
                                                                                        log_dir=log_dir, order=order, log_path=log_path, unlearn_type=unlearn_type)
    acc_perc_restored = (clean_acc) - (acc_after)

    if retrain:
        unlearning_result.update({
            'acc_clean': clean_acc,
            'acc_before': acc_before,
            'acc_after_certif': acc_unlearn,
            'acc_after_finetune': acc_finetune_after,
            'acc_after_approx': acc_after,
            'acc_diff': acc_perc_restored,
            'diverged': diverged,
            'n_gradients': sum(logs),
            'unlearning_duration_s': unlearning_duration_s,
            'finetune_duration_s': finetune_duration_s
        })
    else:
        unlearning_result.update({
            'acc_clean': clean_acc,
            'acc_before': acc_before,
            'acc_after_finetune': acc_finetune_after,
            'acc_after_approx': acc_after,
            'acc_diff': acc_perc_restored,
            'diverged': diverged,
            'n_gradients': sum(logs),
            'unlearning_duration_s': unlearning_duration_s,
            'finetune_duration_s': finetune_duration_s
        })
        
    unlearning_result.save()


def main(model_folder, config_file, verbose, unlearn_class, 
        retrain, dataset, order, orgclass, targclass, label_mode, unlearn_attack_with_path, 
        log_path, random_seed, unlearn_type, network):
    config_file = os.path.join(model_folder, config_file)
    unlearn_kwargs = Config.from_json(config_file)
    
    random_seed = int(random_seed)
    if len(unlearn_attack_with_path) > 0:
        for i in range(len(unlearn_kwargs['ratio'])):
            ratio = unlearn_kwargs['ratio'][i]
            print('current ratio:', ratio )
            run_experiment_unlearn_with_image_input(model_folder, unlearn_kwargs, verbose=verbose, 
                                                unlearn_ratio=ratio, unlearn_class=unlearn_class, 
                                                retrain=retrain, order=order, orgclass=orgclass, targclass=targclass, label_mode=label_mode,
                                                unlearn_attack_with_path=unlearn_attack_with_path, 
                                                dataset=dataset, log_path=log_path, random_seed=random_seed, 
                                                unlearn_type=unlearn_type, network=network)
    else:
        for i in range(len(unlearn_kwargs['ratio'])):
            ratio = unlearn_kwargs['ratio'][i]
            print('current ratio:', ratio )
            run_experiment(model_folder, unlearn_kwargs, verbose=verbose, 
                        unlearn_ratio=ratio, unlearn_class=unlearn_class, 
                        retrain=retrain, order=order, 
                        unlearn_attack_with_path=unlearn_attack_with_path, 
                        dataset=dataset, log_path=log_path, random_seed=random_seed, 
                        unlearn_type=unlearn_type, network=network)
    
    

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
