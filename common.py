import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical
from vgg16 import get_VGG
from util import LoggedGradientTape,UnlearningResult, reduce_dataset, measure_time
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.backend import clear_session

from util import LoggedGradientTape, ModelTmpState, CSVLogger, measure_time, GradientLoggingContext
from core import approx_retraining


def evaluate_model_diff(method, model, new_model, x_valid, y_valid, remove_x,remove_y, val_remove_x,val_remove_y, val_remain_x,val_remain_y, diverged=False, verbose=False, log_dir=None, log_path='models/log',valid_orgc_near_attc = None, valid_attc_set=None):

    if method == 'finetune':
        y_pred = model.predict(x_valid)
        y_pred=np.argmax(y_pred,axis=1)

        with open(log_path, 'a') as f:
            f.write("model orig martix: \n")
            f.write(classification_report(np.argmax(y_valid,axis=1), y_pred))

    acc_before_fix = model.evaluate(x_valid, y_valid, verbose=0)[1]
    acc_after_fix = -1
    if not diverged:
        acc_after_fix = new_model.evaluate(x_valid, y_valid, verbose=0)[1]
    
    acc_for_moved_class = new_model.evaluate(remove_x,remove_y, verbose=0)[1]
    acc_for_moved_class_org = model.evaluate(remove_x,remove_y, verbose=0)[1]

    acc_for_moved_val = new_model.evaluate(val_remove_x,val_remove_y, verbose=0)[1]
    acc_for_moved_val_org = model.evaluate(val_remove_x,val_remove_y, verbose=0)[1]
    
    acc_restored = (acc_after_fix - acc_before_fix) 
    
    acc_without_moved_class_val_org = model.evaluate(val_remain_x,val_remain_y, verbose=0)[1]
    acc_without_moved_class_val = new_model.evaluate(val_remain_x,val_remain_y, verbose=0)[1]

    targetattacklog = ''

    if valid_orgc_near_attc is not None:
        (valid_orgc_near_attc_X, valid_orgc_near_attc_Y) = valid_orgc_near_attc
        targetattacklog += '\ntotal sample orgc_near_attc: ' + str(len(valid_orgc_near_attc_X)) + '\n'
        acc_valid_orgc_near_attc = new_model.evaluate(valid_orgc_near_attc_X,valid_orgc_near_attc_Y, verbose=0)[1]


        y_pred_cc = new_model.predict(valid_orgc_near_attc_X)
        y_pred_cc=np.argmax(y_pred_cc,axis=1)

        acc_valid_orgc_near_attc_org = model.evaluate(valid_orgc_near_attc_X,valid_orgc_near_attc_Y, verbose=0)[1]
        targetattacklog += f"acc_orgc_near_attc org: {acc_valid_orgc_near_attc_org}, after: {acc_valid_orgc_near_attc}\nPredict label: {y_pred_cc}\n"

    if valid_attc_set is not None:
        aa, (valid_attc_X, valid_attc_Y)  = valid_attc_set
        acc_valid_attc = new_model.evaluate(valid_attc_X, valid_attc_Y, verbose=0)[1]
        acc_valid_attc_org = model.evaluate(valid_attc_X, valid_attc_Y, verbose=0)[1]
        targetattacklog += f"acc_attc_class org: {acc_valid_attc_org}, after: {acc_valid_attc}\n"

    y_pred_prob = new_model.predict(x_valid)
    y_pred = np.argmax(y_pred_prob,axis=1)

    y_pred_org_prob = model.predict(x_valid)
    y_pred_org = np.argmax(y_pred_org_prob,axis=1)

    y_valid_int = np.argmax(y_valid,axis=1)
    y_pred_wrong_prediction = []

    targetclass = np.argmax(val_remove_y,axis=1)[0]
    # go through each element in arr
    total = 0
    for index,element in enumerate(y_pred):
        if y_pred_org[index] == y_valid_int[index] and y_pred_org[index] == targetclass:
            total += 1

    for index,element in enumerate(y_pred):
        if element != y_valid_int[index] and y_pred_org[index] == y_valid_int[index] and y_pred_org[index] == targetclass:
            y_pred_wrong_prediction.append({"index": index, "org_prob": y_pred_org_prob[index], "after_prob": y_pred_prob[index], "after_predict": element})
    
    predict_counts={}
    total_samples = len(y_pred_wrong_prediction)
    for wrong_prediction in y_pred_wrong_prediction:
        with open(log_path + '_wrong predict', 'a') as f:
            f.write(f"{wrong_prediction} \n")
        predict = wrong_prediction["after_predict"]
        if predict in predict_counts:
            predict_counts[predict]["count"]  += 1
        else:
            predict_counts[predict] = {"count": 1, "percentage": 0}

    for predict in predict_counts:
        count = predict_counts[predict]["count"]
        predict_counts[predict]["percentage"] = (count / total_samples) * 100


    targetattacklog += f"Wrong Predict for original class: {predict_counts}, total: {total}"
    
    
    with open(log_path, 'a') as f:
        f.write(classification_report(np.argmax(y_valid,axis=1), y_pred))
    
    print(f'''>> 
        acc_before={acc_before_fix}, 
        acc_for_moved_class_org={acc_for_moved_class_org}, 
        acc_for_moved_val_org={acc_for_moved_val_org},
        acc_without_moved_class_val_org={acc_without_moved_class_val_org}\n'''
        f'''------------------------------
        acc_after={acc_after_fix},
        acc_for_moved_class={acc_for_moved_class}, 
        acc_for_moved_val={acc_for_moved_val},
        acc_without_moved_class_val={acc_without_moved_class_val},
        '''
        f'''
        {targetattacklog}\n''')
    
    with open(log_path, 'a') as f:
        f.write(f'''>> 
        acc_before={acc_before_fix},
        acc_for_moved_class_org={acc_for_moved_class_org}, 
        acc_for_moved_val_org={acc_for_moved_val_org},
        acc_without_moved_class_val_org={acc_without_moved_class_val_org}\n'''
        f'''------------------------------
        acc_after={acc_after_fix}, 
        acc_for_moved_class={acc_for_moved_class}, 
        acc_for_moved_val={acc_for_moved_val},
        acc_without_moved_class_val={acc_without_moved_class_val},
        \n'''
        f'''
        {targetattacklog}\n''')

    return acc_before_fix, acc_after_fix, diverged

def evaluate_unlearning(model_init, model_weights, data, 
                        removedData, validRemovedClass, validRemainClass, removed_y_train_random, 
                        unlearn_kwargs, unlearn_filename=None,
                        clean_acc=1.0, verbose=False, cm_dir=None, 
                        log_dir=None, order=2, log_path='models/log', 
                        valid_orgc_near_attc = None, valid_attc_set=None, unlearn_type=0):
    clear_session()
    (x_train, y_train), _, (x_valid, y_valid) = data
    (val_remove_x,val_remove_y) =  validRemovedClass
    (remove_x,remove_y) =  removedData
    (val_remain_x,val_remain_y) =  validRemainClass
    (model_init_f, model_init_p) = model_init
    model = model_init_f(**model_init_p)
    model.load_weights(model_weights)
    new_theta, diverged, logs, duration_s = unlearn_update(x_train, y_train, removedData, model, 
                                                        x_valid, y_valid, validRemovedClass, validRemainClass, 
                                                        removed_y_train_random,unlearn_kwargs, 
                                                        verbose=verbose, cm_dir=cm_dir, 
                                                        log_dir=log_dir, order=order, unlearn_type=unlearn_type)

    (model_init_f, model_init_p) = model_init
    new_model = model_init_f(**model_init_p)
    new_model.set_weights(new_theta)
    # if unlearn_filename is not None:
    #     new_model.save_weights(unlearn_filename)
        
    
    acc_before, acc_after, diverged = evaluate_model_diff( 'Unlearn_tau=' + str(unlearn_kwargs['tau']) + '_type=' + str(unlearn_type) + '_order=' + str(order),
        model, new_model, x_valid, y_valid, remove_x, remove_y, val_remove_x,val_remove_y, val_remain_x,val_remain_y ,diverged=diverged, verbose=verbose, log_dir=log_dir, log_path=log_path, valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set)
    return acc_before, acc_after, diverged, logs, duration_s


def unlearn_update(z_x, z_y, removedData, model, x_val, y_val, validRemovedClass, validRemainClass, removed_y_train_random, unlearn_kwargs,
                verbose=False, cm_dir=None, log_dir=None, order=2, unlearn_type=0):
    (remove_x,remove_y) = removedData
    remove_x = tf.constant(remove_x, dtype=tf.float32)
    remove_y = tf.constant(remove_y, dtype=tf.int32)
    with GradientLoggingContext('unlearn'):

        new_theta, diverged, duration_s = iter_approx_retraining(z_x, z_y, model, x_val, y_val, remove_x, remove_y, 
                                                                validRemovedClass, validRemainClass, removed_y_train_random, 
                                                                verbose=verbose, cm_dir=cm_dir, log_dir=log_dir, unlearn_type=unlearn_type,
                                                                order=order, **unlearn_kwargs)
    return new_theta, diverged, LoggedGradientTape.logs['unlearn'], duration_s


def iter_approx_retraining(z_x, z_y, model, x_val, y_val, remove_x, remove_y, 
                           validRemovedClass, validRemainClass, removed_y_train_random, 
                           hvp_batch_size=256, max_inner_steps=1,
                           steps=1, verbose=False, cm_dir=None, log_dir=None, unlearn_type=0,
                           order=2, **unlearn_kwargs):
    """Iterative approximate retraining.

    Args:
        z_x (np.ndarray): Original features.
        z_y (np.ndarray): Original labels.
        z_x_delta (np.ndarray): Changed features.
        z_y_delta (np.ndarray): Changed labels.
        delta_idx (np.ndarray): Indices of the data to change.
        steps (int, optional): Number of iterations. Defaults to 1.
        mixing_ratio (float, optional): Ratio of unchanged data to mix in. Defaults to 1.
        cm_dir (str, optional): If provided, plots confusion matrices afrer each iterations into this directory.
                                Defaults to None.
        verbose (bool, optional): Verbosity switch. Defaults to False.

    Returns:
        list: updated model parameters
        bool: whether the LiSSA algorithm diverged
    """

    # setup loggers
    if log_dir is None:
        step_logger, batch_logger, hvp_logger = None, None, None
    else:
        step_logger = CSVLogger('step', ['step', 'batch_acc', 'val_acc', 'delta_size',
                                        'new_errors', 'remaining_delta'], os.path.join(log_dir, 'log_step.csv'))
        batch_logger = CSVLogger('batch', ['step', 'inner_step', 'batch_acc'], os.path.join(log_dir, 'log_batch.csv'))
        hvp_logger = CSVLogger('hvp', ['step', 'inner_step', 'i', 'update_norm'], os.path.join(log_dir, 'log_hvp.csv'))
        acc_logger = CSVLogger('hvp', ['step', 'inner_step', 'i', 'update_norm'], os.path.join(log_dir, 'log_hvp.csv'))

    model_weights = model.get_weights()
    analysis_time = 0  # allow for additional (slow) analysis code that is not related to the algorithm itself
    # the TmpState context managers restore the states of weights, z_x, z_y, ... afterwards
    (val_remove_x,val_remove_y) =  validRemovedClass
    (val_remain_x,val_remain_y) =  validRemainClass
    val_acc_after_list = []
    acc_without_moved_class_val_list = []
    acc_for_moved_val_list = []
    diverged = None
    with measure_time() as total_timer, ModelTmpState(model):

        #idx, prio_idx = get_delta_idx(model, z_x, z_y_delta, hvp_batch_size)
        batch_acc_before = model.evaluate(z_x, z_y, verbose=0)[1]
        for step in range(0, steps+1):
            with measure_time() as t:
                val_acc_before = model.evaluate(x_val, y_val, verbose=0)[1]
                analysis_time += t()
            if step == 0:
                # calc initial metrics in step 0
                batch_acc_after = batch_acc_before
                val_acc_after = val_acc_before
            else:
                # fixed arrays during unlearning
                for istep in range(1, max_inner_steps+1):
                    hvp_logger.step = step
                    hvp_logger.inner_step = istep
                    # update model prediction after each model update
                    new_theta, diverged = approx_retraining(model, remove_x, remove_y, removed_y_train_random, order=order, 
                                                            hvp_x=z_x, hvp_y=z_y, hvp_logger=hvp_logger, 
                                                            unlearn_type=unlearn_type, **unlearn_kwargs)
                    
                    # don't update if the LiSSA algorithm diverged
                    if diverged:
                        break
                    # update weights
                    model_weights[-len(new_theta):] = new_theta
                    model.set_weights(model_weights)
                    batch_acc_after = model.evaluate(z_x, z_y, verbose=0)[1]
                    if verbose:
                        print(f"> {istep}: batch_acc = {batch_acc_after}")
                    if batch_logger is not None:
                        batch_logger.log(step=step, inner_step=istep, batch_acc=batch_acc_after)
                    if batch_acc_after == 1.0:
                        break
                with measure_time() as t:
                    val_acc_after = model.evaluate(x_val, y_val, verbose=0)[1]
                    analysis_time += t()
                    # x_valid_without_removed_class = np.setdiff1d(x_val, val_remove_x, True)
                    # y_valid_without_removed_class = np.setdiff1d(y_val, val_remove_y)
                    acc_without_moved_class_val = model.evaluate(val_remain_x,val_remain_y, verbose=0)[1]
                    acc_for_moved_val = model.evaluate(val_remove_x,val_remove_y, verbose=0)[1]
                    val_acc_after_list.append(val_acc_after)
                    acc_without_moved_class_val_list.append(acc_without_moved_class_val)
                    acc_for_moved_val_list.append(acc_for_moved_val)
            
            # get index of next delta set
            with measure_time() as t:
                
                print(f">> iterative approx retraining: step = {step}, train_acc (before/after) = {batch_acc_before} / {batch_acc_after}, "
                        f"val_acc = {val_acc_before} / {val_acc_after}")
                
                if cm_dir is not None:
                    title = f'After Unlearning Step {step}' if step > 0 else 'Before Unlearning'
                    plot_cm(x_val, y_val, model, title=title,
                            outfile=os.path.join(cm_dir, f'cm_unlearning_{step:02d}.png'))
                analysis_time += t()
        duration_s = total_timer() - analysis_time
    plot_acc(val_acc_after_list, acc_without_moved_class_val_list, acc_for_moved_val_list, cm_dir)
    return model_weights, diverged, duration_s


def get_delta_idx(model, x, y, batch_size, return_acc=True):
    y_pred = np.argmax(batch_pred(model, x), axis=1)
    prio_idx = np.argwhere(y_pred != np.argmax(y, axis=1))[:, 0]
    print(f">> {len(prio_idx)} samples in prio idx")
    idx = np.random.choice(prio_idx, min(batch_size, len(prio_idx)), replace=False)
    return idx, prio_idx


def get_mixed_delta_idx(delta_idx, n_samples, mixing_ratio=1.0, prio_idx=None):
    """Mix regular training data into delta set.

    Args:
        delta_idx (np.ndarray): Indices of the data to unlearn.
        n_samples (int): Total number of samples.
        mixing_ratio (float, optional): Ratio of regular data points to mix in. Defaults to 1.0.
        prio_idx (np.ndarray, optional): Indices of training samples to prioritize during unlearning.
                                                Defaults to None.

    Returns:
        np.ndarray: Indeces of delta samples with added regular data.
    """
    if mixing_ratio == 0.0:
        return delta_idx

    priority_idx = list(set(prio_idx) - set(delta_idx)) if prio_idx is not None else []
    if mixing_ratio == -1:
        return np.hstack((delta_idx, priority_idx)).astype(np.int)

    remaining_idx = list(set(range(n_samples)) - set(delta_idx) - set(priority_idx))
    n_total = np.ceil(mixing_ratio*delta_idx.shape[0]).astype(np.int) + delta_idx.shape[0]
    n_prio = min(n_total, len(priority_idx))
    n_regular = max(n_total - len(priority_idx) - len(delta_idx), 0)
    idx = np.hstack((
        delta_idx,
        np.random.choice(priority_idx, n_prio, replace=False),
        np.random.choice(remaining_idx, n_regular, replace=False)))
    return idx.astype(np.int)


def plot_cm(x, y_true, model, title='confusion matrix', outfile=None):
    y_pred = np.argmax(batch_pred(model, x), axis=1)
    y_true = np.argmax(y_true, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    df_cm = pd.DataFrame(cm, range(n_classes), range(n_classes))
    sns.set(font_scale=1.4)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax, cbar=False)
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile, dpi=300)
    plt.close()

def plot_acc(val_acc_after_list, acc_without_moved_class_val_list, acc_for_moved_val_list, path):
    x = [i for i in range(len(val_acc_after_list))]
    plt.plot(x, val_acc_after_list, marker='*', ms=10, label="val_after")
    plt.plot(x, acc_without_moved_class_val_list, marker='*', ms=10, label="without_moved_val")
    plt.plot(x, acc_for_moved_val_list, marker='*', ms=10, label="for_moved_val")
    plt.xticks(rotation=45)
    plt.xlabel("Steps")
    plt.ylabel("Acc")
    plt.title("Acc Analysis")
    plt.legend(loc="upper left")
    for y in [val_acc_after_list, acc_without_moved_class_val_list, acc_for_moved_val_list]:
        for x1, yy in zip(x, y):
            plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=20, rotation=0)
    plt.savefig(os.path.join(path, "Acc_Analysis.jpg"))
    plt.close()

def batch_pred(model, x, batch_size=256):
    preds = []
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        preds.append(model(x[start:end]))
    return tf.concat(preds, 0)

def fine_tuning(model_init, model_weight, data, removed_y_train_random, removedData, validRemovedClass, validRemainClass, epochs = 1, clean_acc=1.0, repaired_filepath='finetune_model.hdf5', train_kwargs=None, log_dir=None, log_path=None, unlearn_kwargs=None, valid_orgc_near_attc = None, valid_attc_set=None):
    clear_session()
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = data
    (remove_x,remove_y) =  removedData
    (model_init_f, model_init_p) = model_init
    model = model_init_f(sgd=True, lr_init=0.01, **model_init_p)
    model.load_weights(model_weight)
    with measure_time() as t:
        model.fit(remove_x, removed_y_train_random, validation_data=(x_test, y_test), verbose=1, epochs = epochs, batch_size=64).history
        duration_s = t()
    new_theta = model.get_weights()
    model.load_weights(model_weight)

    (model_init_f, model_init_p) = model_init
    new_model = model_init_f(**model_init_p)
    new_model.set_weights(new_theta)
    if repaired_filepath is not None:
        new_model.save_weights(repaired_filepath)
    
    (val_remove_x,val_remove_y) =  validRemovedClass
    (val_remain_x,val_remain_y) =  validRemainClass
    
    acc_before, acc_after, _ = evaluate_model_diff('finetune', model, new_model, x_valid, y_valid, remove_x,remove_y, val_remove_x,val_remove_y, val_remain_x,val_remain_y, log_dir=log_dir, log_path=log_path,valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set)
    return acc_before, acc_after, duration_s


def train_retrain(model, train, val, test, model_folder, data, removedData, validRemovedClass, validRemainClass, dataset, epochs=150, batch_size=64, data_augmentation=False, log_path='models/log', valid_orgc_near_attc = None, valid_attc_set=None):
    model_save_path = os.path.join(model_folder, 'best_model_retrain.hdf5')
    csv_save_path = os.path.join(model_folder, 'train_log_retrain.csv')
    json_report_path = os.path.join(model_folder, 'certified_unlearn_result.json')
    metric_for_min = 'loss'
    # metric_for_min = 'val_loss'
    model_checkpoint_loss = ModelCheckpoint(model_save_path, monitor=metric_for_min, save_best_only=True,
                                            save_weights_only=True)
    #csv_logger = CSVLogger(csv_save_path)
    #callbacks = [model_checkpoint_loss, csv_logger]
    callbacks = [model_checkpoint_loss]
    # if x is sparse train test is a generator
    (x_train, y_train) = train
    (x_val, y_val) = val
    (x_test, y_test) = test
    hist = model.fit(x_train, y_train,batch_size=batch_size, epochs=100, validation_data=test, verbose=1, callbacks=callbacks).history
    # else its a numpy array
    
    best_loss = np.min(hist[metric_for_min]) if metric_for_min in hist else np.inf
    best_loss_epoch = np.argmin(hist[metric_for_min]) + 1 if metric_for_min in hist else 0
    print('Best model has test loss {} after {} epochs'.format(best_loss, best_loss_epoch))
    if dataset == 'cifar100':
        best_model = get_VGG(output=100)
    elif dataset == 'cifar10':
        best_model = get_VGG()
    elif dataset == 'stl10':
        best_model = get_VGG(input_shape=(96, 96, 3))

    best_model.load_weights(model_save_path)
    # in sparse case x test and train are generators
    y_test_hat = np.argmax(best_model.predict(x_test), axis=1)
    test_loss = best_model.evaluate(x_test, y_test, batch_size=1000, verbose=0)[0]
    val_loss = best_model.evaluate(x_val, y_val, batch_size=1000, verbose=0)[0]
    report = classification_report(np.argmax(y_test, axis=1), y_test_hat, digits=4, output_dict=True)
    report['train_loss'] = best_loss
    report['test_loss'] = test_loss
    report['val_loss'] = val_loss
    report['epochs_for_min'] = int(best_loss_epoch)  # json does not like numpy ints
    json.dump(report, open(json_report_path, 'w'), indent=4)
    acc_unlearn = best_model.evaluate(x_val, y_val, verbose=0)[1]
    report['val_acc'] = acc_unlearn
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = data
    (remove_x,remove_y) =  removedData
    (val_remove_x,val_remove_y) =  validRemovedClass
    (val_remain_x,val_remain_y) =  validRemainClass
    acc_before, acc_after, _ = evaluate_model_diff('retrain',best_model, best_model, x_valid, y_valid, remove_x,remove_y, val_remove_x,val_remove_y, val_remain_x,val_remain_y, log_path=log_path, valid_orgc_near_attc = valid_orgc_near_attc, valid_attc_set=valid_attc_set)
    return best_model, best_loss, acc_unlearn