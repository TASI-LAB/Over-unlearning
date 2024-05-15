import random
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import os
from os.path import dirname as parent
import copy

from util import LoggedGradientTape,UnlearningResult, reduce_dataset, measure_time
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.backend import clear_session

import json

@tf.function
def hvp(model, x, y, v,trainable_linear = -6):
    """ Hessian vector product. """
    # 1st gradient of Loss w.r.t weights
    with LoggedGradientTape() as tape:
        # first gradient
        grad_L = get_gradients(model, x, y, trainable_linear=trainable_linear)
        assert len(v) == len(grad_L)
        # v^T * \nabla L
        v_dot_L = [v_i * grad_i for v_i, grad_i in zip(v, grad_L)]
        # tape.watch(self.model.weights)
        # second gradient computation
        hvp = tape.gradient(v_dot_L, model.trainable_weights[trainable_linear:])
    # for embedding layers, gradient can be of type indexed slices and need to be converted
    for i in range(len(hvp)):
        if type(hvp[i]) == tf.IndexedSlices:
            hvp[i] = tf.convert_to_tensor(hvp[i])
    return hvp


def get_gradients(model, x_tensor, y_tensor, batch_size=256, trainable_linear = -6):
    """ Calculate dL/dW (x, y) """
    grads = []
    for start in range(0, x_tensor.shape[0], batch_size):
        with LoggedGradientTape() as tape:
            tape.watch(model.trainable_weights[trainable_linear:])
            result = model(x_tensor[start:start+batch_size])
            loss = model.loss(y_tensor[start:start+batch_size], result)
            grads.append(tape.gradient(loss, model.trainable_weights[trainable_linear:]))
    grads = list(zip(*grads))
    for i in range(len(grads)):
        grads[i] = tf.add_n(grads[i])
    # for embedding layers, gradient can be of type indexed slices and need to be converted
    for i in range(len(grads)):
        if type(grads[i]) == tf.IndexedSlices:
            grads[i] = tf.convert_to_tensor(grads[i])
    return grads


@tf.function
def get_gradients_diff(model, x_delta_tensor, y_delta_tensor, removed_y_train_random, unlearn_type=0, batch_size=256, trainable_linear = -6):
    """
    x0,x1,x2,x3 y0,y1,y2,y3 
    y2 from true -> false
    Compute d/dW [ Loss(x_delta, y_delta) - Loss(x,y) ]
    This saves one gradient call compared to calling `get_gradients` twice.
    """

    # type 0 black
    # type 1 white
    # type 2 random label
    # type 3 random pixcel
    grads = []

    (w,h,c) = x_delta_tensor[0].shape
    if unlearn_type == 1:
        random_img = np.array([[[255 for j in range(c)] for i in range(w)] for k in range(w)])
        x = tf.convert_to_tensor(np.array([random_img for i in range(batch_size)]), dtype=tf.float32)
        
    elif unlearn_type == 0:
        random_img = np.array([[[0 for j in range(c)] for i in range(w)] for k in range(w)])
        x = tf.convert_to_tensor(np.array([random_img for i in range(batch_size)]), dtype=tf.float32)
        
    elif unlearn_type == 3:

        random_img = np.array([[[random.randint(0, 255) for j in range(c)] for i in range(w)] for k in range(w)])
        x = tf.convert_to_tensor(np.array([random_img for i in range(batch_size)]), dtype=tf.float32)
    
    if unlearn_type != 2 and unlearn_type != 4:
        x = ((x/255)-.5)
        result_x = model(x)
        
    for start in range(0, x_delta_tensor.shape[0], batch_size):
        with LoggedGradientTape() as tape:
            shape = len(x_delta_tensor[start:start + batch_size])
            tape.watch(model.trainable_weights[trainable_linear:])
            result_x_delta = model(x_delta_tensor[start:start + batch_size])
            loss_x_delta = model.loss(y_delta_tensor[start:start + batch_size], result_x_delta)

            if unlearn_type == 0 or unlearn_type == 1 or unlearn_type == 3:
                if shape != batch_size:
                    result_x = model(x[:shape])
                loss_x = model.loss(y_delta_tensor[start:start + batch_size], result_x)
            else:
                
                loss_x  = model.loss(removed_y_train_random[start:start + batch_size], result_x_delta)
            if unlearn_type == 4:
                diff = - loss_x_delta
            else:
                diff = loss_x - loss_x_delta

            grads.append(tape.gradient(diff, model.trainable_weights[trainable_linear:]))
    grads = list(zip(*grads))
    for i in range(len(grads)):
        grads[i] = tf.add_n(grads[i])
    # for embedding layers, gradient can be of type indexed slices and need to be converted
    for i in range(len(grads)):
        if type(grads[i]) == tf.IndexedSlices:
            grads[i] = tf.convert_to_tensor(grads[i])
    return grads


def get_inv_hvp_lissa(model, x, y, v, hvp_batch_size, scale, damping, iterations=-1, verbose=False,
                    repititions=1, early_stopping=True, patience=20, hvp_logger=None, trainable_linear = -6):
    """
    Calculate H^-1*v using the iterative scheme proposed by Agarwal et al with batch updates.
    The scale and damping parameters have to be found by trial and error to achieve convergence.
    Rounds can be set to average the results over multiple runs to decrease variance and stabalize the results.
    """
    i = tf.constant(0)
    n_batches = 100 * np.ceil(x.shape[0] / hvp_batch_size) if iterations == -1 else iterations
    shuffle_indices = [tf.constant(range(x.shape[0]), dtype=tf.int32) for _ in range(repititions)]
    def cond(i, u, shuff_idx, update_min): return tf.less(i, n_batches) and tf.math.is_finite(tf.norm(u[0]))

    def body(i, u, shuff_idx, update_min):
        i_mod = ((i * hvp_batch_size) % x.shape[0]) // hvp_batch_size
        start, end = i_mod * hvp_batch_size, (i_mod+1) * hvp_batch_size
        if sp.issparse(x):
            batch_hvps = hvp(model, tf.gather(x, shuff_idx)[start:end].toarray(),
                             tf.gather(y, shuff_idx)[start:end], u, trainable_linear=trainable_linear)
        else:
            batch_hvps = hvp(model, tf.gather(x, shuff_idx)[start:end],
                            tf.gather(y, shuff_idx)[start:end], u,trainable_linear=trainable_linear)
        new_estimate = [a + (1-damping) * b - c/scale for (a, b, c) in zip(v, u, batch_hvps)]
        update_norm = np.sum(np.sum(np.abs(old - new)) for old, new in zip(u, new_estimate))
        if early_stopping and update_norm > update_min[0] and update_min[-1] >= patience:
            tf.print(f"Early stopping at iteration {i+1}. Update norm {update_norm} > {update_min}")
            if i < patience:
                i = n_batches + 1
            else:
                i = n_batches
        if update_norm < update_min[0]:
            update_min = [update_norm, 1]
        if verbose:
            tf.print(i, update_norm)  # [tf.norm(ne) for ne in new_estimate][:5])
        if hvp_logger is not None:
            if isinstance(i, tf.Tensor):
                hvp_logger.log(step=hvp_logger.step, inner_step=hvp_logger.inner_step, i=i.numpy(), update_norm=update_norm)
            else:
                hvp_logger.log(step=hvp_logger.step, inner_step=hvp_logger.inner_step, i=i, update_norm=update_norm)
        if i+1 == n_batches:
            tf.print(f"No convergence after {i+1} iterations. Stopping.")
        update_min[-1] += 1
        return i+1, new_estimate, shuff_idx, update_min

    estimate = None
    for r in range(repititions):
        loop_vars = (i, v, shuffle_indices[r], [np.inf, -1])
        res = tf.while_loop(cond, body, loop_vars)
        # i encodes the exit reason of the body:
        #   i == n_batches:     maximum number of iterations reached
        #   i == n_batches+1:   early stopping criterium reached
        #   i == n_batches+2:   early stopping after first iterations (diverged)
        if res[0] == n_batches+2:
            return res[1], True
        # if one iteration failed averaging makes no sense anymore
        if not all([tf.math.is_finite(tf.norm(e)) for e in res[1]]):
            return res[1], True
        res_upscaled = [r/scale for r in res[1]]
        if estimate is None:
            estimate = [r/repititions for r in res_upscaled]
        else:
            for j in range(len(estimate)):
                estimate[j] += res_upscaled[j] / repititions
    diverged = not all([tf.math.is_finite(tf.norm(e)) for e in estimate])
    return estimate, diverged


def approx_retraining(model, remove_x, remove_y, removed_y_train_random, order=2, 
                      hvp_x=None, hvp_y=None, tau=None, hvp_logger=None,
                      conjugate_gradients=False, verbose=False, 
                      unlearn_type=0, **unlearn_kwargs):
    """ Perform parameter update using influence functions. """
    #tau = 2e-5
    if order == 2:
        tau = 1
    trainable_linear = None
    if 'trainable_linear' in unlearn_kwargs:
        trainable_linear = unlearn_kwargs['trainable_linear']
    else:
        trainable_linear = -6
    # d_theta = 0
    if order == 1:
            # first order update
            diff = get_gradients_diff(model, remove_x, remove_y, removed_y_train_random, unlearn_type=unlearn_type, trainable_linear = trainable_linear)
            d_theta = diff
            diverged = False
    # second order update
    elif order == 2:
        diff = get_gradients_diff(model, remove_x, remove_y ,removed_y_train_random, unlearn_type=unlearn_type, trainable_linear = trainable_linear)
        # skip hvp if diff == 0
        if np.sum(np.sum(d) for d in diff) == 0:
            d_theta = diff
            diverged = False
        elif conjugate_gradients:
            raise NotImplementedError('Conjugate Gradients is not implemented yet!')
        else:
            assert hvp_x is not None and hvp_y is not None
            d_theta, diverged = get_inv_hvp_lissa(model, hvp_x, hvp_y, diff, unlearn_kwargs['second-order']['hvp_batch_size'], unlearn_kwargs['second-order']['scale'], unlearn_kwargs['second-order']['damping'], verbose=verbose, hvp_logger=hvp_logger, trainable_linear=trainable_linear)
            
    if order != 0:
        # only update trainable weights (non-invasive workaround for BatchNorm layers in CIFAR model)
        # d_theta = [d_theta.pop(0) if w.trainable and i >= len(model.weights) -26 else tf.constant(0, dtype=tf.float32) for i, w in enumerate(model.weights)]
        # update_pos = len(model.trainable_weights) - len(d_theta)
        # theta_approx = [w - tau * d_theta.pop(0) if i >= update_pos else w for i,
        #                 w in enumerate(model.trainable_weights)]
        # theta_approx = [theta_approx.pop(0) if w.trainable else w for w in model.weights]
        # theta_approx = [w.numpy() for w in theta_approx]
        # theta_approx = [w - tau * d_t for w, d_t in zip(model.weights, d_theta)]
        
        # d_theta = [d_theta.pop(0) if w.trainable else 0 for w in model.weights]
        # theta_approx = [w - tau * d_t for w, d_t in zip(model.get_weights(), d_theta)]
        # d_theta = [d_theta.pop(0) if w.trainable and i >= len(model.weights) -26 else tf.constant(0, dtype=tf.float32) for i, w in enumerate(model.weights)]
        # update_pos = len(model.trainable_weights) - len(d_theta)
        # theta_approx = [w - tau * d_theta.pop(0) if i >= update_pos else w for i,
        #                 w in enumerate(model.trainable_weights)]
        # theta_approx = [theta_approx.pop(0) if w.trainable else w for w in model.weights]
        # theta_approx = [w.numpy() for w in theta_approx]
        
        
        
        # d_theta_trainable = copy.deepcopy(model.trainable_weights)
        
        # d_theta_trainable = d_theta_trainable[:-26] + (d_theta)
        
        # d_theta = [d_theta_trainable.pop(0) if w.trainable else 0 for w in model.weights]
        # theta_approx = [w - tau * d_t for w, d_t in zip(model.get_weights(), d_theta)]
        
        update_pos = len(model.trainable_weights) - len(d_theta)
        theta_approx = [w - tau * d_theta.pop(0) if i >= update_pos else w for i,
                        w in enumerate(model.trainable_weights)]
        theta_approx = [theta_approx.pop(0) if w.trainable else w for w in model.weights]
        theta_approx = [w.numpy() for w in theta_approx]
    return theta_approx, diverged


