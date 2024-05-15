## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import sys
import tensorflow as tf
import numpy as np
import random
import time

from attack_setup.setup_cifar import CIFAR, CIFARModel
from attack_setup.setup_cifar100 import CIFAR100, CIFAR100Model
from attack_setup.setup_stl10 import STL10, STL10Model

from l2_attack_black import BlackBoxL2

from PIL import Image


def show(img, name = "output.png", whether_save_np=True):
    """
    Show MNSIT digits in the console.
    """
    if whether_save_np:
        np.save(name, img)
    # fig = np.around((img + 0.5)*255)
    # fig = fig.astype(np.uint8).squeeze()
    # pic = Image.fromarray(fig)
    # pic.save(name)
    # remap = "  .*#"+"#"*100
    # img = (img.flatten()+.5)*3
    # if len(img) != 784: return
    # print("START")
    # for i in range(28):
    #     print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def generate_data(data, samples, targeted_one_class, tc, oc, targeted=True, start=0, inception=False, label_name=None):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    tc = label_name.index(tc) 
    oc = label_name.index(oc) 
    if targeted_one_class:
        if targeted:
            for i in range(len(data.train_data)):
                if (data.tl[i] == oc):
                    inputs.append(data.train_data[i])
                    targets.append(np.eye(data.train_labels.shape[1])[tc])
                    labels.append(data.train_labels[i])
                    true_ids.append(i)
        else:
            for i in range(len(data.train_data)):
                if (data.tl[i] == oc):
                    inputs.append(data.train_data[i])
                    targets.append(data.train_labels[i])
                    labels.append(data.train_labels[i])
                    true_ids.append(i)

    else:
        for i in range(samples):
            if targeted:
                if inception:
                    # for inception, randomly choose 10 target classes
                    seq = np.random.choice(range(1,1001), 10)
                    # seq = [580] # grand piano
                else:
                    # for CIFAR and MNIST, generate all target classes
                    seq = range(data.test_labels.shape[1])               
                for j in seq:
                    # skip the original image label
                    if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                        continue
                    inputs.append(data.test_data[start+i])
                    targets.append(np.eye(data.test_labels.shape[1])[j])
                    labels.append(data.test_labels[start+i])
                    true_ids.append(start+i)
            else:
                inputs.append(data.test_data[start+i])
                targets.append(data.test_labels[start+i])
                labels.append(data.test_labels[start+i])
                true_ids.append(start+i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)

    return inputs, targets, labels, true_ids

def main(args):
    with tf.compat.v1.Session() as sess:
        use_log = not args['use_zvalue']
        is_inception = args['dataset'] == "imagenet"
        # load networkv
        print('Loading model', args['dataset'])
        if args['dataset'] == "mnist":
            data, model =  MNIST(), MNISTModel(args['model_weight_path'], sess, use_log)
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            # data, model =  MNIST(), MNISTModel("models/mnist-distilled-100", sess, use_log)
        elif args['dataset'] == "cifar10":
            data, model = CIFAR(), CIFARModel(args['model_weight_path'], sess, use_log, network=args['network'])
            label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            # data, model = CIFAR(), CIFARModel("models/cifar-distilled-100", sess, use_log)
        elif args['dataset'] == "cifar100":
            data, model = CIFAR100(), CIFAR100Model(args['model_weight_path'], sess, use_log, network=args['network'])
            label_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 
                                'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 
                                'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 
                                'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 
                                'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
                                'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        elif args['dataset'] == "stl10":
            data, model = STL10(), STL10Model(args['model_weight_path'], sess, use_log)
            label_name = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            
        print('Done...')
        if args['numimg'] == 0:
            args['numimg'] = len(data.test_labels) - args['firstimg']
        print('Using', args['numimg'], 'test images')
        # load attack module
        attack = BlackBoxL2(sess, model, batch_size=128, max_iterations=args['maxiter'], print_every=args['print_every'], 
                    early_stop_iters=args['early_stop_iters'], confidence=0, learning_rate = args['lr'], initial_const=args['init_const'], 
                    binary_search_steps=args['binary_steps'], targeted=not args['untargeted'], use_log=use_log, use_tanh=args['use_tanh'], 
                    use_resize=args['use_resize'], adam_beta1=args['adam_beta1'], adam_beta2=args['adam_beta2'], reset_adam_after_found=args['reset_adam'],
                    solver=args['solver'], save_ckpts=args['save_ckpts'], load_checkpoint=args['load_ckpt'], start_iter=args['start_iter'],
                    init_size=args['init_size'], use_importance=not args['uniform'])

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        print('Generate data')
        all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, samples=args['numimg'], targeted_one_class=args['targeted_one_class'], tc=args['target_class'], oc=args['original_class'], targeted=not args['untargeted'],
                                        start=args['firstimg'], inception=is_inception, label_name=label_name)


        print('Done...')
        os.system("mkdir -p {}/".format(args['save']))
        img_no = 0
        total_success = 0
        l2_total = 0.0


        # end_id = args['end_no']
        # exit(0)
        for i in range(len(all_inputs)):
            inputs = all_inputs[i:i+1]
            targets = all_targets[i:i+1]
            labels = all_labels[i:i+1]
            # test if the image is correctly classified
            original_predict = model.model.predict(inputs)

            original_predict = tf.squeeze(original_predict)
            original_prob = tf.sort(original_predict)
            original_prob = original_prob.eval(session = sess)
            original_class = tf.argsort(original_predict)
            original_class = original_class.eval(session = sess)
           
            if original_class[-1] != np.argmax(labels):
                print("skip wrongly classified image no. {}, original class {}, classified as {}".format(i, np.argmax(labels), original_class[-1]))
                continue
            
            img_no += 1
            timestart = time.time()
            adv, const = attack.attack_batch(inputs, targets, args["just_within_edge"], labels)
            if type(const) is list: 
                const = const[0]
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            timeend = time.time()
            l2_distortion = np.sum((adv-inputs)**2)**.5
            adversarial_predict = model.model.predict(adv)
            # adversarial_predict.eval(session=tf.compat.v1.Session()) 
            # adversarial_predict = tf.convert_to_tensor(adversarial_predict, dtype=tf.float32)
            # adversarial_predict = np.squeeze(adversarial_predict)
            # adversarial_prob = np.sort(adversarial_predict)
            # adversarial_class = np.argsort(adversarial_predict)
            
            adversarial_predict = tf.squeeze(adversarial_predict)
            adversarial_prob = tf.sort(adversarial_predict)
            adversarial_prob = adversarial_prob.eval(session = sess)
            adversarial_class = tf.argsort(adversarial_predict)
            adversarial_class = adversarial_class.eval(session = sess)
            success = False
            if args['untargeted']:
                if adversarial_class[-1] != original_class[-1]:
                    success = True
            else:
                if adversarial_class[-1] == np.argmax(targets):
                    success = True
            if l2_distortion > 20.0:
                success = False
            if success:
                total_success += 1
                l2_total += l2_distortion
            suffix = "id{}_seq{}_prev{}_adv{}_{}_dist{}".format(all_true_ids[i], i, original_class[-1], adversarial_class[-1], success, l2_distortion)
            print("Saving to", suffix)
            # show(inputs, "{}/{}/{}_original_{}.png".format(args['save'], args['dataset'], img_no, suffix))
            # show(adv, "{}/{}/{}_adversarial_{}.png".format(args['save'], args['dataset'], img_no, suffix))
            if not args["just_within_edge"]:
                if success:
                    show(adv, "{}/{}_adversarial_{}.png".format(args['save'], img_no, suffix), whether_save_np=True)
            else:
                if not success:
                    show(adv, "{}/{}_adversarial_{}.png".format(args['save'], img_no, suffix), whether_save_np=True)
                
            # show(adv - inputs, "{}/{}/{}_diff_{}.png".format(args['save'], args['dataset'], img_no, suffix))
            print("[STATS][L1] total = {}, seq = {}, id = {}, time = {:.3f}, success = {}, const = {:.6f}, prev_class = {}, new_class = {}, distortion = {:.5f}, success_rate = {:.3f}, l2_avg = {:.5f}".format(img_no, i, all_true_ids[i], timeend - timestart, success, const, original_class[-1], adversarial_class[-1], l2_distortion, total_success / float(img_no), 0 if total_success == 0 else l2_total / total_success))
            sys.stdout.flush()

        # t = np.random.randn(28*28).reshape(1,28,28,1)
        # print(model.model.predict(t))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["cifar10", "cifar100", "stl10"], default="cifar10")
    parser.add_argument("-s", "--save", default="./saved_results")
    parser.add_argument("-n", "--numimg", type=int, default=1, help = "number of test images to attack")
    parser.add_argument("-m", "--maxiter", type=int, default=20000, help = "set 0 to use default value")
    parser.add_argument("-p", "--print_every", type=int, default=10, help = "print objs every PRINT_EVERY iterations")
    parser.add_argument("-o", "--early_stop_iters", type=int, default=100, help = "print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("-f", "--firstimg", type=int, default=0)
    parser.add_argument("-b", "--binary_steps", type=int, default=0)
    parser.add_argument("-c", "--init_const", type=float, default=10.0)
    parser.add_argument("-z", "--use_zvalue", action='store_true')
    
    
    parser.add_argument("-u", "--untargeted", action='store_true')

    parser.add_argument("-t", "--targeted_one_class", action='store_true', default=True)
    
    parser.add_argument("-oc", "--original_class", type=str,default='airplane')
    parser.add_argument("-tc", "--target_class", type=str,default='cat')
    
    parser.add_argument("-r", "--reset_adam", action='store_true',default=True, help = "reset adam after an initial solution is found")
    parser.add_argument("--use_resize", action='store_true', help = "resize image (only works on imagenet!)")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1216)
    parser.add_argument("--solver", choices=["adam", "newton", "adam_newton", "fake_zero"], default="adam")
    parser.add_argument("--save_ckpts", default="", help = "path to save checkpoint file")
    parser.add_argument("--load_ckpt", default="", help = "path to numpy checkpoint file")
    parser.add_argument("--start_iter", default=0, type=int, help = "iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--init_size", default=32, type=int, help = "starting with this size when --use_resize")
    parser.add_argument("--uniform", action='store_true', help = "disable importance sampling")
    
    parser.add_argument("--just_within_edge", action='store_true', help = "return adversarial attack result just within edge")
    parser.add_argument("--model_weight_path", type=str,default='')
    parser.add_argument("--network", type=str, default="vgg16", choices=["vgg16", "resnet18", "resnet50", "vgg19", "vgg10", "resnet"], help="model choice")
    
    parser.add_argument("--start_no", type=int, default=0, help="start id of image")
    parser.add_argument("--end_no", type=int, default=0, help="end id of image")
    args = vars(parser.parse_args())
    # add some additional parameters
    # learning rate
    args['lr'] = 1e-2
    args['inception'] = False
    args['use_tanh'] = True
    # args['use_resize'] = False
    if args['maxiter'] == 0:
        if args['dataset'] == "imagenet":
            if args['untargeted']:
                args['maxiter'] = 1500
            else:
                args['maxiter'] = 50000
        elif args['dataset'] == "mnist":
            args['maxiter'] = 3000
        else:
            args['maxiter'] = 1000
    if args['init_const'] == 0.0:
        if args['binary_steps'] != 0:
            args['init_const'] = 0.01
        else:
            args['init_const'] = 0.5
    if args['binary_steps'] == 0:
        args['binary_steps'] = 1
    # set up some parameters based on datasets
    if args['dataset'] == "imagenet":
        args['inception'] = True
        args['lr'] = 2e-3
        # args['use_resize'] = True
        # args['save_ckpts'] = True
    # for mnist, using tanh causes gradient to vanish
    if args['dataset'] == "mnist":
        args['use_tanh'] = False
    # when init_const is not specified, use a reasonable default
    if args['init_const'] == 0.0:
        if args['binary_search']:
            args['init_const'] = 0.01
        else:
            args['init_const'] = 0.5
    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    print(args)
    main(args)

