# Vulnerabilities in Machine Unlearning Services
This is the source code for our NDSS'24 paper "A Duty to Forget, a Right to be Assured? Exposing Vulnerabilities in Machine Unlearning Services". The code is partly based on MachineUnlearning code https://github.com/alewarne/MachineUnlearning, and zoo attack code https://github.com/huanzhang12/ZOO-Attack

Our algorithm provide a comprehensive vulnerabilities evaluation on Machine Unlearning services.

## Requirements
Please view ```environment.yml```

Python==3.7

torch==1.13.1

torchvision==0.14.1

tqdm==4.65.0

numpy==1.21.6

## Dataset
- Cifar 10 
- Cifar 100
- STL 10
All three datasets should be processed into 6 np files: X_test.npy, X_train.npy, X_valid.npy, y_test.npy, y_train.npy, y_valid.npy. You can download these processed datasets from: https://drive.google.com/drive/folders/1OuvHRK804mE4WlWStm03jBFagbfGHatM?usp=sharing, please place the dataset in folder train_test_data.



## Train base model before unlearning
If you want to train the base model for unlearning by yourself, run 
```
python train.py --dataset ['cifar10', 'cifar100' or 'stl10'] --network ['vgg16', 'resnet']  --model_folder [FOLDER_TO_KEEP_MODEL_CONFIG_AND_WEIGHT] 
```

Here is the example for training base vgg 16 model on dataset cifar 10:

```
python train.py --dataset cifar10 --network vgg16 --model_folder VGG16
```
The training config is in VGG16/train_config.json, by modifying the file.


After training, a training log ```train_log.csv``` will be generated under model_folder, you could choose the model use for unlearning based on val_loss, and rename your chosen model as best_model.hdf5.

## Machine Unlearn for normal samples
Using ```unlearn.py``` to unlearn data samples.

### Basic arguments for Normal Unlearn:

```--model_folder```: base directory to save models and results in

```--config_file```: config file with parameters for this experiment

```--unlearn_class```: the class would like to be unlearn

```--retrain```: enable model retrain

```--dataset```: dataset name, default with cifar10

```--network```: network structure, default with vgg16

```--log_path```: path for save log

```--random_seed```: sample random choice seed

### Unlearning Config
A config file is required in your model_folder, here is an example for vgg16:
```
// If you would like to do experiments with several tau, you can add values in tau_list
{
    "steps": 1,
    "tau": 2e-4,
    "tau_list": [1e-3, 2e-3,3e-3,4e-3, 5e-3,6e-3,7e-3, 8e-3,9e-3,10e-2],
    "ratio": [0.1,0.25,0.5], 
    "first-order": {
        "tau": 2e-4,
        "tau_list": [1e-3, 2e-3,3e-3,4e-3, 5e-3,6e-3,7e-3, 8e-3,9e-3,10e-2],
        "ratio": [0.1,0.25,0.5]
    },
    "fine-tuning": {
        "epochs": [1,]
    }
}
```
### Unlearning example
Here is the example to unlearn normal data sample on vgg16 for cifar 10:
```
python unlearn.py --model_folder VGG16 --dataset cifar10 --network vgg16  --unlearn_class airplane --log_path VGG16/log
```
The solutions will be generated in the model_folder.

## Generating malicious samples
If you want to generated attack sample target from one class to another, and attack across the decision boundary, run 
```
python generate_attack.py -d ['cifar10', 'cifar100' or 'stl10'] --network ['vgg16', 'resnet']   -s [PATH_TO_SAVE_SAMPLE] -oc [ORIGINAL_CLASS] -tc [TARGET_CLASS] -t --model_weight_path [MODEL_WEIGHTS] 

//e.g.
python generate_attack.py -d cifar10 --network vgg16 -s VGG16/airplane_to_cat_cross_edge -oc airplane -tc cat -t --model_weight_path VGG16/best_model.hdf5 
```

If you want to generated attack sample target from one class to another, but do not attack across the decision boundary, run 
```
python generate_attack.py -d ['cifar10', 'cifar100' or 'stl10'] --network ['vgg16', 'resnet']   -s [PATH_TO_SAVE_SAMPLE] -oc [ORIGINAL_CLASS] -tc [TARGET_CLASS] -t --model_weight_path [MODEL_WEIGHTS] --just_within_edge

//e.g.
python generate_attack.py -d cifar10 --network vgg16 -s VGG16/airplane_to_cat_in_edge -oc airplane -tc cat -t --model_weight_path VGG16/best_model.hdf5 --just_within_edge
```

If you want to generated attack sample target from one class untargeted, and attack across the decision boundary, run 
```
python generate_attack.py -d ['cifar10', 'cifar100' or 'stl10'] --network ['vgg16', 'resnet']   -s [PATH_TO_SAVE_SAMPLE] -oc [ORIGINAL_CLASS] -t --model_weight_path [MODEL_WEIGHTS] --untargeted

//e.g.
python generate_attack.py -d cifar10 --network vgg16 -s VGG16/airplane_cross_edge -oc airplane -t --model_weight_path VGG16/best_model.hdf5 --untargeted
```


## Machine Unlearn for malicious samples
Also using unlearn.py to unlearn, but the folder with attacked samples are needed.
#### Extend arguments for unlearning with malicious samples
```--orgclass```: the original class of the attack sample
    
```--targclass```: the class we want to impact by attack

```--unlearn_attack_with_path```: unlearn dataset path
### Unlearning example for malicious samples
Using the same unlearn config as normal samples, feeding malicious sample during unlearning by run the following example code:

```
python unlearn.py --model_folder VGG16 --dataset cifar10 --network vgg16 --orgclass airplane --targclass cat --unlearn_class airplane  --log_path VGG16/airplane_to_cat_cross_edge_log --unlearn_attack_with_path VGG16/airplane_to_cat_cross_edge
```
The solutions for both with attack and without attack will be generated in the model_folder.
