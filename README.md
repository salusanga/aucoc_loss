 ---

<div align="center">    
 
# Expert load matters: operating networks at high accuracy and low manual effort

[![Static Badge](https://img.shields.io/badge/paper-arXiv-darkred)
](https://arxiv.org/abs/2308.05035)
[![Static Badge](https://img.shields.io/badge/NeurIPS-2023-blue)](https://neurips.cc/Conferences/2023)

</div>

Official code for the NeurIPS 2023 paper [Expert load matters: operating networks at high accuracy and low manual effort](https://arxiv.org/abs/2308.05035). 


## Abstract

In human-AI collaboration systems for critical applications, in order to ensure minimal error, users should set an operating point based on model confidence to determine when the decision should be delegated to human experts. Samples for which model confidence is lower than the operating point would be manually analysed by experts to avoid mistakes. Such systems can become truly useful only if they consider two aspects: models should be confident only for samples for which they are accurate, and the number of samples delegated to experts should be minimized. The latter aspect is especially crucial for applications where available expert time is limited and expensive, such as healthcare. The trade-off between the model accuracy and the number of samples delegated to experts can be represented by a curve that is similar to an ROC curve, which we refer to as confidence operating characteristic (COC) curve. In this paper, we argue that deep neural networks should be trained by taking into account both accuracy and expert load and, to that end, propose a new complementary loss function for classification that maximizes the area under this COC curve. This promotes simultaneously the increase in network accuracy and the reduction in number of samples delegated to humans. We perform experiments on multiple computer vision and medical image datasets for classification. Our results demonstrate that the proposed loss improves classification accuracy and delegates less number of decisions to experts, achieves better out-of-distribution samples detection and on par calibration performance compared to existing loss functions.


## Overview
1. [Preparation](#Preparation)
    1. [Requirements](#Requirements)
    2. [Setup "Weights and Biases"](#wandb)
    3. [Set paths](#paths)
    4. [Download Data](#data)
2. [Code organisation](#code)
3. [Training and evaluation for in-distribution experiment](#traineval)
    1. [Launch the training + evaluation](#launchtraineval)
    2. [Run only evaluation on test set](#eval)
4. [Evaluation for OOD experiments](#ood)
5. [Contact](#contact)
6. [How to cite](#6-how-to-cite)



## 1. Preparation<a name="Preparation"></a>

### 1.1. Requirements<a name="Requirements"></a>
The code is run with Python 3.8 and PyTorch 1.10.2. To install the packages, use:
```bash
conda create --name aucocloss --file requirements.txt
```
To activate the environment run:
```bash
conda activate aucocloss
```

### 1.2. Setup "Weights and Biases" for results visualisation and metrics tracking<a name="wandb"></a>

The code uses [weights and biases (wandb)](https://wandb.ai/site) visualisation tool to load the final test results and the training/validation metrics actively during the training. [Here](https://docs.wandb.ai/quickstart) it is explained how to get started with it. In order to average the results for every model over 3 seeds as in the paper it will be just necessary to group the runs by "model".


### 1.3. Set paths<a name="paths"></a>

In the file `configs/paths.yaml` insert your local paths to datasets, checkpoints and wandb directories. 


### 1.4. Download data<a name="data"></a>

Before running the code, download and extract the respective datasets to the directory previously specified by the entry `datasets_path` in `configs/paths.yaml`.

<details>
  <summary>CIFAR100</summary>

  Dowload CIFAR100 from [here](https://www.cs.toronto.edu/~kriz/cifar.html).
</details>

<details>
  <summary>Tiny-Imagenet</summary>

  Download Tiny-Imagenet by running within your dataset location:
  ```bash
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  ```
And then run [this script](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4) for properly unzipping it.
</details>

<details>
  <summary>MedMNIST</summary>

  Dowload [MedMNIST (V2)](https://medmnist.com/) from [here](https://zenodo.org/record/6496656).
</details>

## 2. Code organisation<a name="code"></a>

The base directory--which can be different from the code directory--will contain the following sub-directories:

    .
    ├── configs                 # Contains subfolders for configuration
    │   ├── base_config         # Contains "base" config files, i.e. which dataset-architecture combination
    │   ├── loss_config         # Contains "loss" config files, i.e. which loss function and its hyperparam.
    │   ├── paths.yaml          # File to indicate the local paths
    ├── data                    # Contains all the dataloaders for the datsaets
    ├── loss_functions          # Base folder for all the losses
    │   ├── baselines           # Contains the loss implementation for the baselines used
    │   ├── auc_loss_bw.py      # Implementation of the proposed AUCOCLoss
    │   ├── compute_loss.py     # Code for loss selection
    ├── metrics                 # Evaluation metrics
    ├── models                  # Backbones
    ├── src                     # Source files for training and evaluation
    ├── utils                   # Helpers, e.g. to handle checkpoints, set temp. scaling
    ├── main.py                 # To run train/eval for main experiments
    ├── main_ood.py             # To run OOD eval
    ├── README.md              
    └── requirements.txt        # To setup the conda environment

## 3. Training and evaluation for in-distribution experiments<a name="traineval"></a>
### 3.1. Launch the training + evaluation<a name="launchtraineval"></a>
```bash
conda activate aucocloss
python -u main.py --base_config_file $1 --loss_config_file $2
```

Where `base_config_file` is one of the files from `configs/base_config/` with the setup used in the paper. `loss_config_file` is one from `configs/loss_config/`. The names of the files indicate already respectively which dataset-architecture combination and which loss function/which hyperparam. selection is being employed.

E.g. to run training and evaluation on CIFAR100 with Wide-ResNet-10-28 (all the other hyperparam already set as in the main paper), with AUCOCLoss with cross entropy as primary loss and weighting factor lamda=1:

```bash
conda activate aucocloss
python -u main.py --base_config_file cifar100_wide_resnet_28 --loss_config_file auc_secondary_bw_CE_l1
```
base_config_file with "LT", e.g. cifar100_LT_002, specify the long-tailed experiments.

### 3.2. Run only evaluation on test set<a name="eval"></a>
```bash
conda activate aucocloss
python -u main.py --base_config_file $1 --loss_config_file $2 --train_mode 0
```

Where `base_config_file` and `loss_config_file` are defined and used as before. `loss_config_file` is required to retrieve information about which model to evaluate.

## 4. Evaluation for OOD experiments<a name="ood"></a>

```bash
conda activate aucocloss
python -u main_ood.py --base_config_file $1 --loss_config_file $2
```
Where `base_config_file` must be one among:

- ood_cifar100_cifar100_c_all_pert: ID is CIFAR100, OOD is CIFAR100-C on all perturbations
- ood_cifar100_cifar100_c_gaussian: ID is CIFAR100, OOD is CIFAR100-C with gaussian noise
- ood_cifar100_svhn: ID is CIFAR100, OOD is SVHN

`loss_config_file` is required to retrieve information about which model to evaluate.

## 5. Contact<a name="contact"></a>

For any questions, suggestions, or issues with the code, please contact Sara at: sara.sangallii@vision.ee.ethz.ch.


## 6. How to cite<a name="6-how-to-cite"></a>

If you find this code useful in your research, please consider citing the paper:

```bibtex
@inproceedings{sangalli2023aucocloss,
  title={Expert load matters: operating networks at high accuracy and low manual effort},
  author={Sangalli, Sara and Erdil, Ertunc and Konukoglu, Ender},
  booktitle={Neural Information Processing Systems},
  year={2023}
}
```

