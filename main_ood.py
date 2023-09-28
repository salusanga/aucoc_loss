import sklearn, argparse, os
from easydict import EasyDict
import wandb
from utils.temperature_scaling import set_temperature_scaling
from utils.read_config_file import read_config_file
from data.get_dataloaders_cifar import get_dataloaders_cifar
from data.get_dataloaders_svhn import get_dataloaders_svhn
from data.get_dataloaders_cifar_c import get_dataloaders_cifar_c
from utils.set_seed import set_seed

from models.wide_resnet import wide_resnet_cifar
from models.resnet import resnet50, resnet18


import warnings
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def evaluate_ood(settings, test_loader_id, test_loader_ood, checkpoint_file):
    """
    Evaluate the performance of a neural network model on in-distribution (ID) and
    out-of-distribution (OOD) data using multiple confidence scoring methods.

    Parameters:
        settings (object): An object containing various settings and configurations.
        test_loader_id (DataLoader): DataLoader for in-distribution test data.
        test_loader_ood (DataLoader): DataLoader for out-of-distribution test data.
        checkpoint_file (str): Path to the pre-trained model checkpoint file.

    Returns:
        None

    This function evaluates the given neural network model on both in-distribution
    and out-of-distribution datasets. It calculates Receiver Operating Characteristic
    Area Under the Curve (ROC AUC) scores for three confidence scoring methods:
    Maximum Probability Score (MPS), Maximum Logit (MaxLogit), and Energy-Based Model (EBM).
    Additionally, it computes ROC AUC scores for ODIN (Out-of-Distribution Detector
    with Temperature Scaling) variant.

    The results are logged to wandb with specific names for each evaluation method
    and data type. The function also prints out the calculated AUC scores for each
    method and data type.

    Note:
    - The provided `settings` object should contain device information, model architecture,
      and other configurations necessary for evaluation.
    - The `checkpoint_file` should point to a pre-trained model checkpoint that can be
      loaded using PyTorch.

    """
    print(" ---> Starting the test.")
    net = setup_network(settings)
    net = nn.DataParallel(net)
    checkpoint_dict = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint_dict["net_state_dict"])
    net.to(settings.device)
    net.eval()
    labels_np = np.zeros(len(test_loader_id.dataset) + len(test_loader_ood.dataset))
    logits_np = np.zeros(
        (
            (len(test_loader_id.dataset) + len(test_loader_ood.dataset)),
            settings.num_classes,
        )
    )
    labels_np[: len(test_loader_id.dataset)] = 1
    confidences_np = np.zeros(
        len(test_loader_id.dataset) + len(test_loader_ood.dataset)
    )
    confidences_ML_np = np.zeros(
        len(test_loader_id.dataset) + len(test_loader_ood.dataset)
    )
    confidences_EBM_np = np.zeros(
        len(test_loader_id.dataset) + len(test_loader_ood.dataset)
    )

    # Run the test in-ditribution
    with torch.no_grad():
        print("--> Running in distribution")
        for batch_idx, test_data in enumerate(test_loader_id, 0):
            data, test_targets = test_data
            data, test_targets = data.to(settings.device), test_targets.to(
                settings.device
            )
            test_outputs = net(data)
            # MPS
            confidences = F.softmax(test_outputs, dim=1).detach().cpu().numpy()
            confidences_max = np.max(confidences, axis=1)
            # To compute MaxLogit
            confidences_ML = np.max(test_outputs.detach().cpu().numpy(), axis=1)
            # To compute EBM
            confidences_EBM = np.log(
                np.sum(
                    np.exp(test_outputs.detach().cpu().numpy()),
                    axis=1,
                )
            )

            samples_batch = test_targets.size(0)
            offset = batch_idx * test_loader_id.batch_size
            logits_np[offset : offset + samples_batch, :] = (
                test_outputs.detach().cpu().numpy()
            )
            confidences_np[offset : offset + samples_batch] = confidences_max
            confidences_ML_np[offset : offset + samples_batch] = confidences_ML
            confidences_EBM_np[offset : offset + samples_batch] = confidences_EBM

        # Run the test in-ditribution
        print("--> Running out of distribution")
        for batch_idx, test_data in enumerate(test_loader_ood, 0):
            data, test_targets = test_data
            data, test_targets = data.to(settings.device), test_targets.to(
                settings.device
            )
            test_outputs = net(data)
            # To compute MPS
            confidences = F.softmax(test_outputs, dim=1).detach().cpu().numpy()
            confidences_max = np.max(confidences, axis=1)
            # To compute MaxLogit
            confidences_ML = np.max(test_outputs.detach().cpu().numpy(), axis=1)
            # To compute EBM
            confidences_EBM = np.log(
                np.sum(
                    np.exp(test_outputs.detach().cpu().numpy()),
                    axis=1,
                )
            )
            samples_batch = test_targets.size(0)
            offset = batch_idx * test_loader_ood.batch_size + len(
                test_loader_id.dataset
            )
            logits_np[offset : offset + samples_batch, :] = (
                test_outputs.detach().cpu().numpy()
            )
            confidences_np[offset : offset + samples_batch] = confidences_max
            confidences_ML_np[offset : offset + samples_batch] = confidences_ML
            confidences_EBM_np[offset : offset + samples_batch] = confidences_EBM

    # Rescale all logits if test for ODIN (MPS with TS)
    confidences_np_all_scaled = softmax(logits_np / settings.temperature, axis=1)
    confidences_np_all_scaled = np.max(confidences_np_all_scaled, axis=1)

    test_MPS = sklearn.metrics.roc_auc_score(labels_np, confidences_np) * 100.0
    test_ML = sklearn.metrics.roc_auc_score(labels_np, confidences_ML_np) * 100.0
    test_EBM = sklearn.metrics.roc_auc_score(labels_np, confidences_EBM_np) * 100.0

    test_ODIN = (
        sklearn.metrics.roc_auc_score(labels_np, confidences_np_all_scaled) * 100.0
    )

    wandb.run.summary["test_AUC_MPS_" + settings.model_to_test_suffix] = test_MPS
    wandb.run.summary["test_AUC_ML_" + settings.model_to_test_suffix] = test_ML
    wandb.run.summary["test_AUC_EBM_" + settings.model_to_test_suffix] = test_EBM
    wandb.run.summary["test_AUC_ODIN_" + settings.model_to_test_suffix] = test_ODIN
    print(
        "   - Test MPS {:.2f}, ML {:.2f}, EBM {:.2f}, ODIN {:.2f} for OOD.\n".format(
            test_MPS, test_ML, test_EBM, test_ODIN
        ),
    )


def setup_network(settings):
    if "cifar" in settings.dataset:
        net = wide_resnet_cifar(
            depth=settings.depth,
            width=settings.widen_factor,
            num_classes=settings.num_classes,
        )
    elif settings.net_type == "resnet50":
        net = resnet50(settings.num_classes)
    elif settings.net_type == "resnet18":
        net = resnet18(settings.num_classes)
    else:
        warnings.warn("Model is not listed.")
    net.to(settings.device)
    return net


if __name__ == "__main__":
    """
    Entry point for running model OOD evaluation on specified datasets and perturbations.

    This script performs OOD evaluation for different settings on multiple seeds, for CIFAR100-C
    with a specific corruption type, CIFAR100-C on all provided corruptions and SVHN.
    It loads configuration settings, sets up the environment, and runs evaluations using
    Weights & Biases (WandB) for logging and tracking results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)
    # Set parameters
    parser = argparse.ArgumentParser(description="Run train and/or test.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="aucoc-loss",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "--paths_config_file",
        default="paths",
        type=str,
        help="Settings for paths.",
    )
    parser.add_argument(
        "--base_config_file", type=str, help="Settings for dataset and model."
    )
    parser.add_argument(
        "--loss_config_file",
        type=str,
        help="Settings for the loss function to be used.",
    )
    parser.add_argument(
        "--cifar_all",
        default=0,
        type=int,
        help="Flag to select all perturbations for CIFAR100-C.",
    )
    parser.add_argument(
        "--corruptions_list",
        type=list,
        default=[""],
        help="List of corruptions.",
    )
    parser.add_argument(
        "--model_to_test_suffix",
        type=str,
        default="best_auc",
        help="Suffix of the best model to test, either 'best_auc' or 'best_acc' or 'last_epoch'.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=bool,
        default=True,
        help="Set cudnn benchmark on (1) or off (0) (default is on).",
    )

    settings = vars(parser.parse_args())
    settings = read_config_file("configs", settings["paths_config_file"], settings)
    settings = read_config_file(
        settings["base_config_path"], settings["base_config_file"], settings
    )
    settings = read_config_file(
        settings["loss_config_path"], settings["loss_config_file"], settings
    )
    settings = EasyDict(settings)
    # Setup other parameters: device, directory for checkpoints and plots of this model
    settings.device = device
    settings.checkpoint_dir = os.path.join(
        settings.checkpoints_path,
        settings.project_name,
        settings.dataset,
        settings.net_type,
        str(settings.batch_size),
        settings.loss_type,
        settings.model_name,
    )
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_START_METHOD"] = "thread"
    seeds = np.arange(0, 3)

    # Setup of corruptions for CIFAR100-C
    if settings.cifar_all == 1:
        settings.corruptions_list = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "zoom_blur",
            "brightness",
            "fog",
            "frost",
            "glass_blur",
            "snow",
            "motion_blur",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
        ]
    elif settings.cifar_all == 0 and settings.dataset_ood == "cifar100_c":
        settings.corruptions_list = [settings.corruption_type]
    # Run eval for multiple seeds
    for seed in seeds:
        set_seed(seed)
        settings.seed = seed
        # Set wandb project name to gather results
        if settings.cifar_all == 1:
            project_name_wandb = "aucocloss-OOD-{}-{}-{}-all-perturbations".format(
                settings.dataset,
                settings.dataset_ood,
                settings.net_type,
            )
        else:
            if settings.dataset_ood == "cifar100_c":
                project_name_wandb = (
                    "aucocloss-OOD-{}-{}-{}-{}".format(  # MSP nulla, MaxLogit
                        settings.dataset,
                        settings.dataset_ood,
                        settings.net_type,
                        settings.corruption_type,
                    )
                )
            else:
                project_name_wandb = (
                    "aucocloss-OOD-{}-{}-{}".format(  # MSP nulla, MaxLogit
                        settings.dataset,
                        settings.dataset_ood,
                        settings.net_type,
                    )
                )
        # Iterate over corruptions, for SVHN the list is empty
        for settings.corruption_type in settings.corruptions_list:
            with wandb.init(
                project=project_name_wandb,
                config=settings,
                dir=settings.dir_wandb,
            ):
                # Setup ID and OOD datasets
                _, val_loader_id, test_loader_id = get_dataloaders_cifar(settings)
                if settings.dataset_ood == "SVHN":
                    test_loader_ood = get_dataloaders_svhn(settings)
                elif settings.dataset_ood == "cifar100_c":
                    test_loader_ood = get_dataloaders_cifar_c(settings)
                print(
                    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Run number {:2d}, OOD dataset: {}, perturbation: {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(
                        settings.seed, settings.dataset_ood, settings.corruption_type
                    )
                )
                wandb.run.name = settings.model_name + "/seed-{:2d}-{}".format(
                    settings.seed, settings.corruption_type
                )
                checkpoint_file = "{}/{}_{:02d}_{}.pth".format(
                    settings.checkpoint_dir,
                    settings.model_name,
                    settings.seed,
                    settings.model_to_test_suffix,
                )
                # Temperature needed for ODIN
                set_temperature_scaling(val_loader_id, checkpoint_file, settings)
                # OOD evaluation
                evaluate_ood(settings, test_loader_id, test_loader_ood, checkpoint_file)
