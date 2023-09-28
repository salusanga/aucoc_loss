import os
import argparse
import numpy as np
from easydict import EasyDict
import wandb
import torch
import warnings

from utils.set_seed import set_seed
from utils.read_config_file import read_config_file
from utils.temperature_scaling import set_temperature_scaling
from utils.print_final_results import print_final_results
from data.get_dataloaders_cifar import get_dataloaders_cifar
from data.get_dataloaders_tiny_imagenet import get_dataloaders_tiny_imagenet
from data.get_dataloaders_medmnist import get_dataloaders_medmnist
from data.imbalanced_CIFAR import get_dataloaders_cifar_LT
from src.trainer import Trainer
from src.eval import get_new_results


def main():
    """
    Main function for training and testing a neural network model on a specified dataset.

    This function handles the entire process, including setting up the environment,
    loading configuration, training and testing the model, and collecting and reporting
    evaluation results. It supports multiple runs with different random seeds for
    statistical analysis.

    The function takes command-line arguments for various settings and configuration
    files and is designed to be run as a standalone script.
    It uses the Weights & Biases (WandB) framework for logging and visualization.

    Note: Before running this function, ensure that all required configuration files
    and datasets are correctly set up, refer to README.md for more details.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)
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
        "--train_mode",
        type=int,
        default=1,
        help="Whether run training.",
    )
    parser.add_argument(
        "--eval_mode", type=int, default=1, help="Whether run evaluation."
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of seeds to use for training.",
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
        "--resume_training",
        type=int,
        default=0,
        help="Whether resume a pre-trained model.",
    )
    parser.add_argument(
        "--use_pretrained_model",
        type=int,
        default=0,
        help="Whether start a whole new training from pretraied model.",
    )
    parser.add_argument(
        "--model_to_test_suffix",
        type=str,
        default="best_auc",
        help="Suffix of the model to test, either 'best_auc' or 'best_acc' or 'last_epoch'.",
    )
    parser.add_argument(
        "--best_val_auc_runs",
        type=list,
        default=[],
        help="List to track val AUC over runs.",
    )
    parser.add_argument(
        "--best_val_acc_runs",
        type=list,
        default=[],
        help="List to track val accuracy over runs.",
    )
    parser.add_argument(
        "--best_val_ece_runs",
        type=list,
        default=[],
        help="List to track val ece over runs.",
    )
    parser.add_argument(
        "--use_temperature_scaling",
        type=int,
        default=0,
        help="Whether to find the optimal temperature for temperature scaling.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=bool,
        default=True,
        help="Set cudnn benchmark on (1) or off (0) (default is on).",
    )
    parser.add_argument(
        "--use_scheduler",
        type=int,
        default=1,
        help="Whether to use a scheduler to train the network.",
    )

    settings = vars(parser.parse_args())
    settings = read_config_file("configs", settings["paths_config_file"], settings)
    settings = read_config_file(
        settings["base_config_path"], settings["base_config_file"], settings
    )
    settings = read_config_file(
        settings["loss_config_path"], settings["loss_config_file"], settings
    )
    settings = EasyDict(settings)  # To access dictionary entries as attributes

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
    settings.plots_dir = os.path.join(
        settings.plots_path,
        settings.project_name,
        settings.dataset,
        settings.net_type,
        str(settings.batch_size),
        settings.loss_type,
        settings.model_name,
    )

    if settings.use_pretrained_model == 1:
        settings.checkpoint_pretrained_dir = os.path.join(
            settings.checkpoints_path,
            settings.project_name,
            settings.dataset,
            settings.net_type,
            str(settings.batch_size),
            settings.loss_type_pretrained,
            settings.model_pretrained_name,
        )
    if not os.path.exists(settings.plots_dir):
        os.makedirs(settings.plots_dir)
    print("Saving checkpoint at", settings.checkpoint_dir)

    # Run trainig and test settings.num_seeds times, with fixed seeds for reproducibility
    seeds = np.arange(0, settings.num_seeds)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_START_METHOD"] = "thread"
    test_acc_runs = []
    test_em_ece_runs = []
    test_ew_ece_runs = []
    test_cw_ece_runs = []
    test_auc_runs = []
    test_ks_runs = []
    test_brier_runs = []

    # Multiple runs to have mean and std statistics
    for seed in seeds:
        set_seed(seed)
        settings.seed = seed

        # Set the correct wandb project name
        project_name_wandb = "aucocloss-{}-{}-{}".format(
            settings.dataset,
            settings.net_type,
            str(settings.batch_size),
        )

        with wandb.init(
            project=project_name_wandb,
            config=settings,
            dir=settings.dir_wandb,
        ):
            # Get dataset loaders
            if settings.dataset == "cifar100":
                train_loader, val_loader, test_loader = get_dataloaders_cifar(settings)
            elif settings.dataset == "tiny-imagenet":
                train_loader, val_loader, test_loader = get_dataloaders_tiny_imagenet(
                    settings
                )
            elif "mnist" in settings.dataset:
                train_loader, val_loader, test_loader = get_dataloaders_medmnist(
                    settings
                )
            elif "cifar100_LT" in settings.dataset:
                train_loader, val_loader, test_loader = get_dataloaders_cifar_LT(
                    settings
                )
            else:
                warnings.warn("Dataset is not listed.")

            print(
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Run number {:2d}  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(
                    settings.seed
                )
            )
            wandb.run.name = settings.model_name + "/seed-{:2d}".format(settings.seed)
            # Train model
            if settings.train_mode == 1:
                trainer = Trainer(settings, train_loader, val_loader)
                trainer.train()

            # Test the checpoint specified by settings.model_to_test_suffix
            checkpoint_file = "{}/{}_{:02d}_{}.pth".format(
                settings.checkpoint_dir,
                settings.model_name,
                settings.seed,
                settings.model_to_test_suffix,
            )
            if settings.use_temperature_scaling == 1:
                set_temperature_scaling(val_loader, checkpoint_file, settings)
            # Get test results
            get_new_results(
                settings,
                checkpoint_file,
                test_loader,
                test_em_ece_runs,
                test_ew_ece_runs,
                test_cw_ece_runs,
                test_acc_runs,
                test_auc_runs,
                test_ks_runs,
                test_brier_runs,
            )

    # Print mean and std statistics of the metrics over over multiple runs
    if settings.num_seeds > 1:
        print_final_results(
            settings,
            test_em_ece_runs,
            test_ew_ece_runs,
            test_cw_ece_runs,
            test_acc_runs,
            test_auc_runs,
            test_ks_runs,
            test_brier_runs,
        )


if __name__ == "__main__":
    main()
