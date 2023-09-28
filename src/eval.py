from models.wide_resnet import wide_resnet_cifar
from models.resnet import resnet50, resnet18, resnet34
from metrics.calculate_ece_metrics import (
    calculate_ECE_metrics,
)
from metrics.metrics import (
    compute_auc,
    brier_multi,
    ks_metrics,
)
import warnings
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
import wandb
import numpy as np


def evaluate(settings, test_loader, checkpoint_file):
    """
    Evaluate a trained neural network model on a test dataset and calculate various metrics.
    This function loads a trained neural network model, evaluates its performance on a test dataset,
    and calculates various metrics including calibration errors, accuracy, AUCOC, KS statistic, and Brier score.
    The metrics are computed and returned as a tuple.
    Parameters:
        - settings: A parser with configuration parameters.
        - test_loader: A DataLoader containing the test dataset.
        - checkpoint_file: Path to the checkpoint file containing the trained model weights.

    Returns:
        A tuple containing various evaluation metrics:
        - test_eq_mass_ece: Expected Mass Calibration Error (EM-ECE)
        - test_eq_width_ece: Expected Width Calibration Error (EW-ECE)
        - test_class_wise_ece: Class-wise Expected Calibration Error (CW-ECE)
        - test_accuracy: Test accuracy
        - test_auc: Test AUCOC
        - test_ks: Kolmogorov-Smirnov (KS) statistic
        - test_brier: Brier score


    """
    print(" ---> Starting the test.")
    net = setup_network(settings)
    net = nn.DataParallel(net)
    checkpoint_dict = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint_dict["net_state_dict"])
    net.to(settings.device)
    net.eval()
    labels_np = np.zeros(len(test_loader.dataset))
    predictions_np = np.zeros(len(test_loader.dataset))
    confidences_np = np.zeros((len(test_loader.dataset), settings.num_classes))
    logits_np = np.zeros((len(test_loader.dataset), settings.num_classes))

    test_eq_width_ece = 0
    test_eq_mass_ece = 0
    test_class_wise_ece = 0
    test_accuracy = 0
    test_auc = 0
    test_brier = 0
    test_ks = 0
    total = 0
    correct = 0

    # Run the test
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader, 0):
            data, test_targets = test_data
            if "mnist" in settings.dataset:
                test_targets = torch.squeeze(test_targets, 1).long()
            data, test_targets = data.to(settings.device), test_targets.to(
                settings.device
            )
            test_outputs = net(data)

            _, predictions = torch.max(test_outputs, 1)

            total += test_targets.size(0)
            correct += predictions.eq(test_targets).cpu().sum()

            confidences = F.softmax(test_outputs, dim=1).detach().cpu().numpy()
            samples_batch = test_targets.size(0)
            offset = batch_idx * test_loader.batch_size
            logits_np[offset : offset + samples_batch, :] = (
                test_outputs.detach().cpu().numpy()
            )
            labels_np[offset : offset + samples_batch] = (
                test_targets.detach().cpu().numpy()
            )
            predictions_np[offset : offset + samples_batch] = (
                predictions.detach().cpu().numpy()
            )
            if settings.use_temperature_scaling == 0:
                confidences_np[
                    offset : offset + samples_batch,
                    :,
                ] = confidences
    # Compute Accuracy
    test_accuracy = float((100.0 * correct / total).detach())
    # Rescale logits if test for TS
    if settings.use_temperature_scaling == 1:
        confidences_np = softmax(logits_np / settings.temperature, axis=1)

    # Compute Brier and KS scores
    one_hot_labels = np.zeros(
        (int(labels_np.size), int(labels_np.max()) + 1), dtype=int
    )
    one_hot_labels[np.arange(int(labels_np.size)), labels_np.astype(int)] = 1
    test_ks = ks_metrics(confidences_np, predictions_np, labels_np)
    test_brier = brier_multi(one_hot_labels, confidences_np)

    # Compute ECE metrics
    (
        test_eq_mass_ece,
        test_eq_width_ece,
        test_class_wise_ece,
    ) = calculate_ECE_metrics(
        confidences_np,
        labels_np,
        test_eq_mass_ece,
        test_eq_width_ece,
        test_class_wise_ece,
    )
    # Compute AUCOC
    test_auc = compute_auc(confidences_np, labels_np) * 100.0

    # Load to wandb
    if settings.use_temperature_scaling == 1:
        wandb.run.summary[
            "test_eq_mass_ece_TS_" + settings.model_to_test_suffix
        ] = test_eq_mass_ece
        wandb.run.summary[
            "test_eq_width_ece_TS_" + settings.model_to_test_suffix
        ] = test_eq_width_ece
        wandb.run.summary[
            "test_class_wise_ece_TS_" + settings.model_to_test_suffix
        ] = test_class_wise_ece
        wandb.run.summary[
            "test_accuracy_TS_" + settings.model_to_test_suffix
        ] = test_accuracy
        wandb.run.summary["test_auc_TS_" + settings.model_to_test_suffix] = test_auc
        wandb.run.summary["ks_TS_" + settings.model_to_test_suffix] = test_ks
        wandb.run.summary["brier_TS_" + settings.model_to_test_suffix] = test_brier
    else:
        wandb.run.summary[
            "test_eq_mass_ece_" + settings.model_to_test_suffix
        ] = test_eq_mass_ece
        wandb.run.summary[
            "test_eq_width_ece_" + settings.model_to_test_suffix
        ] = test_eq_width_ece
        wandb.run.summary[
            "test_class_wise_ece_" + settings.model_to_test_suffix
        ] = test_class_wise_ece
        wandb.run.summary[
            "test_accuracy_" + settings.model_to_test_suffix
        ] = test_accuracy
        wandb.run.summary["test_auc_" + settings.model_to_test_suffix] = test_auc
        wandb.run.summary["ks_" + settings.model_to_test_suffix] = test_ks
        wandb.run.summary["brier_" + settings.model_to_test_suffix] = test_brier

    print(
        "   - Test accuracy {:.3f}, Test AUCOC {:.3f}, Test EM-ECE {:.3f}, Test EW-ECE {:.3f}, Test CW-ECE {:.3f}, Test KS {:.3f}, Test BRIER {:.3f}, for {}.\n".format(
            test_accuracy,
            test_auc,
            test_eq_mass_ece,
            test_eq_width_ece,
            test_class_wise_ece,
            test_ks,
            test_brier,
            settings.model_to_test_suffix,
        ),
    )

    return (
        test_eq_mass_ece,
        test_eq_width_ece,
        test_class_wise_ece,
        test_accuracy,
        test_auc,
        test_ks,
        test_brier,
    )


def setup_network(settings):
    if "wide_resnet" in settings.net_type:
        net = wide_resnet_cifar(
            depth=settings.depth,
            width=settings.widen_factor,
            num_classes=settings.num_classes,
        )
    elif settings.net_type == "resnet50":
        net = resnet50(settings.num_classes)
    elif settings.net_type == "resnet18":
        net = resnet18(settings.num_classes)
    elif settings.net_type == "resnet34":
        net = resnet34(settings.num_classes)
    else:
        warnings.warn("Model is not listed.")
    net.to(settings.device)
    return net


def get_new_results(
    settings,
    checkpoint_file,
    test_loader,
    test_em_ece_runs,
    test_ew_ece_runs,
    test_cw_ece_runs,
    test_acc_runs,
    test_auc_runs,
    ks_runs,
    brier_runs,
):
    """
    Function to gather results for test.
    Parameters:
        - settings: parser for all parameters
        - checkpoint_file: name of the file of the checkpoint to be tested
        - test_loader: Torch dataloader of test set. Dataloaders are stored in aucoc_loss/data/
        - lists to append the results for all the runs (seeds)
    """
    test_em_ece, test_ew_ece, test_cw_ece, test_acc, test_auc, ks, brier = evaluate(
        settings, test_loader, checkpoint_file
    )
    test_em_ece_runs.append(test_em_ece)
    test_ew_ece_runs.append(test_ew_ece)
    test_cw_ece_runs.append(test_cw_ece)
    test_acc_runs.append(test_acc)
    test_auc_runs.append(test_auc)
    ks_runs.append(ks)
    brier_runs.append(brier)
