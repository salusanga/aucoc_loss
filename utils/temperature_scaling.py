"""
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
"""
from scipy.special import softmax
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from metrics.calculate_ece_metrics import (
    calculate_ECE_metrics,
)
from src.eval import setup_network


def set_temperature_scaling(val_loader, checkpoint_file, settings):
    """
    Function to find the optimal temperature for temperature scaling, based on lowest equal mass
    ECE. Optimal temperature saved in settings.temperature.
    Parameters:
        - val_loader: Torch dataloader of validation set to be used to find the optimal temperature.
            Dataloaders can be found in /aucocloss/data/
        - checkpoint_file: file where the checkpoint to be assessed is stored
        - settings: parser for all parameters

    Optimal temperature saved in settings.temperature.
    """
    net = setup_network(settings)
    net = nn.DataParallel(net)
    checkpoint_dict = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint_dict["net_state_dict"])
    net.to(settings.device)
    if "mnist" in settings.dataset:
        length_dataset = len(val_loader.dataset)
    else:
        length_dataset = int(np.floor(settings.val_set_perc * len(val_loader.dataset)))
    labels_np = np.zeros(length_dataset)
    predictions_np = np.zeros(length_dataset)
    confidences_np = np.zeros((length_dataset, settings.num_classes))
    logits_np = np.zeros((length_dataset, settings.num_classes))

    val_eq_width_ece_before_scale = 0
    val_eq_mass_ece_before_scale = 0
    val_class_wise_ece_before_scale = 0

    net.eval()

    for batch_idx, val_data in enumerate(val_loader, 0):
        data, val_targets = val_data
        if "mnist" in settings.dataset:
            val_targets = torch.squeeze(val_targets, 1).long()
        data, val_targets = data.to(settings.device), val_targets.to(settings.device)
        val_outputs = net(data)

        # Calculate losses
        _, predictions = torch.max(val_outputs, 1)  # Get predictions

        # Create arrays for the whole dataset
        confidences = F.softmax(val_outputs, dim=1).detach().cpu().numpy()
        samples_batch = val_targets.size(0)
        offset = batch_idx * val_loader.batch_size
        logits_np[offset : offset + samples_batch, :] = (
            val_outputs.detach().cpu().numpy()
        )
        labels_np[offset : offset + samples_batch] = val_targets.detach().cpu().numpy()
        predictions_np[offset : offset + samples_batch] = (
            predictions.detach().cpu().numpy()
        )
        confidences_np[offset : offset + samples_batch, :] = confidences

    # Calculate metrics: accuracy, EM-ECE, EW-ECE, CW-ECE
    (
        val_eq_mass_ece_before_scale,
        val_eq_width_ece_before_scale,
        val_class_wise_ece_before_scale,
    ) = calculate_ECE_metrics(
        confidences_np,
        labels_np,
        val_eq_mass_ece_before_scale,
        val_eq_width_ece_before_scale,
        val_class_wise_ece_before_scale,
    )
    T_opt = 1.0
    T = 0.1
    ece_val = 10**7

    for i in range(100):
        val_eq_width_ece_after_scale = 0
        val_eq_mass_ece_after_scale = 0
        val_class_wise_ece_after_scale = 0
        settings.temperature = T
        confidences_from_scale = softmax(logits_np / settings.temperature, axis=1)

        (
            val_eq_mass_ece_after_scale,
            val_eq_width_ece_after_scale,
            val_class_wise_ece_after_scale,
        ) = calculate_ECE_metrics(
            confidences_from_scale,
            labels_np,
            val_eq_mass_ece_after_scale,
            val_eq_width_ece_after_scale,
            val_class_wise_ece_after_scale,
        )
        if ece_val > val_eq_mass_ece_after_scale:
            T_opt = T
            ece_val = val_eq_mass_ece_after_scale
        T += 0.1

    settings.temperature = T_opt
    print("Optimal temperature found: {:.3f}".format(settings.temperature))
    print(
        "Validation before temperature - EM-ECE: {:.3f} after temperature - EM-ECE: {:.3f}".format(
            val_eq_mass_ece_before_scale, ece_val
        )
    )
