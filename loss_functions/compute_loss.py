import warnings
import torch
from torch.nn import functional as F
from loss_functions.baselines.focal_loss import FocalLoss
from loss_functions.baselines.focal_loss_ada_gamma import FocalLossAdaptive
from loss_functions.baselines.mmce import MMCE_weighted
from loss_functions.baselines.soft_avuc import SoftAvULoss
from loss_functions.baselines.soft_ece import SBEceLoss
from loss_functions.auc_loss_bw import AUCLossBw


# Compute loss with sum of loss instead of average
def compute_loss(settings, logits, targets):
    """
    This function selects and computes the appropriate loss function based on the settings provided.
    The choice of loss function is controlled by the 'settings.loss_type' parameter. Depending on
    the loss type, the function computes the corresponding loss and returns it.
    Parameters:
        - settings: parser with all the required information.
        - logits: The predicted logits from the model.
        - targets: The ground truth labels.

    Returns:
        - loss: The computed loss value(s) depending on the selected loss function(s).
    """

    ### Baselines ###
    if settings.loss_type == "cross-entropy":
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = criterion(logits, targets)
        return loss

    elif settings.loss_type == "focal-loss":
        criterion = FocalLoss(gamma=settings.gamma_FL)
        return criterion(logits, targets)

    elif settings.loss_type == "focal-loss-ada":
        criterion = FocalLossAdaptive(device=settings.device, gamma=settings.gamma_FL)
        return criterion(logits, targets)

    elif settings.loss_type == "mmce":
        criterion_secondary = MMCE_weighted(device=settings.device)(logits, targets)
        if settings.primary_loss_type == "cross-entropy":
            criterion_primary = torch.nn.CrossEntropyLoss()(logits, targets)
        elif settings.primary_loss_type == "focal-loss":
            criterion_primary = FocalLoss(gamma=settings.gamma_FL, size_average=True)(
                logits, targets
            )
        return (
            (torch.mul(criterion_secondary, settings.lamda) + criterion_primary)
            * len(targets),
            criterion_primary * len(targets),
            criterion_secondary * len(targets) * settings.lamda,
        )

    elif settings.loss_type == "soft_avuc":
        criterion_secondary = SoftAvULoss(temp=settings.temp_savuc, k=settings.k_savuc)(
            logits=logits,
            labels=targets,
        )

        if settings.primary_loss_type == "cross-entropy":
            criterion_primary = torch.nn.CrossEntropyLoss()(logits, targets)
        elif settings.primary_loss_type == "focal-loss":
            criterion_primary = FocalLoss(gamma=settings.gamma_FL, size_average=True)(
                logits, targets
            )
        return (
            (torch.mul(criterion_secondary, settings.lamda) + criterion_primary),
            criterion_primary,
            criterion_secondary * settings.lamda,
        )

    elif settings.loss_type == "soft_ece":
        criterion_secondary = SBEceLoss(temp=settings.temp_soft_ece)(
            logits=logits,
            labels=targets,
        )

        if settings.primary_loss_type == "cross-entropy":
            criterion_primary = torch.nn.CrossEntropyLoss()(logits, targets)
        elif settings.primary_loss_type == "focal-loss":
            criterion_primary = FocalLoss(gamma=settings.gamma_FL, size_average=True)(
                logits, targets
            )
        return (
            (torch.mul(criterion_secondary, settings.lamda) + criterion_primary),
            criterion_primary,
            criterion_secondary * settings.lamda,
        )

    ### Proposed loss ###

    elif settings.loss_type == "auc_secondary_bw":
        confidences = F.softmax(logits, dim=1)
        criterion_secondary = AUCLossBw.apply(confidences, targets, settings.lamda)
        if settings.primary_loss_type == "cross-entropy":
            criterion_primary = torch.nn.CrossEntropyLoss(reduction="sum")(
                logits, targets
            )
        elif settings.primary_loss_type == "focal-loss":
            criterion_primary = FocalLoss(gamma=settings.gamma_FL)(logits, targets)

        sum_losses = (
            torch.mul(criterion_secondary, len(targets)) * settings.lamda
            + criterion_primary
        )

        return (
            sum_losses,
            criterion_primary,
            torch.mul(criterion_secondary, len(targets)) * settings.lamda,
        )

    else:
        warnings.warn("Loss function is not listed.")
