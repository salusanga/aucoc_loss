"""
Code adapted from:
Karandikar, Archit et al. “Soft Calibration Objectives for Neural Networks.”
 Neural Information Processing Systems (2021).
 """
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

EPS = 1e-5


class SBEceLoss(nn.Module):
    def __init__(self, temp, n_bins=15):
        super(SBEceLoss, self).__init__()
        self.temp = temp
        self.n_bins = n_bins

    def forward(self, logits, labels):
        device = labels.device
        all_confidences = F.log_softmax(logits, dim=1)
        predictions, confidences = torch.max(all_confidences, 1)
        soft_binning_anchors = torch.arange(
            1.0 / (2.0 * self.n_bins), 1.0, 1.0 / (self.n_bins)
        )
        difference_confidences_bin_anchors = torch.zeros(
            confidences.size(dim=0), soft_binning_anchors.size(dim=0)
        )
        for i in range(confidences.size(dim=0)):
            for j in range(soft_binning_anchors.size(dim=0)):
                difference_confidences_bin_anchors[i, j] = (
                    -((confidences[i] - soft_binning_anchors[j]) ** 2) / self.temp
                )
                j = j + 1
            i = i + 1

        predictions_soft_binning_coeffs = F.softmax(
            difference_confidences_bin_anchors,
            dim=1,
        ).to(device)
        sum_coeffs_for_bin = torch.sum(predictions_soft_binning_coeffs, dim=0).to(
            device
        )
        eps_tensor = EPS * torch.ones_like(sum_coeffs_for_bin)
        confidences_intermediate = (
            torch.unsqueeze(confidences, 1)
            .repeat([1, soft_binning_anchors.size(dim=0)])
            .to(device)
        )
        # Obtain vector of bin confidences
        sum_mul_confidences_coefficients_for_bin = torch.sum(
            torch.mul(confidences_intermediate, predictions_soft_binning_coeffs), dim=0
        )
        bins_confidence = torch.div(
            sum_mul_confidences_coefficients_for_bin,
            torch.max(sum_coeffs_for_bin, eps_tensor),
        )

        # Obtain vector of bin accuracies
        accuracies_intermediate = (
            torch.unsqueeze(predictions.eq(labels), 1)
            .repeat([1, soft_binning_anchors.size(dim=0)])
            .to(device)
        )

        sum_mul_accuracies_coefficients_for_bin = torch.sum(
            torch.mul(accuracies_intermediate, predictions_soft_binning_coeffs), dim=0
        )
        bins_accuracy = torch.div(
            sum_mul_accuracies_coefficients_for_bin,
            torch.max(sum_coeffs_for_bin, eps_tensor),
        )

        bin_weights = F.normalize(sum_coeffs_for_bin, p=1, dim=0)

        soft_binning_ece = torch.sqrt(
            torch.tensordot(
                torch.square(torch.sub(bins_confidence, bins_accuracy)),
                bin_weights,
                dims=1,
            )
        )
        return soft_binning_ece

        # confidence_pred_tile = (torch.unsqueeze(confidences, 1)).repeat(
        #     [1, soft_binning_anchors.size(dim=0)]
        # )
        # confidence_pred_tile = torch.unsqueeze(confidences, 2)

        # bin_anchors_tile = (torch.unsqueeze(soft_binning_anchors, 0)).repeat(
        #     [confidences.size(dim=0), 1]
        # )
        # bin_anchors_tile = torch.unsqueeze(bin_anchors_tile, 2).to(device)

        # confidence_pred_bin_anchors_product = torch.cat(
        #     (confidence_pred_tile, bin_anchors_tile), dim=2
        # )

    #


# print(
#     confidence_pred_tile.size(),
#     bin_anchors_tile.size(),
#     confidence_pred_bin_anchors_product.size(),
# )
