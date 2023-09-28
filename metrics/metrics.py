"""
Metrics to measure AUCOC and calibration of a trained deep neural network.
References and readaptation of calibration metrics partly from:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
"""

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

from loss_functions.auc_loss_bw import get_thresholds_from_cdf_np


def compute_auc(confidences, labels):
    """
    This function computes the AUCOC (Area Under the Curve of 1 - % of samples the NN takes a decision for
    VS 1-error) using softmax confidences and ground truth labels. It calculates the AUCOC by varying a
    confidence threshold and measuring the trade-off between the percentage of samples where the neural
    network makes a decision and the 1-error rate.

    Parameters:
        - confidences: numpy array of softmax output [num samples, classes].
        - labels: numpy array of labels [num samples].

    Returns:
        AUCOC (Area Under the Curve of 1 - % of samples the NN takes a decision for VS 1-error).
    """
    size_dataset = labels.size
    r = np.amax(confidences, axis=1)
    thresholds_confidence = get_thresholds_from_cdf_np(r)
    thresholds_confidence = np.sort(r)
    predictions = np.argmax(confidences, axis=1)
    samples_without_decision = np.zeros_like(thresholds_confidence)
    one_minus_error = np.zeros_like(thresholds_confidence)

    for i in range(thresholds_confidence.size):
        num = np.sum(
            np.logical_and(
                np.asarray(r >= thresholds_confidence[i]),
                np.not_equal(labels, predictions),
            )
        )
        denum = np.sum(np.asarray(r >= thresholds_confidence[i])) + 1e-10

        one_minus_error[i] = 1.0 - (num / denum).item()

        samples_without_decision[i] = (
            np.sum(np.asarray(r < thresholds_confidence[i])) / size_dataset
        ).item()
    aucoc = auc(samples_without_decision, one_minus_error)
    return aucoc


# Some keys used for the following dictionaries
COUNT = "count"
CONF = "conf"
ACC = "acc"
BIN_ACC = "bin_acc"
BIN_CONF = "bin_conf"


def _bin_initializer(bin_dict, num_bins=15):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=15):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(
                bin_dict[binn][COUNT]
            )
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece


def maximum_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


def average_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    non_empty_bins = 0
    ace = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        if bin_count > 0:
            non_empty_bins += 1
        ace += abs(bin_accuracy - bin_confidence)
    return ace / float(non_empty_bins)


def l2_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    l2_sum = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        l2_sum += (float(bin_count) / num_samples) * (
            bin_accuracy - bin_confidence
        ) ** 2
        l2_error = math.sqrt(l2_sum)
    return l2_error


def test_classification_net_logits(logits, labels):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    softmax = F.softmax(logits, dim=1)
    confidence_vals, predictions = torch.max(softmax, dim=1)
    labels_list.extend(labels.cpu().numpy().tolist())
    predictions_list.extend(predictions.cpu().numpy().tolist())
    confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


def test_classification_net(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)

            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            confidence_vals, predictions = torch.max(softmax, dim=1)

            labels_list.extend(label.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)

    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


class ECELoss:
    """
    Compute ECE (Expected Calibration Error).
    """

    def __init__(self, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calculate_ece(self, softmaxes, labels):
        confidences = np.amax(softmaxes, axis=1)
        predictions = np.argmax(softmaxes, axis=1)
        accuracies = np.equal(predictions, labels)
        ece = np.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater_equal(confidences, bin_lower.item()) * np.less_equal(
                confidences, bin_upper.item()
            )
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item() * 100.0


class AdaptiveECELoss:
    """Compute Adaptive ECE."""

    def __init__(self, n_bins=15):
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, self.nbins + 1), np.arange(npt), np.sort(x)
        )

    def calculate_ece(self, softmaxes, labels):
        confidences = np.amax(softmaxes, axis=1)
        predictions = np.argmax(softmaxes, axis=1)
        accuracies = np.equal(predictions, labels)
        n, bin_boundaries = np.histogram(
            confidences,
            self.histedges_equalN(confidences),
        )
        # print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = np.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater_equal(confidences, bin_lower.item()) * np.less_equal(
                confidences, bin_upper.item()
            )
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item() * 100.0


class ClasswiseECELoss:
    """Compute Classwise ECE."""

    def __init__(self, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calculate_sce(self, softmaxes, labels):
        num_classes = int((np.amax(labels) + 1).item())
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = np.zeros(1)
            labels_in_class = np.equal(labels, i)

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = np.greater_equal(
                    class_confidences, bin_lower.item()
                ) * np.less_equal(class_confidences, bin_upper.item())
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = np.mean(labels_in_class[in_bin])
                    avg_confidence_in_bin = np.mean(class_confidences[in_bin])
                    class_sce += (
                        np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    )

            if i == 0:
                per_class_sce = class_sce
            else:
                per_class_sce = np.concatenate((per_class_sce, class_sce), axis=0)

        sce = np.mean(per_class_sce)
        return sce.item() * 100.0


# Calibration error scores in the form of loss metrics
class ECELossTorch(nn.Module):
    """Compute ECE (Expected Calibration Error)."""

    def __init__(self, n_bins=15):
        super(ECELossTorch, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def brier_multi(targets, probs):
    """Brier score for multi class problem."""
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def ks_metrics(confidences, predictions, labels):
    """KS score."""
    c = np.amax(confidences, axis=1)
    r = np.equal(predictions, labels)
    nsamples = r.size
    integrated_accuracy = np.cumsum(c) / nsamples
    integrated_scores = np.cumsum(r) / nsamples

    ks = np.amax(np.abs(integrated_accuracy - integrated_scores))
    return ks
