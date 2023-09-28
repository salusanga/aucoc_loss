"""
This file contains method for generating calibration related plots, eg. reliability plots.
References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
Added 
"""
import os
import math
import matplotlib.pyplot as plt
import wandb
import numpy as np
from numpy import cov, linalg, atleast_2d
from sklearn.metrics import auc
from scipy import stats
from loss_functions.auc_loss_bw import (
    get_thresholds_from_cdf_np,
)

plt.rcParams.update({"font.size": 20})

# Some keys used for the following dictionaries
COUNT = "count"
CONF = "conf"
ACC = "acc"
BIN_ACC = "bin_acc"
BIN_CONF = "bin_conf"


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
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


def reliability_plot(confs, preds, labels, plot_name, num_bins=15):
    """
    Method to draw a reliability plot from a model's predictions and confidences.
    """
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align="edge", width=0.05, color="pink", label="Expected")
    plt.bar(bns, y, align="edge", width=0.05, color="blue", alpha=0.5, label="Actual")
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")

    plt.savefig(plot_name + ".png")


def bin_strength_plot(confs, preds, labels, plot_name, num_bins=15):
    """
    Method to draw a plot for the number of samples in each confidence bin.
    """
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples)) * 100
        y.append(n)
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(
        bns,
        y,
        align="edge",
        width=0.05,
        color="blue",
        alpha=0.5,
        label="Percentage samples",
    )
    plt.ylabel("Percentage of samples")
    plt.xlabel("Confidence")
    plt.savefig(plot_name + ".png")


def roc_no_decision(confidences, labels, settings, plot_name):
    """
    Plots of 1 - % of samples you take a decision for VS 1-error.
    """
    size_dataset = labels.size
    r = np.amax(confidences, axis=1)
    thresholds_confidence = get_thresholds_from_cdf_np(r)
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

    # print(one_minus_error[-100:], samples_without_decision[-100:])
    fig = plt.figure(figsize=(10, 8))
    auc_plot = auc(samples_without_decision, one_minus_error)
    plt.plot(samples_without_decision, one_minus_error, label="AUC = %0.3f" % auc_plot)
    plt.ylabel("1 - error (%)")
    plt.xlabel("Samples without decision (%)")
    plt.ylim([0.9, 1.00])
    plt.yticks(np.arange(0.9, 1.0, step=0.05))
    plt.grid(ls="--", lw=0.5, markevery=0.05)
    plt.legend(loc="lower right")
    plt.savefig(plot_name + ".png")
    if settings.plot_together == 1:
        settings.np_all_one_minus_error[:, settings.count] = one_minus_error
        settings.np_all_undecided[:, settings.count] = samples_without_decision
    compute_interesting_values(
        settings.model_to_test_suffix,
        one_minus_error,
        samples_without_decision,
        settings.use_temperature_scaling,
    )


def plot_roc_together(settings):
    """
    Plots of 1 - % of samples you take a decision for VS 1-error for multiple models together.
    - settings: parser for all parameters
    - settings.plots_folder: path to plots folder
    - checkpoint_file: useful to understand which checkpoint analysing,
        i.e. based on best accuracy or best AUCOC.
    """
    list_colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple"]

    # uncomment
    plot_name = os.path.join(
        settings.plots_folder,
        settings.project_name,
        settings.dataset,
        settings.net_type,
        "compare_roc" + settings.model_to_test_suffix + "_CE.png",
    )

    plt.figure(figsize=(10, 8))
    for i in range(len(settings.loss_configs_array)):
        samples_without_decision = settings.np_all_undecided[:, i]
        one_minus_error = settings.np_all_one_minus_error[:, i]

        auc_plot = auc(samples_without_decision, one_minus_error)
        auc_plot = auc_plot * 100.0
        # print(samples_without_decision, one_minus_error)
        plt.plot(
            samples_without_decision * 100.00,
            one_minus_error * 100.00,
            color=list_colors[i],
            label="{}, AUCOC {:.2f}".format(
                settings.loss_type_array[i],
                auc_plot,
                # settings.use_temperature_scaling_array[i],
            ),
        )
        if samples_without_decision[-1] < 0.99000:
            plt.plot(
                [
                    samples_without_decision[-1] * 100.00,
                    samples_without_decision[-1] * 100.00,
                ],
                [one_minus_error[-1] * 100.00, 0],
                color=list_colors[i],
                linestyle="dashed",
                alpha=0.7,
            )
    plt.ylabel("E" + r"$[c|r>r_0]$" + ": accuracy of the network (%)")
    plt.xlabel(r"$\tau_0$" + ": samples to be analysed manually (%)")
    plt.ylim([75.00, 100.00])
    plt.xlim([00.0, 100.00])
    plt.yticks(np.arange(75.00, 100.00, step=10.00))
    plt.grid(ls="--", lw=0.5, markevery=10.00)
    plt.legend(loc="lower right", fontsize="small")
    plt.savefig(plot_name)


def compute_interesting_values(
    suffix, accuracies_vect, tau0_vect, use_temperature_scaling
):
    """
    Function to report specific operating points on COC curve. E.g. tau0 at specific accuracy levels.
    """
    int_list_acc = [0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95]
    int_list_t0 = [0.2, 0.3, 0.4, 0.5]

    # tau0 for certain accuracies
    for acc in int_list_acc:
        difference_array = np.absolute(accuracies_vect - acc)
        index = difference_array.argmin()
        print(
            "At accuracy {:.2f} tau0 is {:.2f}".format(
                acc * 100.00, tau0_vect[index] * 100.00
            )
        )
        if use_temperature_scaling == 0:
            wandb.run.summary["t0_at_" + str(acc * 100.00) + suffix] = (
                tau0_vect[index] * 100.00
            )
        elif use_temperature_scaling == 1:
            wandb.run.summary["t0_at_" + str(acc * 100.00) + suffix + "_TS"] = (
                tau0_vect[index] * 100.00
            )


def compute_cov_alpha(r):
    pdf = stats.gaussian_kde(r, bw_method=None, weights=None)
    factor = pdf.scotts_factor()
    precision = (linalg.inv(atleast_2d(cov(r, rowvar=1, bias=False)))) / factor**2
    dtype = np.common_type(precision, r)
    whitening = np.linalg.cholesky(precision).astype(dtype, copy=False)
    covariance = factor**2 * atleast_2d(cov(r, rowvar=1, bias=False))
    covariance = covariance[0, 0]
    alpha = 1 / whitening[0, 0]
    return covariance, alpha
