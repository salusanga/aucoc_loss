"""
Code adapted from:
Karandikar, Archit et al. “Soft Calibration Objectives for Neural Networks.”
 Neural Information Processing Systems (2021).
"""
import torch.nn.functional as F
import torch
from torch import nn


class SoftAvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default);
    1: model uncertainty]
    """

    def __init__(self, temp, k):
        super(SoftAvULoss, self).__init__()
        self.eps = 1e-10
        self.temp = temp
        self.k = k

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(mc_preds, dim=0)) - self.expected_entropy(
            mc_preds
        )

    def accuracy_vs_uncertainty(
        self, prediction, true_label, uncertainty, optimal_threshold
    ):
        # number of samples accurate and certain
        n_ac = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and certain
        n_ic = torch.zeros(1, device=true_label.device)
        # number of samples accurate and uncertain
        n_au = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device)

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if (true_label[i].item() == prediction[i].item()) and uncertainty[
                i
            ].item() <= optimal_threshold:
                """accurate and certain"""
                n_ac += 1
            elif (true_label[i].item() == prediction[i].item()) and uncertainty[
                i
            ].item() > optimal_threshold:
                """accurate and uncertain"""
                n_au += 1
            elif (true_label[i].item() != prediction[i].item()) and uncertainty[
                i
            ].item() <= optimal_threshold:
                """inaccurate and certain"""
                n_ic += 1
            elif (true_label[i].item() != prediction[i].item()) and uncertainty[
                i
            ].item() > optimal_threshold:
                """inaccurate and uncertain"""
                n_iu += 1

        print("n_ac: ", n_ac, " ; n_au: ", n_au, " ; n_ic: ", n_ic, " ;n_iu: ", n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels):

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        entropies = self.entropy(probs)
        entropies_norm = entropies / probs.size(dim=1)
        sigmoid = torch.nn.Sigmoid()
        soft_unc_funct = sigmoid(
            1
            / self.temp
            * torch.log(
                torch.div(
                    entropies_norm * (1.0 - self.k), (1.0 - entropies_norm) * self.k
                )
            )
        )

        n_ac = torch.zeros(
            1, device=logits.device
        )  # number of samples accurate and certain
        n_ic = torch.zeros(
            1, device=logits.device
        )  # number of samples inaccurate and certain
        n_au = torch.zeros(
            1, device=logits.device
        )  # number of samples accurate and uncertain
        n_iu = torch.zeros(
            1, device=logits.device
        )  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=logits.device)
        avu_loss = torch.zeros(1, device=logits.device)

        for i in range(len(labels)):
            if labels[i].item() == predictions[i].item():
                """accurate and certain"""
                n_ac += (1 - soft_unc_funct[i]) * (1 - torch.tanh(entropies[i]))
                """accurate and uncertain"""
                n_au += soft_unc_funct[i] * torch.tanh(entropies[i])
            elif labels[i].item() != predictions[i].item():
                """inaccurate and certain"""
                n_ic += (1 - soft_unc_funct[i]) * (1 - torch.tanh(entropies[i]))
                """inaccurate and uncertain"""
                n_iu += (soft_unc_funct[i]) * torch.tanh(entropies[i])

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        p_ac = (n_ac) / (n_ac + n_ic)
        p_ui = (n_iu) / (n_iu + n_ic)
        # print('Actual AvU: ', self.accuracy_vs_uncertainty(predictions, labels, uncertainty, optimal_threshold))
        avu_loss = -1 * torch.log(avu + self.eps)
        return avu_loss
