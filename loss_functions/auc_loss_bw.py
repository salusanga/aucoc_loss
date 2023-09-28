import torch
from torch.autograd import Function
import numpy as np
from numpy import cov, linalg, atleast_2d
from scipy import stats
import math

NPY_SQRT1_2 = 0.707106781186547524400844362104849039  # /* 1/sqrt(2) */


class AUCLossBw(Function):
    """
    This class defines the proposed loss function, AUCOCLoss, with a custom gradient implementation.
    It takes softmax confidences, target class vectors, and a weighting factor as input and calculates
    the loss. The loss function includes the computation of the area under the ROC curve (AUC) and
    the gradient with respect to the model's confidences. More details about the mathematical
    derivations are provided in supplementary materials.

    Parameters:
        - confidences: Softmax confidences of shape [batch_size, num_classes].
        - targets: Target class vector.
        - lamda: Scalar indicating the weighting factor of the secondary loss.
    """

    @staticmethod
    def forward(ctx, confidences, targets, lamda):
        r, predictions = torch.max(confidences, 1)
        device = r.device
        c = torch.squeeze(confidences.gather(1, torch.unsqueeze(targets, 1)))
        r0 = get_thresholds_from_cdf(r).to(device)
        cov, alpha = compute_cov_alpha(r.detach().cpu().numpy())
        r_rep = r.unsqueeze(-1).repeat(1, r0.shape[0])
        c_rep = c.unsqueeze(-1).repeat(1, r0.shape[0])

        area = torch.zeros(1, device=device)

        r0_minus_r = r0 - r_rep
        ones_minus_r = 1.0 - r_rep

        t0 = torch.sum(
            (1 / (r.size(dim=0)))
            * (
                ndtr(r0_minus_r * (1 / math.sqrt(cov)))
                - ndtr(-r_rep * (1 / math.sqrt(cov)))
            ),
            0,
        )

        one_minus_t0 = torch.ones_like(t0) - t0
        erf_diff_area = (
            ndtr(ones_minus_r * (1 / math.sqrt(cov)))
            - ndtr(r0_minus_r * (1 / math.sqrt(cov)))
        ) * c_rep
        integrals = (
            (1 / (r.size(dim=0))) * torch.sum(erf_diff_area, 0) * (1 / one_minus_t0)
        )

        delta_t0 = t0[1:] - t0[:-1]

        delta_t0 = torch.cat((torch.tensor([t0[0]]).to(device), delta_t0), dim=0)
        integrals_sum = integrals[1:] + integrals[:-1]
        integrals_sum = torch.cat(
            (torch.tensor([integrals[0]]).to(device), integrals_sum), dim=0
        )
        area = 0.5 * torch.sum(integrals_sum * (delta_t0))
        auc_loss = -torch.log(area)
        ctx.save_for_backward(
            r_rep,
            r0,
            one_minus_t0,
            c_rep,
            confidences,
            predictions,
            targets,
            area,
            delta_t0,
        )
        ctx.alpha = alpha
        ctx.cov = cov
        ctx.lamda = lamda

        return auc_loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            r_rep,
            r0,
            one_minus_t0,
            c_rep,
            confidences,
            predictions,
            targets,
            area,
            delta_t0,
        ) = ctx.saved_tensors
        alpha = ctx.alpha
        cov = ctx.cov
        lamda = ctx.lamda

        # Auxiliuary terms
        grad_input = torch.zeros_like(confidences)
        coefficient = -1 / (math.sqrt(2 * math.pi) * alpha * r_rep.size(dim=0))
        r0_minus_r = r0 - r_rep
        ones_minus_r = 1.0 - r_rep

        # Terms for dA/dr_n and r_0_coefficient
        diff_exp_1 = (
            torch.exp(-torch.pow(ones_minus_r, 2) * (1 / (2 * (alpha**2))))
            - torch.exp(-torch.pow(r0_minus_r, 2) * (1 / (2 * (alpha**2))))
        ) * c_rep
        r0_coefficient = (
            torch.sum(
                torch.multiply(
                    c_rep,
                    torch.exp(-torch.pow(r0_minus_r, 2) * (1 / (2 * (alpha**2)))),
                ),
                axis=0,
            )
            * 1
            / torch.sum(
                torch.exp(-torch.pow(r0_minus_r, 2) * (1 / (2 * (alpha**2)))),
                axis=0,
            )
        )
        diff_exp_2 = (
            torch.exp(-torch.pow(r0_minus_r, 2) * (1 / (2 * (alpha**2))))
            - torch.exp(-torch.pow(r_rep, 2) * (1 / (2 * (alpha**2))))
        ) * r0_coefficient

        # External integral for r_n
        integrals_r = coefficient * (diff_exp_1 + diff_exp_2) * 1 / one_minus_t0

        integrals_sum_r = integrals_r[:, 1:] + integrals_r[:, :-1]
        integrals_sum_r = torch.cat(
            (torch.unsqueeze(integrals_r[:, 0], dim=1), integrals_sum_r), dim=1
        )
        grad_area_r = 0.5 * (-1 / area) * torch.sum(integrals_sum_r * delta_t0, 1)

        # External integral for r*
        integrals_r_star = (
            (1 / r_rep.size(dim=0))
            * (
                ndtr(ones_minus_r * (1 / math.sqrt(cov)))
                - ndtr(r0_minus_r * (1 / math.sqrt(cov)))
            )
            * 1
            / one_minus_t0
        )
        integrals_sum_r_star = integrals_r_star[:, 1:] + integrals_r_star[:, :-1]
        integrals_sum_r_star = torch.cat(
            (torch.unsqueeze(integrals_r_star[:, 0], dim=1), integrals_sum_r_star),
            dim=1,
        )
        grad_area_r_star = (
            0.5 * (-1 / area) * torch.sum(integrals_sum_r_star * delta_t0, 1)
        )

        # Update the gradients for r_n and r*
        grad_input.scatter_(
            1,
            torch.unsqueeze(predictions, 1),
            torch.unsqueeze(
                grad_area_r,
                1,
            ),
        )
        grad_input.scatter_add_(
            1,
            torch.unsqueeze(targets, 1),
            torch.unsqueeze(
                grad_area_r_star,
                1,
            ),
        )

        return grad_input * lamda, None, None, None


# Auxiliary functions


def ndtr(a):
    x = a * NPY_SQRT1_2
    z = torch.abs(x)

    y = torch.where((z < NPY_SQRT1_2), 0.5 + 0.5 * torch.erf(x), 0.5 * torch.erfc(z))
    y = torch.where(torch.logical_and(z >= NPY_SQRT1_2, x > 0), 1.0 - y, y)

    return y


def compute_cdf(r, r0_vector, device, alpha):
    i = 0
    cdf = torch.zeros_like(r0_vector)
    for r0 in r0_vector:
        cdf[i] = 1 - torch.sum(
            torch.where(
                r <= r0,
                (
                    (torch.exp(-(r0 - r) / alpha) - torch.exp(-(1.0 - r) / alpha))
                    / r.size(dim=0)
                ).to(device),
                (
                    2.0
                    - torch.exp(-(r - r0) / alpha)
                    - torch.exp(-(1.0 - r) / alpha) / r.size(dim=0)
                ).to(device),
            )
        )
        i = i + 1
    # print(cdf[0:10], r0_vector[0:10])


def get_thresholds_from_cdf(confidences):
    """Get thresholds from confidences distribution."""
    num_thresholds = confidences.size(dim=0)
    confidences_sorted, _ = torch.sort(confidences)

    cdf = torch.arange(confidences_sorted.size(dim=0)) / float(num_thresholds)
    index_samples_uniform_cdf = torch.arange(
        0,
        confidences_sorted.size(dim=0),
        step=int(confidences_sorted.size(dim=0) / num_thresholds),
    )
    samples_uniform_cdf = cdf[index_samples_uniform_cdf]
    thresholds = confidences_sorted[np.where(cdf == samples_uniform_cdf)[0]]
    # thresholds = thresholds[thresholds <= float(0.99995)]
    return thresholds


def get_thresholds_from_cdf_np(confidences):
    """Get thresholds from confidences distribution, implemented in numpy."""
    num_thresholds = confidences.size
    thresholds = np.zeros(num_thresholds)
    confidencs_sorted = np.sort(confidences)
    # get the cdf confidencs_sorted of y
    cdf = np.arange(confidencs_sorted.size) / float(num_thresholds)
    index_samples_uniform_cdf = np.arange(
        0, confidences.size, step=int(confidences.size / num_thresholds)
    )

    samples_uniform_cdf = cdf[index_samples_uniform_cdf]
    j = 0
    for i in samples_uniform_cdf:
        thresholds[j] = confidencs_sorted[np.where(cdf == i)[0]]
        j += 1
    return thresholds


def compute_cov_alpha(r):
    """Covariance and alpha computation."""
    pdf = stats.gaussian_kde(r, bw_method=None, weights=None)
    factor = pdf.scotts_factor()
    precision = (linalg.inv(atleast_2d(cov(r, rowvar=1, bias=False)))) / factor**2
    dtype = np.common_type(precision, r)
    whitening = np.linalg.cholesky(precision).astype(dtype, copy=False)
    covariance = factor**2 * atleast_2d(cov(r, rowvar=1, bias=False))
    covariance = covariance[0, 0]
    alpha = 1 / whitening[0, 0]
    return covariance, alpha
