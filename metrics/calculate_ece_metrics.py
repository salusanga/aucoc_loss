from metrics.metrics import (
    ECELoss,
    AdaptiveECELoss,
    ClasswiseECELoss,
)


def calculate_ECE_metrics(
    confidences, targets, eq_mass_ece, eq_width_ece, class_wise_ece
):
    """Helper function to gather ECE, AdaECE and Class-Wise ECE results."""

    equal_intervals_ece_loss = ECELoss()
    equal_mass_ece_loss = AdaptiveECELoss()
    class_wise_ece_loss = ClasswiseECELoss()
    eq_width_ece += equal_intervals_ece_loss.calculate_ece(
        softmaxes=confidences, labels=targets
    )
    eq_mass_ece += equal_mass_ece_loss.calculate_ece(
        softmaxes=confidences, labels=targets
    )
    class_wise_ece += class_wise_ece_loss.calculate_sce(
        softmaxes=confidences, labels=targets
    )
    return eq_mass_ece, eq_width_ece, class_wise_ece
