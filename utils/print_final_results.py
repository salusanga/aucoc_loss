import statistics
import wandb


def print_final_results(
    settings,
    test_em_ece_runs,
    test_ew_ece_runs,
    test_cw_ece_runs,
    test_acc_runs,
    test_auc_runs,
    test_ks_runs,
    test_brier_runs,
):
    print("%%%%%%%%%%%%%%%%%%% FINAL RESULTS %%%%%%%%%%%%%%%%%%%")

    print("------- TEST RESULTS AVERAGED OVER {} RUNS. -------".format(settings.num_seeds))
    print()
    print(
        "---> Final accuracy mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_acc_runs),
            statistics.stdev(test_acc_runs),
            settings.model_to_test_suffix,
        )
    )

    print(
        "---> Final AUC mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_auc_runs),
            statistics.stdev(test_auc_runs),
            settings.model_to_test_suffix,
        )
    )
    print(
        "---> Final EM-ECE mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_em_ece_runs),
            statistics.stdev(test_em_ece_runs),
            settings.model_to_test_suffix,
        )
    )
    print(
        "---> Final EW-ECE mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_ew_ece_runs),
            statistics.stdev(test_ew_ece_runs),
            settings.model_to_test_suffix,
        )
    )
    print(
        "---> Final CW-ECE mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_cw_ece_runs),
            statistics.stdev(test_cw_ece_runs),
            settings.model_to_test_suffix,
        )
    )
    print(
        "---> Final KS mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_ks_runs),
            statistics.stdev(test_ks_runs),
            settings.model_to_test_suffix,
        )
    )
    print(
        "---> Final BRIER mean {:.3f} and std {:.3f} for {} model.".format(
            statistics.mean(test_brier_runs),
            statistics.stdev(test_brier_runs),
            settings.model_to_test_suffix,
        )
    )
