import numpy as np
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import warnings
import wandb
from models.wide_resnet import wide_resnet_cifar
from models.resnet import resnet50, resnet18, resnet34
from utils.load_save_checkpoint import load_checkpoint, save_checkpoint, load_model
from metrics.calculate_ece_metrics import calculate_ECE_metrics
from metrics.metrics import compute_auc
from loss_functions.compute_loss import compute_loss
from loss_functions.baselines.avuc import entropy


class Trainer:
    def __init__(self, settings, train_loader, val_loader):
        """
        Parameters:
            - train_loader, val_loader: Torch loaders. In each epoch, the trainer runs one
            epoch for each loader.
            - settings: Training settings.
        """
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self.settings.device

        # Initialize statistics variables to track during training
        metrics_keys = [
            "eq_mass_ece",
            "eq_width_ece",
            "class_wise_ece",
            "accuracy",
            "auc",
        ]
        self.train_stats_best_eq_mass = OrderedDict(
            {metrics_key: 0 for metrics_key in metrics_keys}
        )
        self.val_stats_best_eq_mass = OrderedDict(
            {metrics_key: 0 for metrics_key in metrics_keys}
        )
        self.current_train_stats = OrderedDict(
            {metrics_key: 0 for metrics_key in metrics_keys}
        )
        self.current_val_stats = OrderedDict(
            {metrics_key: 0 for metrics_key in metrics_keys}
        )
        self.best_accuracy = 0.0
        self.best_auc = 0.0

    def train(self):
        self.start_epoch = 0
        self._setup_network()
        self._setup_optimizer()

        if self.settings.resume_training == 1:
            (
                self.settings,
                self.start_epoch,
                self.net,
                self.optimizer,
                self.scheduler,
                self.train_stats_best_eq_mass,
                self.val_stats_best_eq_mass,
            ) = load_checkpoint(
                self.net,
                self.optimizer,
                self.scheduler,
                self.settings.model_name,
                self.settings.seed,
                self.settings.checkpoint_dir,
            )

        if self.settings.use_pretrained_model == 1:
            self.net = load_model(self.net, self.settings)

        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.net, log="all")

        for self.epoch in range(self.start_epoch, self.settings.num_epochs):
            self.settings.current_epoch = self.epoch
            print()
            print("---> Starting epoch number: ", self.epoch)
            self._train_one_epoch()
            self._validate_one_epoch()
            self._wandb_log()
            print(
                "---> Metrics of epoch {}:\n".format(self.epoch),
                "   - Train loss {:.3f}, Train accuracy {:.3f}, Train EM-ECE {:.3f}.\n".format(
                    self.train_loss, self.train_accuracy, self.train_eq_mass_ece
                ),
                "   - Val accuracy {:.3f}, Val EM-ECE {:12f}, Val AUC {:.12f}.".format(
                    self.val_accuracy, self.val_eq_mass_ece, self.val_auc
                ),
            )
            # Update best metrics and save checkpoints
            if self.epoch == self.start_epoch or (
                self.val_eq_mass_ece < self.val_stats_best_eq_mass["eq_mass_ece"]
            ):
                for key in self.val_stats_best_eq_mass:
                    self.val_stats_best_eq_mass[key] = self.current_val_stats[key]
                for key in self.train_stats_best_eq_mass:
                    self.train_stats_best_eq_mass[key] = self.current_train_stats[key]

                checkpoint_file = "{}/{}_{:02d}_best_ece.pth".format(
                    self.settings.checkpoint_dir,
                    self.settings.model_name,
                    self.settings.seed,
                )
                save_checkpoint(
                    self.epoch,
                    self.net,
                    self.optimizer,
                    self.scheduler,
                    self.train_stats_best_eq_mass,
                    self.val_stats_best_eq_mass,
                    self.settings,
                    self.settings.checkpoint_dir,
                    checkpoint_file,
                )

            if self.current_val_stats["accuracy"] > self.best_accuracy:
                print(
                    " /// New best validation accuracy {:.3f} at epoch {}. \\\ ".format(
                        self.val_accuracy, self.epoch
                    )
                )
                checkpoint_file = "{}/{}_{:02d}_best_acc.pth".format(
                    self.settings.checkpoint_dir,
                    self.settings.model_name,
                    self.settings.seed,
                )
                save_checkpoint(
                    self.epoch,
                    self.net,
                    self.optimizer,
                    self.scheduler,
                    self.train_stats_best_eq_mass,
                    self.val_stats_best_eq_mass,
                    self.settings,
                    self.settings.checkpoint_dir,
                    checkpoint_file,
                )
                self.best_accuracy = self.current_val_stats["accuracy"]

            if self.current_val_stats["auc"] > self.best_auc:
                print(
                    " -- New best validation AUC {:.3f} at epoch {}. -- ".format(
                        self.val_auc, self.epoch
                    )
                )
                checkpoint_file = "{}/{}_{:02d}_best_auc.pth".format(
                    self.settings.checkpoint_dir,
                    self.settings.model_name,
                    self.settings.seed,
                )

                save_checkpoint(
                    self.epoch,
                    self.net,
                    self.optimizer,
                    self.scheduler,
                    self.train_stats_best_eq_mass,
                    self.val_stats_best_eq_mass,
                    self.settings,
                    self.settings.checkpoint_dir,
                    checkpoint_file,
                )
                self.best_auc = self.current_val_stats["auc"]

        checkpoint_file = "{}/{}_{:02d}_last_epoch.pth".format(
            self.settings.checkpoint_dir,
            self.settings.model_name,
            self.settings.seed,
        )
        save_checkpoint(
            self.epoch,
            self.net,
            self.optimizer,
            self.scheduler,
            self.train_stats_best_eq_mass,
            self.val_stats_best_eq_mass,
            self.settings,
            self.settings.checkpoint_dir,
            checkpoint_file,
        )
        print(
            "Best accuracy {:.3f}, best AUC {:.3f}, best EM-ECE {:.3f}.".format(
                self.best_accuracy,
                self.best_auc,
                self.val_stats_best_eq_mass["eq_mass_ece"],
            )
        )
        self.settings.best_val_auc_runs.append(self.best_auc)
        self.settings.best_val_acc_runs.append(self.best_accuracy)
        self.settings.best_val_ece_runs.append(
            self.val_stats_best_eq_mass["eq_mass_ece"]
        )

        wandb.run.summary["val_best_acc"] = self.best_accuracy
        wandb.run.summary["val_best_auc"] = self.best_auc
        wandb.run.summary["val_best_ece"] = self.val_stats_best_eq_mass["eq_mass_ece"]

    def _train_one_epoch(self):
        """Do one epoch for train set."""
        self.net.train()
        self.train_loss = 0
        correct = 0
        total = 0
        self.train_eq_width_ece = 0
        self.train_eq_mass_ece = 0
        self.train_class_wise_ece = 0
        self.train_accuracy = 0
        self.train_primary = 0
        self.train_secondary = 0
        torch.autograd.set_detect_anomaly(True)

        for _, train_data in enumerate(self.train_loader, 0):
            self.optimizer.zero_grad()
            data, train_targets = train_data
            if "mnist" in self.settings.dataset:
                train_targets = torch.squeeze(train_targets, 1).long()
            data, train_targets = data.to(self.device), train_targets.to(self.device)
            train_outputs = self.net(data)

            # Calculate losses
            if "primary_loss_type" in self.settings:
                loss, primary, secondary = compute_loss(
                    self.settings, train_outputs, train_targets
                )

            else:
                loss = compute_loss(self.settings, train_outputs, train_targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2)
            self.train_loss += loss.item()
            if "primary_loss_type" in self.settings:
                self.train_primary += primary.item()
                self.train_secondary += secondary.item()

            self.optimizer.step()

            _, predictions = torch.max(train_outputs, 1)

            # Calculate metrics: accuracy, EM-ECE, EW-ECE, CW-ECE
            total += train_targets.size(0)
            correct += predictions.eq(train_targets).cpu().sum()
            confidences = F.softmax(train_outputs, dim=1).detach().cpu().numpy()
            (
                self.train_eq_mass_ece,
                self.train_eq_width_ece,
                self.train_class_wise_ece,
            ) = calculate_ECE_metrics(
                confidences,
                train_targets.detach().cpu().numpy(),
                self.train_eq_mass_ece,
                self.train_eq_width_ece,
                self.train_class_wise_ece,
            )
        # Track AUCOC over last train batch of each epoch
        self.train_aucoc_last_batch = (
            compute_auc(
                (F.softmax(train_outputs, dim=1)).detach().cpu().numpy(),
                train_targets.detach().cpu().numpy(),
            )
            * 100.0
        )
        self.scheduler.step()

        self.train_accuracy = 100.0 * correct / total
        self.train_eq_mass_ece /= len(self.train_loader)
        self.train_eq_width_ece /= len(self.train_loader)
        self.train_class_wise_ece /= len(self.train_loader)

        # Update the current stats
        self.current_train_stats["eq_mass_ece"] = self.train_eq_mass_ece
        self.current_train_stats["eq_width_ece"] = self.train_eq_width_ece
        self.current_train_stats["class_wise_ece"] = self.train_class_wise_ece
        self.current_train_stats["accuracy"] = self.train_accuracy

    def _validate_one_epoch(self):
        self.net.eval()
        if "mnist" in self.settings.dataset or "cifar100_LT" in self.settings.dataset:
            length_dataset = len(self.val_loader.dataset)
        else:
            length_dataset = int(
                np.floor(self.settings.val_set_perc * len(self.train_loader.dataset))
            )

        labels_np = np.zeros(length_dataset)
        predictions_np = np.zeros(length_dataset)
        confidences_np = np.zeros((length_dataset, self.settings.num_classes))
        correct = 0
        total = 0
        self.val_loss = 0
        self.val_eq_width_ece = 0
        self.val_eq_mass_ece = 0
        self.val_class_wise_ece = 0
        self.val_accuracy = 0
        self.val_primary = 0
        self.val_secondary = 0

        """Do one epoch for val set."""
        with torch.no_grad():
            for batch_idx, val_data in enumerate(self.val_loader, 0):
                data, val_targets = val_data
                if "mnist" in self.settings.dataset:
                    val_targets = torch.squeeze(val_targets, 1).long()
                data, val_targets = data.to(self.device), val_targets.to(self.device)
                val_outputs = self.net(data)

                # Calculate losses
                if "primary_loss_type" in self.settings:
                    loss, primary, secondary = compute_loss(
                        self.settings, val_outputs, val_targets
                    )
                else:
                    loss = len(val_data) * compute_loss(
                        self.settings, val_outputs, val_targets
                    )
                self.val_loss += loss.item()
                if "primary_loss_type" in self.settings:
                    self.val_primary += primary.item()
                    self.val_secondary += secondary.item()
                _, predictions = torch.max(val_outputs, 1)

                # Create arrays for the whole dataset
                correct += predictions.eq(val_targets).cpu().sum()
                confidences = F.softmax(val_outputs, dim=1).detach().cpu().numpy()
                samples_batch = val_targets.size(0)
                total += samples_batch
                offset = batch_idx * self.val_loader.batch_size
                labels_np[offset : offset + samples_batch] = (
                    val_targets.detach().cpu().numpy()
                )
                predictions_np[offset : offset + samples_batch] = (
                    predictions.detach().cpu().numpy()
                )
                confidences_np[offset : offset + samples_batch, :] = confidences

        # Calculate metrics: accuracy, EM-ECE, EW-ECE, CW-ECE
        self.val_accuracy = float((100.0 * correct / total).detach())
        (
            self.val_eq_mass_ece,
            self.val_eq_width_ece,
            self.val_class_wise_ece,
        ) = calculate_ECE_metrics(
            confidences_np,
            labels_np,
            self.val_eq_mass_ece,
            self.val_eq_width_ece,
            self.val_class_wise_ece,
        )
        # Compute val AUCOC
        self.val_auc = compute_auc(confidences_np, labels_np) * 100.0
        # Check minimum and median of delta rn distribution
        r = np.sort(np.amax(confidences, axis=1))
        delta_r = r[1:] - r[:-1]
        self.min_val_delta_r = np.amin(delta_r)
        self.median_val_delta_r = np.median(delta_r)

        # Update the current stats
        self.current_val_stats["eq_mass_ece"] = self.val_eq_mass_ece
        self.current_val_stats["eq_width_ece"] = self.val_eq_width_ece
        self.current_val_stats["class_wise_ece"] = self.val_class_wise_ece
        self.current_val_stats["accuracy"] = self.val_accuracy
        self.current_val_stats["auc"] = self.val_auc

    def _setup_network(self):
        if "wide_resnet" in self.settings.net_type:
            self.net = wide_resnet_cifar(
                depth=self.settings.depth,
                width=self.settings.widen_factor,
                num_classes=self.settings.num_classes,
            )
        elif self.settings.net_type == "resnet50":
            self.net = resnet50(self.settings.num_classes)
        elif self.settings.net_type == "resnet18":
            self.net = resnet18(self.settings.num_classes)
        elif self.settings.net_type == "resnet34":
            self.net = resnet34(self.settings.num_classes)
        else:
            warnings.warn("Model is not listed.")

        self.net = nn.DataParallel(self.net)
        self.net.to(self.settings.device)

    def _setup_optimizer(self):
        if self.settings.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.settings.base_lr,
            )
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.settings.milestones,
                gamma=self.settings.gamma,
            )
        elif self.settings.optimizer == "Adadelta":
            self.optimizer = optim.Adadelta(
                self.net.parameters(), lr=self.settings.base_lr
            )
            gamma = self.settings.gamma
            milestones = self.settings.milestones
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma,
            )
        else:
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self.settings.base_lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
            )
            milestones = self.settings.milestones

            if self.settings.use_scheduler == 0:
                milestones = [1000]

            gamma = self.settings.gamma
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma,
            )

    def _wandb_log(self):
        if "primary_loss_type" in self.settings:
            wandb.log(
                {
                    "train_loss": self.train_loss,
                    "train_primary": self.train_primary,
                    "train_secondary": self.train_secondary,
                    "val_loss": self.val_loss,
                    "val_primary": self.val_primary,
                    "val_secondary": self.val_secondary,
                    "train_eq_width_ece": self.train_eq_width_ece,
                    "train_eq_mass_ece": self.train_eq_mass_ece,
                    "train_class_wise_ece": self.train_class_wise_ece,
                    "train_accuracy": self.train_accuracy,
                    "val_eq_width_ece": self.val_eq_width_ece,
                    "val_eq_mass_ece": self.val_eq_mass_ece,
                    "val_class_wise_ece": self.val_class_wise_ece,
                    "val_accuracy": self.val_accuracy,
                    "val_auc": self.val_auc,
                    "train_auc_last_batch": self.train_aucoc_last_batch,
                    "median_val_delta_r": self.median_val_delta_r,
                    "min_val_delta_r": self.min_val_delta_r,
                }
            )
        else:
            wandb.log(
                {
                    "train_loss": self.train_loss,
                    "val_loss": self.val_loss,
                    "train_eq_width_ece": self.train_eq_width_ece,
                    "train_eq_mass_ece": self.train_eq_mass_ece,
                    "train_class_wise_ece": self.train_class_wise_ece,
                    "train_accuracy": self.train_accuracy,
                    "val_eq_width_ece": self.val_eq_width_ece,
                    "val_eq_mass_ece": self.val_eq_mass_ece,
                    "val_class_wise_ece": self.val_class_wise_ece,
                    "val_accuracy": self.val_accuracy,
                    "val_auc": self.val_auc,
                    "train_auc_last_batch": self.train_aucoc_last_batch,
                    "median_val_delta_r": self.median_val_delta_r,
                    "min_val_delta_r": self.min_val_delta_r,
                }
            )
