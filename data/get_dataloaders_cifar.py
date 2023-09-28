import os
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import warnings


def get_dataloaders_cifar(settings):
    """
    Create DataLoader objects for training, validation, and testing on CIFAR-10 or CIFAR-100 dataset.

    Parameters:
        settings (object): An object containing various settings and configurations.

    Returns:
        tuple: A tuple containing three DataLoader objects (train_loader, val_loader, test_loader).

    This function prepares DataLoader objects for training, validation, and testing on either
    the CIFAR-10 or CIFAR-100 dataset, based on the value of 'settings.dataset'.

    The training dataset is transformed with random crops and horizontal flips for data augmentation,
    while the validation and test datasets are only normalized. The train-validation split is
    configured to allocate 'settings.val_set_perc' percentage of the training data to the validation set.
    """

    # Prepare dataset
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_location = os.path.join(settings.datasets_path, settings.dataset)

    if settings.dataset == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=dataset_location,
            train=True,
            download=False,
            transform=transform_train,
        )
        val_set = torchvision.datasets.CIFAR10(
            root=dataset_location,
            train=True,
            download=False,
            transform=transform_test,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=dataset_location,
            train=False,
            download=False,
            transform=transform_test,
        )

    elif settings.dataset == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=dataset_location,
            train=True,
            download=False,
            transform=transform_train,
        )
        val_set = torchvision.datasets.CIFAR100(
            root=dataset_location,
            train=True,
            download=False,
            transform=transform_test,
        )
        test_set = torchvision.datasets.CIFAR100(
            root=dataset_location,
            train=False,
            download=False,
            transform=transform_test,
        )

    else:
        warnings.warn("Dataset is not listed")

    # Create train-val split. Validation set is a certain % of entire train set
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(settings.val_set_perc * num_train))

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=settings.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
