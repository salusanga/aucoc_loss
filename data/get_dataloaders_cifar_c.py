import os
import numpy as np
import torchvision
import torch
from torchvision import transforms


def get_dataloaders_cifar_c(settings):
    """
    Create a DataLoader for testing on CIFAR-100 corrupted dataset.

    Parameters:
        settings (object): An object containing various settings and configurations.

    Returns:
        DataLoader: A DataLoader for testing on the CIFAR100-C dataset.

    This function prepares a DataLoader for testing on a corrupted version of the CIFAR-100 dataset.
    It loads the CIFAR-100 dataset, with the specified corruption type to it, and returns a DataLoader
    configured with the given batch size and other settings.

    The CIFAR-100 dataset is assumed to be located at the path specified in 'settings.datasets_path'
    and the corruption type is specified in 'settings.corruption_type'.
    """

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_location_id = os.path.join(settings.datasets_path, settings.dataset)

    print("Test on " + settings.corruption_type)
    test_data = torchvision.datasets.CIFAR100(
        root=dataset_location_id, train=False, transform=test_transform, download=False
    )
    test_data.data = np.load(
        dataset_location_id + "/CIFAR-100-C/%s.npy" % settings.corruption_type
    )
    test_data.targets = torch.LongTensor(
        np.load(dataset_location_id + "/CIFAR-100-C/labels.npy")
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader
