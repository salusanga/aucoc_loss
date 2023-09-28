import os
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader


def get_dataloaders_svhn(settings):
    """
    Create a DataLoader object for testing on the SVHN dataset for Out-of-Distribution (OOD) experiments.

    Parameters:
        settings (object): An object containing various settings and configurations.

    Returns:
        DataLoader: A DataLoader for testing on the SVHN dataset for OOD experiments.

    This function prepares a DataLoader object for testing on the SVHN (Street View House Numbers) dataset
    for Out-of-Distribution (OOD) experiments. It uses the specified dataset paths, batch size, and data
    normalization defined in 'settings' to create the DataLoader object.

    The SVHN dataset is used as an OOD dataset in experiments, and this function is specifically designed
    for loading the test split of SVHN.
    """
    # Prepare dataset
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset_location = os.path.join(settings.datasets_path, settings.dataset_ood)

    test_set = torchvision.datasets.SVHN(
        root=dataset_location,
        split="test",
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader
