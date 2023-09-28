import os
import PIL
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import medmnist
from medmnist import INFO


def get_dataloaders_medmnist(settings):
    """
    Create DataLoader objects for training, validation, and testing on the MedMNIST datasets.

    Parameters:
        settings (object): An object containing various settings and configurations.

    Returns:
        tuple: A tuple containing three DataLoader objects (train_loader, val_loader, test_loader).

    This function prepares DataLoader objects for training, validation, and testing on the MedMNIST dataset.
    It uses the specified dataset paths, dataset name, batch size, and data transformations defined in 'settings'
    to create the DataLoader objects.

    This function is designed to work with various datasets within the MedMNIST collection.

    """
    info = INFO[settings.dataset]  # Load info of the chosen MedMNIST dataset
    as_rgb = 1  # Converted to a 3-channels input
    n_channels = 3 if as_rgb else info["n_channels"]
    settings.num_classes = len(info["label"])
    resize_medmnist = 0
    # Prepare dataset
    if resize_medmnist:
        data_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

    dataset_location = os.path.join(settings.datasets_path, settings.dataset)

    DataClass = getattr(medmnist, info["python_class"])

    train_set = DataClass(
        split="train",
        transform=data_transform,
        root=dataset_location,
        download=False,
        as_rgb=as_rgb,
    )
    val_set = DataClass(
        split="val",
        transform=data_transform,
        root=dataset_location,
        download=False,
        as_rgb=as_rgb,
    )
    test_set = DataClass(
        split="test",
        transform=data_transform,
        root=dataset_location,
        download=False,
        as_rgb=as_rgb,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=settings.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=settings.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
