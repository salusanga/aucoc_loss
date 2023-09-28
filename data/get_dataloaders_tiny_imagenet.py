import os
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

"""
Create train, val, test iterators for Tiny ImageNet.
Train set size: 100000
Val set size: 10000
Test set size: 10000
Number of classes: 200
Code readapted from https://github.com/torrvision/focal_calibration
"""

import os
import torch
import numpy as numpy
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import glob
from PIL import Image

EXTENSION = "JPEG"
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = "wnids.txt"
TEST_ANNOTATION_FILE = "val_annotations.txt"


class TinyImageNet(Dataset):
    """
    Tiny ImageNet dataset available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Parameters:
        - settings: parser with all the required information.
        - split: string indicating which split to return as a data set. Valid options: [`train`, `test`, `val`]
        - transform: A (series) of valid transformation(s)
        - target_transform: A (series) of valid target transformation(s)
        - in_memory: Set to True if there is enough memory (about 5GB) and want to minimize disk IO overhead.

    Output: train, val, test dataset splits.
    """

    def __init__(
        self, settings, split, transform=None, target_transform=None, in_memory=False
    ):
        self.root = os.path.join(
            settings.datasets_path, "tiny-imagenet/tiny-imagenet-200"
        )
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        if self.split == "test":
            self.split_dir = os.path.join(self.root, "val")
        else:
            self.split_dir = os.path.join(self.root, "train")
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self.split_dir, "**", "*.%s" % EXTENSION), recursive=True
            )
        )
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == "train" or self.split == "val":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, EXTENSION)] = i
        elif self.split == "test":
            with open(os.path.join(self.split_dir, TEST_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

            # file_name = file_path.split('/')[-1]
        return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = self.split
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        if img.mode == "L":
            img = img.convert("RGB")
        return self.transform(img) if self.transform else img


def get_dataloaders_tiny_imagenet(settings):
    """
    Utility function to load the Tiny-ImageNet dataset.

    Parameters:
        - settings: parser with all the required information.

    Returns:
        - train_loader: Torch training set iterator.
        - val_loader: Torch validation set iterator.
        - test_loader: Torch test set iterator.
    """

    # Img transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    val_test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Create sets
    train_set = TinyImageNet(
        settings=settings, split="train", transform=train_transform, in_memory=True
    )
    val_set = TinyImageNet(
        settings=settings, split="val", transform=val_test_transform, in_memory=True
    )
    test_set = TinyImageNet(
        settings=settings, split="test", transform=val_test_transform, in_memory=True
    )

    # To split train and val set
    num_tot_train = len(train_set)
    indices = list(range(num_tot_train))
    np.random.shuffle(indices)
    split_data_train_val = int(np.floor(settings.val_set_perc * num_tot_train))
    train_idx, val_idx = indices[split_data_train_val:], indices[:split_data_train_val]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=settings.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=settings.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=settings.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
