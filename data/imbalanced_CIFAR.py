"""
Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import argparse
import numpy as np
from easydict import EasyDict
import yaml
from yaml.loader import FullLoader
from torch.utils.data import DataLoader


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(
        self,
        train,
        split,
        root,
        imbalance_ratio,
        indeces=None,
        imb_type="exp",
        download=True,
    ):
        super(IMBALANCECIFAR10, self).__init__(
            root, split, transform=None, target_transform=None
        )
        self.split = split
        if self.split == "train" or self.split == "val":
            self.data = self.data[indeces, ...]
            indeces = [int(x) for x in indeces]
            self.targets = [self.targets[i] for i in indeces]
            # print(len(self.targets))

        if self.split == "train":
            img_num_list = self.get_img_num_per_cls(
                self.cls_num, imb_type, imbalance_ratio
            )
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        self.labels = self.targets

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * (selec_idx).size
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        # print(self.data.shape, len(self.targets))

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({"category_id": int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """

    cls_num = 100
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


def get_dataloaders_cifar_LT(settings):
    """
    This function prepares DataLoader objects for the imbalanced CIFAR-100 dataset,
    simulating a Long-Tailed (LT) distribution of class samples in the training set.
    The imbalance ratio specified in 'settings.imbalance_ratio' controls the class imbalance.
    The training and validation sets are imbalanced, while the test set remains balanced.

    Parameters:
        - settings: parser for all parameters
        - settings.batch_size: batch size
        - settings.imbalance_ratio: specifies the N_min/N_max class samples to control
            the imbalance in the training set

    Output:
        - train_loader: DataLoader for the imbalanced training set
        - val_loader: DataLoader for the validation set
        - test_loader: DataLoader for the balanced test set (to assess the effects
            of the shift on a balanced test set).

    """

    dataset_location = os.path.join(settings.datasets_path, "cifar100")
    num_train = 50000
    indices = list(range(num_train))
    np.random.shuffle(indices)
    splits = int(np.floor(settings.val_set_perc * num_train))

    train_idx, val_idx = indices[splits:], indices[:splits]

    train_set = IMBALANCECIFAR100(
        root=dataset_location,
        split="train",
        indeces=train_idx,
        imbalance_ratio=settings.imbalance_ratio,
        train=True,
        download=True,
    )
    val_set = IMBALANCECIFAR100(
        root=dataset_location,
        split="val",
        indeces=val_idx,
        imbalance_ratio=settings.imbalance_ratio,
        train=True,
        download=True,
    )
    test_set = IMBALANCECIFAR100(
        root=dataset_location,
        train=False,
        split="test",
        imbalance_ratio=settings.imbalance_ratio,
        download=True,
    )
    
    # Define loaders
    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=settings.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=settings.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)
    # Set parameters
    parser = argparse.ArgumentParser(description="Run train and/or test.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="calibration-nn-classification",
        help="Whether run training.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "--train_mode",
        type=int,
        default=1,
        help="Whether run training.",  # CHANGE BACK TO 1
    )
    parser.add_argument(
        "--eval_mode", type=int, default=1, help="Whether run evaluation."
    )
    parser.add_argument("--prova", type=int, default=0, help="Whether run is trial.")
    parser.add_argument(
        "--paths_config_file",
        default="paths",
        type=str,
        help="Settings for paths.",
    )
    parser.add_argument(
        "--base_config_file", type=str, help="Settings for dataset and model."
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=bool,
        default=True,
        help="Set cudnn benchmark on (1) or off (0) (default is on).",
    )

    # settings = vars(parser.parse_args())
    # settings = read_config_file("configs", settings["paths_config_file"], settings)
    # settings = read_config_file(
    #     settings["base_config_path"], settings["base_config_file"], settings
    # )
    # settings = EasyDict(settings)
    # train_loader, val_loader, test_loader = get_dataloaders_cifar_LT(settings)
