from typing import Optional
from torch.functional import split
from torch.utils import data
from torch.utils.data import random_split, DataLoader
import tonic
from torchvision import transforms
import os
import numpy as np

from cifar10dvs import CIFAR10DVS


def get_dataloaders(batch_size: int, n_bins: int, dataset: str = "nmnist", data_dir: str = "data", binarize: bool = False):
    # create the directory if not exist
    os.makedirs(data_dir, exist_ok=True)

    # use a to_frame transform
    sensor_size, num_classes = _get_dataset_info(dataset)
    frame_transform = tonic.transforms.ToFrame(
        sensor_size=sensor_size, n_time_bins=n_bins)

    train_transfs = [
        tonic.transforms.RandomFlipLR(sensor_size),
        frame_transform
    ]
    val_transfs = [
        frame_transform
    ]

    if binarize:
        train_transfs.append(transforms.Lambda(
            lambda x: (x > 0).astype(np.float32)))
        val_transfs.append(transforms.Lambda(
            lambda x: (x > 0).astype(np.float32)))

    train_transform = transforms.Compose(train_transfs)
    val_transform = transforms.Compose(val_transfs)

    train_set, val_set = _get_datasets(
        dataset, train_transform, val_transform, data_dir)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=0)

    return train_loader, val_loader, num_classes


def _get_dataset_info(dataset):
    if dataset == "n-mnist":
        return tonic.datasets.NMNIST.sensor_size, len(tonic.datasets.NMNIST.classes)
    elif dataset == "cifar10-dvs":
        return CIFAR10DVS.sensor_size, 10
    elif dataset == "dvsgesture":
        return tonic.datasets.DVSGesture.sensor_size, len(tonic.datasets.DVSGesture.classes)
    elif dataset == "asl-dvs":
        return tonic.datasets.ASLDVS.sensor_size, len(tonic.datasets.ASLDVS.classes)


def _get_datasets(dataset, train_transform, val_transform, data_dir):
    if dataset == "n-mnist":
        train_set = tonic.datasets.NMNIST(
            save_to=data_dir, transform=train_transform, target_transform=None, train=True)
        val_set = tonic.datasets.NMNIST(
            save_to=data_dir, transform=val_transform, target_transform=None, train=False)

    elif dataset == "cifar10-dvs":
        dataset_train = CIFAR10DVS(
            save_to=data_dir, transform=train_transform, target_transform=None)
        dataset_val = CIFAR10DVS(
            save_to=data_dir, transform=val_transform, target_transform=None)
        print(len(dataset_train))
        train_set, _ = random_split(
            dataset_train, lengths=[8330, 10000 - 8330])
        _, val_set = random_split(dataset_val, lengths=[8330, 10000 - 8330])

    elif dataset == "dvsgesture":
        train_set = tonic.datasets.DVSGesture(
            save_to=data_dir, transform=train_transform, target_transform=None, train=True)
        val_set = tonic.datasets.DVSGesture(
            save_to=data_dir, transform=val_transform, target_transform=None, train=False)

    return train_set, val_set
