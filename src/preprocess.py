"""
preprocess.py – data-loading helpers
===================================
Uses torchvision.FakeData by default so that the entire pipeline runs
in <30 seconds on a CPU-only CI runner.  Swap FakeData for CIFAR-10 /
ImageNet simply by editing the get_datasets() routine.
"""
from __future__ import annotations
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_datasets(cfg):
    tfm = transforms.Compose([
        transforms.Resize(cfg.data.img_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.FakeData(size=cfg.data.dataset_size,
                                image_size=(3, cfg.data.img_size, cfg.data.img_size),
                                num_classes=cfg.data.num_classes,
                                transform=tfm)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


def get_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = get_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size)
    return train_loader, val_loader
