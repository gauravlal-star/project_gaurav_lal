"""
dataset.py  —  Data loading and augmentation for Smart Trash Sorter

Exports:
  WasteImageDataset  (TheDataset in interface.py)
  get_dataloader     (the_dataloader in interface.py)
"""

import os
import copy
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms():
    """Return (train_transform, val_transform) using sizes from config."""
    train_tf = transforms.Compose([
        transforms.Resize((config.resize_y, config.resize_x)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config.resize_y, config.resize_x)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])
    return train_tf, val_tf


# ── Dataset class (alias for ImageFolder) ────────────────────────────────────
class WasteImageDataset(datasets.ImageFolder):
    """
    ImageFolder sub-class that reads from config.data_dir.
    The directory must have one sub-folder per class:
        data/cardboard/  data/glass/  data/metal/
        data/paper/      data/plastic/ data/trash/
    """
    def __init__(self, root: str = None, transform=None):
        root = root or config.data_dir
        super().__init__(root=root, transform=transform)


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloader(data_dir: str = None, batch_size: int = None,
                   split: str = "train"):
    data_dir   = data_dir   or config.data_dir
    batch_size = batch_size or config.batch_size

    train_tf, val_tf = get_transforms()
    # Initialize without transforms first
    full_dataset = WasteImageDataset(root=data_dir, transform=None)

    n       = len(full_dataset)
    n_train = int(n * config.train_split)
    n_val   = int(n * config.val_split)
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # FIX: Copy the dataset so they don't share the same transform reference
    train_set.dataset = copy.copy(full_dataset)
    train_set.dataset.transform = train_tf

    val_set.dataset = copy.copy(full_dataset)
    val_set.dataset.transform   = val_tf

    test_set.dataset = copy.copy(full_dataset)
    test_set.dataset.transform  = val_tf

    split_map   = {"train": train_set, "val": val_set, "test": test_set}
    shuffle_map = {"train": True,      "val": False,   "test": False}

    chosen = split_map[split]
    loader = DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=shuffle_map[split],
        num_workers=2,
        pin_memory=True,
    )
    print(f"[dataset] split={split}  size={len(chosen)}  "
          f"classes={full_dataset.classes}")
    return loader


# ── Convenience: return all three loaders at once ─────────────────────────────
def load_all_splits(data_dir: str = None, batch_size: int = None):
    """Return (train_loader, val_loader, test_loader)."""
    data_dir   = data_dir   or config.data_dir
    batch_size = batch_size or config.batch_size

    train_tf, val_tf = get_transforms()
    full_dataset = WasteImageDataset(root=data_dir, transform=None)

    n       = len(full_dataset)
    n_train = int(n * config.train_split)
    n_val   = int(n * config.val_split)
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # FIX: Copy the dataset here as well
    train_set.dataset = copy.copy(full_dataset)
    train_set.dataset.transform = train_tf
    
    val_set.dataset = copy.copy(full_dataset)
    val_set.dataset.transform   = val_tf
    
    test_set.dataset = copy.copy(full_dataset)
    test_set.dataset.transform  = val_tf

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True)

    print(f"Dataset → Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return _loader(train_set, True), _loader(val_set, False), _loader(test_set, False)
