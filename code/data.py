"""
data.py

CIFAR-10 / CIFAR-100 data loading for the stochastic depth re-implementation.

We use torchvision's transform pipeline for augmentation (RandomHorizontalFlip
+ RandomCrop with padding), which is functionally equivalent to the horizontal
flip and ±4-pixel translation used in the paper but expressed via the standard
PyTorch API rather than manual pixel-offset loops.

The train/val split (45k / 5k) is created with a fixed Generator so the split
is deterministic and reproducible across runs.
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as T


# Per-channel statistics computed over the CIFAR-10 training set.
# For CIFAR-100 we use the same values; differences are negligible.
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)

_VAL_SIZE = 5_000   # held-out portion of train set used for validation


def _base_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])


def _train_transform() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="constant"),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


def build_loaders(
    dataset_name: str = "cifar10",
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).

    The validation set is a fixed 5,000-sample subset of the training data.
    It uses the base (no-augmentation) transform so evaluation is deterministic.
    """
    assert dataset_name in ("cifar10", "cifar100"), \
        f"Unsupported dataset: {dataset_name}"
    cls = torchvision.datasets.CIFAR10 if dataset_name == "cifar10" \
        else torchvision.datasets.CIFAR100

    # Download raw training data (we'll apply transforms per-split below)
    raw_train = cls(root=root, train=True,  download=True, transform=None)
    raw_test  = cls(root=root, train=False, download=True, transform=_base_transform())

    # Deterministic split: 45k train / 5k val
    rng = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(len(raw_train)),
        [len(raw_train) - _VAL_SIZE, _VAL_SIZE],
        generator=rng,
    )

    # Wrap subsets with their respective transforms
    train_set = _TransformSubset(raw_train, list(train_idx), _train_transform())
    val_set   = _TransformSubset(raw_train, list(val_idx),   _base_transform())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        raw_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class _TransformSubset(torch.utils.data.Dataset):
    """Dataset wrapper that applies a transform to an index subset of another dataset."""

    def __init__(self, dataset, indices: list[int], transform: T.Compose):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label
