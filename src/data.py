import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader


def get_transform(train=True):
    """Build the image preprocessing pipeline for CIFAR-10 examples.

    The pipeline resizes CIFAR-10 images to the 224x224 input size expected by
    the pretrained vision transformer backbones, converts them to tensors, and
    normalizes them with ImageNet statistics. Training transforms additionally
    include random horizontal flipping for light augmentation.

    Args:
        train: Whether to include training-time augmentation.

    Returns:
        A torchvision ``Compose`` transform suitable for CIFAR-10 images.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = [
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ]
    if train:
        base.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)


def get_dataloaders(batch_size=32, num_workers=4):
    """Create train and test data loaders for CIFAR-10.

    CIFAR-10 is downloaded into ``./data`` when it is not already present. The
    train loader shuffles examples, while the test loader preserves deterministic
    dataset order for evaluation and prediction exports.

    Args:
        batch_size: Number of images yielded by each loader batch.
        num_workers: Number of worker processes used by each ``DataLoader``.

    Returns:
        A tuple containing ``(train_loader, test_loader)``.
    """
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=get_transform(train=True)
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_test_loader(batch_size=32, num_workers=4):
    """Create the CIFAR-10 test loader and dataset.

    This helper is intended for evaluation notebooks that need both batched test
    data and direct indexed access to the transformed dataset, such as attention
    map visualizations.

    Args:
        batch_size: Number of images yielded by each test batch.
        num_workers: Number of worker processes used by the ``DataLoader``.

    Returns:
        A tuple containing ``(test_loader, test_dataset)``.
    """
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=get_transform(train=False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, test_dataset
