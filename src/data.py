import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader


def get_transform(train=True):
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
