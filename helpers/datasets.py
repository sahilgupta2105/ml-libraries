import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_mnist(dataset_path, batch_size, **kwargs):
    """Loads MNIST dataset and returns the train and test data loaders."""
    mnist_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=True, download=True
    )
    test_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=False, download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader
