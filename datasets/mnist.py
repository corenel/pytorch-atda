"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms

from misc import config as cfg


def get_mnist(train, get_dataset=False, batch_size=cfg.batch_size):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=cfg.dataset_mean,
                                          std=cfg.dataset_std)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=cfg.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    if get_dataset:
        return mnist_dataset
    else:
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=batch_size,
            shuffle=True)
        return mnist_data_loader
