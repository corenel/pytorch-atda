"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

from misc import config as cfg


def get_svhn(train):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=cfg.dataset_mean,
                                          std=cfg.dataset_std)])

    # dataset and data loader
    svhn_dataset = datasets.SVHN(root=cfg.data_root,
                                 split='train' if train else 'test',
                                 transform=pre_process,
                                 download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=cfg.batch_size,
        shuffle=True)

    return svhn_data_loader
