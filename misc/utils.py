"""Helpful functions for ATDA."""

import os
import random
from math import log

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable

from datasets import get_mnist, get_mnist_m, get_svhn, get_usps
from misc import config as cfg


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    restore_model(net, restore)

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(cfg.model_root):
        os.makedirs(cfg.model_root)
    torch.save(net.state_dict(),
               os.path.join(cfg.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(cfg.model_root,
                                                             filename)))


def restore_model(net, restore):
    """Restore network from saved model."""
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))


def get_optimizer(net, name="Adam"):
    """Get optimizer by name."""
    if name == "Adam":
        return optim.Adam(net.parameters(),
                          lr=cfg.learning_rate,
                          betas=(cfg.beta1, cfg.beta2))


def get_data_loader(name, train=True, get_dataset=False):
    """Get data loader by name."""
    if name == "MNIST":
        return get_mnist(train, get_dataset)
    elif name == "MNIST-M":
        return get_mnist_m(train, get_dataset)
    elif name == "SVHN":
        return get_svhn(train, get_dataset)
    elif name == "USPS":
        return get_usps(train, get_dataset)


def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels in data_loader:
            yield (images, labels)


def get_model_params(net, name):
    """Get parameters of models by name."""
    for n, p in net.named_parameters():
        if n == name:
            return p


def calc_similiar_penalty(F_1, F_2):
    """Calculate similiar penalty |W_1^T W_2|."""
    F_1_params = get_model_params(F_1, "classifier.8.weight")
    F_2_params = get_model_params(F_2, "classifier.8.weight")
    similiar_penalty = torch.sum(
        torch.abs(torch.mm(F_1_params.transpose(0, 1), F_2_params)))
    return similiar_penalty


def get_whole_dataset(dataset):
    """Get all images and labels of dataset."""
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=len(dataset))
    for images, labels in data_loader:
        return images, labels


def sample_candidatas(dataset, candidates_num, shuffle=True):
    """Sample images and labels from dataset."""
    # get data and labels
    data, labels = get_whole_dataset(dataset)
    # get indices
    indices = torch.arange(0, len(data))
    if shuffle:
        indices = torch.randperm(len(data))
    # slice indices
    candidates_num = min(len(data), candidates_num)
    excerpt = indices.narrow(0, 0, candidates_num)
    # select items by indices
    images = data.index_select(0, excerpt)
    true_labels = labels.index_select(0, excerpt)
    return images, true_labels


def get_minibatch_iterator(images, labels, batchsize, shuffle=False):
    """Get minibatch iterator with given images and labels."""
    assert len(images) == len(labels), \
        "Number of images and labels must be equal to make minibatches!"

    if shuffle:
        indices = torch.randperm(len(images))

    for start_idx in range(0, len(images), batchsize):
        end_idx = start_idx + batchsize
        if end_idx > len(images):
            end_idx = start_idx + (len(images) % batchsize)

        if shuffle:
            excerpt = indices.narrow(0, start_idx, end_idx)
        else:
            excerpt = slice(start_idx, end_idx)

        yield images.index_select(0, excerpt), labels.index_select(0, excerpt)


def make_labels(labels_dense, num_classes):
    """Convert dense labels into one-hot labels."""
    labels_one_hot = torch.zeros((labels_dense.size(0), num_classes))
    labels_one_hot.scatter_(1, labels_dense, 1)
    return labels_one_hot


def guess_pseudo_labels(images, labels, out_1, out_2, threshold=0.9):
    """Guess labels of target dataset by the two outputs."""
    # get prediction
    _, pred_idx_1 = torch.max(out_1, 1)
    _, pred_idx_2 = torch.max(out_2, 1)
    # find prediction who are the same in two outputs
    equal_idx = torch.nonzero(torch.eq(pred_idx_1, pred_idx_2)).squeeze()
    out_1 = out_1[equal_idx, :]
    out_2 = out_2[equal_idx, :]
    images = images[equal_idx, :]
    labels = labels[equal_idx]
    # filter indices by threshold
    # note that we use log(threshold) since the output is LogSoftmax
    pred_1, _ = torch.max(out_1, 1)
    pred_2, _ = torch.max(out_2, 1)
    max_pred, _ = torch.max(torch.stack([pred_1, pred_2], 1), 1)
    filtered_idx = torch.nonzero(max_pred > log(threshold)).squeeze()
    # get images, pseudo labels and true labels by indices
    _, pred_idx = torch.max(out_1[filtered_idx, :], 1)
    # no need to convert dense labels into one-hot labels
    # pseudo_labels = make_labels(pred_idx, cfg.num_classes)
    pseudo_labels = pred_idx

    return (images[filtered_idx, :],
            pseudo_labels,
            labels[filtered_idx])
