"""Train script for ATDA."""

import torch
from torch import nn

from misc import config as cfg
from misc.utils import (get_model_params, get_optimizer, make_variable,
                        save_model)


def calc_similiar_penalty(F_1, F_2):
    """Calculate similiar penalty |W_1^T W_2|."""
    F_1_params = get_model_params(F_1, "classifier.8.weight")
    F_2_params = get_model_params(F_2, "classifier.8.weight")
    similiar_penalty = torch.sum(
        torch.abs(torch.mul(F_1_params.transpose(0, 1), F_2_params)))
    return similiar_penalty


def pre_train(F, F_1, F_2, F_t, source_data):
    """Pre-train models on source domain dataset."""
    # set train state for Dropout and BN layers
    F.train()
    F_1.train()
    F_2.train()
    F_t.train()

    # set criterion for classifier and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_F = get_optimizer(F, "Adam")
    optimizer_F_1 = get_optimizer(F_1, "Adam")
    optimizer_F_2 = get_optimizer(F_2, "Adam")
    optimizer_F_t = get_optimizer(F_t, "Adam")

    for epoch in range(cfg.num_epochs_pre):
        for step, (images, labels) in enumerate(source_data):
            images = make_variable(images)
            labels = make_variable(labels)

            out_F_1 = F_1(F(images))
            out_F_2 = F_2(F(images))
            out_F_t = F_t(F(images))

            loss_F_1 = criterion(out_F_1, labels)
            loss_F_2 = criterion(out_F_2, labels)
            loss_F_t = criterion(out_F_t, labels)
            loss_similiar = calc_similiar_penalty(F_1, F_2)


def labelling(F, F_1, F_2, targte_data):
    """Genrate pseudo labels for target domain dataset."""
    pass


def domain_adapt(F, F_1, F_2, F_t, labelled_data, target_data_labelled):
    """Perform Doamin Adaptation between source and target domains."""
    pass
