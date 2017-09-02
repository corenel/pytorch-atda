"""Train script for ATDA."""

import torch
from torch import nn

from misc import config as cfg
from misc.utils import (calc_similiar_penalty, get_minibatch_iterator,
                        get_optimizer, guess_pseudo_labels, make_variable,
                        sample_candidatas, save_model)


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

    # start training
    for epoch in range(cfg.num_epochs_pre):
        for step, (images, labels) in enumerate(source_data):
            # convert into torch.autograd.Variable
            images = make_variable(images)
            labels = make_variable(labels)

            # zero-grad optimizer
            optimizer_F.zero_grad()
            optimizer_F_1.zero_grad()
            optimizer_F_2.zero_grad()
            optimizer_F_t.zero_grad()

            # forward networks
            out_F = F(images)
            out_F_1 = F_1(out_F)
            out_F_2 = F_2(out_F)
            out_F_t = F_t(out_F)

            # compute loss
            loss_similiar = calc_similiar_penalty(F_1, F_2)
            loss_F_1 = criterion(out_F_1, labels)
            loss_F_2 = criterion(out_F_2, labels)
            loss_F_t = criterion(out_F_t, labels)
            loss_F = loss_F_1 + loss_F_2 + loss_F_t + loss_similiar
            loss_F.backward()

            # optimize
            optimizer_F.step()
            optimizer_F_1.step()
            optimizer_F_2.step()
            optimizer_F_t.step()

            # print step info
            if ((step + 1) % cfg.log_step == 0):
                print("Epoch [{}/{}] Step[{}/{}] Loss("
                      "Total={:.5f} F_1={:.5f} F_2={:.5f} "
                      "F_t={:.5f} sim={:.5f})"
                      .format(epoch + 1,
                              cfg.num_epochs_pre,
                              step + 1,
                              len(source_data),
                              loss_F.data[0],
                              loss_F_1.data[0],
                              loss_F_2.data[0],
                              loss_F_t.data[0],
                              loss_similiar.data[0],
                              ))

        # save model
        if ((epoch + 1) % cfg.save_step == 0):
            save_model(F, "pretrain-F-{}.pt".format(epoch + 1))
            save_model(F_1, "pretrain-F_1-{}.pt".format(epoch + 1))
            save_model(F_2, "pretrain-F_2-{}.pt".format(epoch + 1))
            save_model(F_t, "pretrain-F_t-{}.pt".format(epoch + 1))

    # save final model
    save_model(F, "pretrain-F-final.pt")
    save_model(F_1, "pretrain-F_1-final.pt")
    save_model(F_2, "pretrain-F_2-final.pt")
    save_model(F_t, "pretrain-F_t-final.pt")


def genarate_labels(F, F_1, F_2, target_dataset):
    """Genrate pseudo labels for target domain dataset."""
    # set eval state for Dropout and BN layers
    F.eval()
    F_1.eval()
    F_2.eval()

    # get candidate samples
    images_tgt, labels_tgt = sample_candidatas(
        data=target_dataset.train_data,
        labels=target_dataset.train_labels,
        candidates_num=cfg.num_target_init,
        shuffle=True)

    # convert into torch.autograd.Variable
    images_tgt = make_variable(images_tgt)
    labels_tgt = make_variable(labels_tgt)

    # forward networks
    out_F_1 = F_1(F(images_tgt))
    out_F_2 = F_2(F(images_tgt))

    # guess pseudo labels
    T_l, pseudo_labels, true_labels = guess_pseudo_labels(
        images_tgt, labels_tgt, out_F_1.data.cpu(), out_F_2.data.cpu())

    return T_l, pseudo_labels, true_labels


def domain_adapt(F, F_1, F_2, F_t, labelled_data, target_data_labelled):
    """Perform Doamin Adaptation between source and target domains."""
    pass
