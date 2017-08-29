"""Main script for ATDA."""

from misc import config as cfg
from misc.utils import (get_data_loader, init_model, init_random_seed,
                        make_variable)
from models import Classifier, Encoder

if __name__ == '__main__':
    # init random seed
    init_random_seed(cfg.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(cfg.src_dataset)
    src_data_loader_test = get_data_loader(cfg.src_dataset, train=False)
    tgt_data_loader = get_data_loader(cfg.tgt_dataset)
    tgt_data_loader_test = get_data_loader(cfg.tgt_dataset, train=False)

    # init models
    F = init_model(net=Encoder(), restore=cfg.model_restore["F"])
    F1 = init_model(net=Classifier(cfg.dropout_keep["F1"],
                                   cfg.conv_dims["input"],
                                   cfg.conv_dims["hidden"],
                                   cfg.conv_dims["output"]),
                    restore=cfg.model_restore["F1"])
    F2 = init_model(net=Classifier(cfg.dropout_keep["F2"],
                                   cfg.conv_dims["input"],
                                   cfg.conv_dims["hidden"],
                                   cfg.conv_dims["output"]),
                    restore=cfg.model_restore["F2"])
    Ft = init_model(net=Classifier(cfg.dropout_keep["Ft"],
                                   cfg.conv_dims["input"],
                                   cfg.conv_dims["hidden"],
                                   cfg.conv_dims["output"]),
                    restore=cfg.model_restore["Ft"])
