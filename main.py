"""Main script for ATDA."""

from core import domain_adapt, evaluate, genarate_labels, pre_train
from misc import config as cfg
from misc.utils import (enable_cudnn_benchmark, get_data_loader, init_model,
                        init_random_seed)
from models import ClassifierA, EncoderA

if __name__ == '__main__':
    print("=== Init ===")
    # init random seed
    init_random_seed(cfg.manual_seed)

    # speed up cudnn
    enable_cudnn_benchmark()

    # load dataset
    src_dataset = get_data_loader(cfg.src_dataset, get_dataset=True)
    src_data_loader = get_data_loader(cfg.src_dataset)
    src_data_loader_test = get_data_loader(cfg.src_dataset, train=False)
    tgt_dataset = get_data_loader(cfg.tgt_dataset, get_dataset=True)
    tgt_data_loader = get_data_loader(cfg.tgt_dataset)
    tgt_data_loader_test = get_data_loader(cfg.tgt_dataset, train=False)

    # init models
    F = init_model(net=EncoderA(), restore=cfg.model_restore["F"])
    F_1 = init_model(net=ClassifierA(cfg.dropout_keep["F_1"]),
                     restore=cfg.model_restore["F_1"])
    F_2 = init_model(net=ClassifierA(cfg.dropout_keep["F_2"]),
                     restore=cfg.model_restore["F_2"])
    F_t = init_model(net=ClassifierA(cfg.dropout_keep["F_t"]),
                     restore=cfg.model_restore["F_t"])

    # show model structure
    print(">>> F model <<<")
    print(F)
    print(">>> F_1 model <<<")
    print(F_1)
    print(">>> F_2 model <<<")
    print(F_2)
    print(">>> F_t model <<<")
    print(F_t)

    # pre-train on source dataset
    print("=== Pre-train ===")
    if cfg.model_trained["pretrain"]:
        print("pass")
    else:
        pre_train(F, F_1, F_2, F_t, src_data_loader)
        print(">>> evaluate F+F_1")
        evaluate(F, F_1, src_data_loader_test)
        print(">>> evaluate F+F_2")
        evaluate(F, F_2, src_data_loader_test)
        print(">>> evaluate F+F_t")
        evaluate(F, F_t, src_data_loader_test)

    # generate pseudo labels on target dataset
    print("=== Generate Pseudo Label ===")
    T_l, pseudo_labels, true_labels = \
        genarate_labels(F, F_1, F_2, tgt_dataset, cfg.num_target_init)
    print(">>> Genrate pseudo labels [{}]".format(pseudo_labels.size(0)))

    # domain adapt between source and target datasets
    print("=== Domain Adapt ===")
    domain_adapt(F, F_1, F_2, F_t,
                 src_dataset, tgt_dataset,
                 T_l, pseudo_labels)
