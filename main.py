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
    # enable_cudnn_benchmark()

    # load dataset
    source_dataset = get_data_loader(cfg.source_dataset, get_dataset=True)
    source_data_loader = get_data_loader(cfg.source_dataset)
    source_data_loader_test = get_data_loader(cfg.source_dataset, train=False)
    target_dataset = get_data_loader(cfg.target_dataset, get_dataset=True)
    # target_data_loader = get_data_loader(cfg.target_dataset)
    # target_data_loader_test = get_data_loader(cfg.target_dataset,train=False)

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
        pre_train(F, F_1, F_2, F_t, source_data_loader)
        print(">>> evaluate F+F_1")
        evaluate(F, F_1, source_data_loader_test)
        print(">>> evaluate F+F_2")
        evaluate(F, F_2, source_data_loader_test)
        print(">>> evaluate F+F_t")
        evaluate(F, F_t, source_data_loader_test)

    # generate pseudo labels on target dataset
    print("=== Generate Pseudo Label ===")
    excerpt, pseudo_labels = \
        genarate_labels(F, F_1, F_2, target_dataset, cfg.num_target_init)
    print(">>> Genrate pseudo labels {}".format(
        len(pseudo_labels)))

    # domain adapt between source and target datasets
    print("=== Domain Adapt ===")
    domain_adapt(F, F_1, F_2, F_t,
                 source_dataset, target_dataset, excerpt, pseudo_labels)
