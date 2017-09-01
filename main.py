"""Main script for ATDA."""

from core import domain_adapt, evaluate, labelling, pre_train
from misc import config as cfg
from misc.utils import get_data_loader, init_model, init_random_seed
from models import ClassifierA, EncoderA

if __name__ == '__main__':
    print("=== Init ===")
    # init random seed
    init_random_seed(cfg.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(cfg.src_dataset)
    src_data_loader_test = get_data_loader(cfg.src_dataset, train=False)
    tgt_data_loader = get_data_loader(cfg.tgt_dataset)
    tgt_data_loader_test = get_data_loader(cfg.tgt_dataset, train=False)

    # init models
    F = init_model(net=EncoderA(), restore=cfg.model_restore["F"])
    F_1 = init_model(net=ClassifierA(cfg.dropout_keep["F1"]),
                     restore=cfg.model_restore["F1"])
    F_2 = init_model(net=ClassifierA(cfg.dropout_keep["F2"]),
                     restore=cfg.model_restore["F2"])
    F_t = init_model(net=ClassifierA(cfg.dropout_keep["Ft"]),
                     restore=cfg.model_restore["Ft"])
    print(">>> F model <<<")
    print(F)
    print(">>> F_1 model <<<")
    print(F_1)
    print(">>> F_2 model <<<")
    print(F_2)
    print(">>> F_t model <<<")
    print(F_t)

    print("=== Pre-train ===")
    pre_train(F, F_1, F_2, F_t, src_data_loader)
    print(">>> evaluate F+F_1")
    evaluate(F, F_1, src_data_loader_test)
    print(">>> evaluate F+F_2")
    evaluate(F, F_2, src_data_loader_test)
    print(">>> evaluate F+F_t")
    evaluate(F, F_t, src_data_loader_test)
