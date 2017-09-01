"""Config for ATDA."""

# params for dataset and data loader
data_root = "data"
src_dataset = "MNIST"
tgt_dataset = "USPS"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 128
image_size = 64

# params for encoder (F)
model_restore = {
    "F": None,
    "F1": None,
    "F2": None,
    "Ft": None
}

# params for classifier(F1, F2, Ft)
dropout_keep = {
    "F1": 0.5,
    "F2": 0.5,
    "Ft": 0.2,
}

# params for training network
num_gpu = 1
num_epochs_pre = 5
num_epochs_adapt = 100
num_target_init = 5000
num_target_max = 40000
log_step = 100
save_step = 5000
manual_seed = None
model_root = "snapshots"
eval_only = False

# params for optimizing models
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
