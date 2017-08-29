"""Config for ATDA."""

# params for dataset and data loader
data_root = "data"
src_dataset = "MNIST"
tgt_dataset = "USPS"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# params for F
F_model_restore = None

# params for F1
F1_model_restore = None

# params for F2
F2_model_restore = None

# params for Ft
Ft_model_restore = None

# params for training network
num_gpu = 1
num_epochs = 20000
log_step = 100
save_step = 5000
manual_seed = None
model_root = "snapshots"
eval_only = False

# params for optimizing models
learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
