from datasets import load_dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from loader.make_loader import make_loader, make_ndt2_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from models.ndt2 import NDT2
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer

# load config
kwargs = {
    "model": "include:src/configs/ndt2.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt2.yaml", config)
config = update_config("src/configs/trainer.yaml", config)

# make log dir
log_dir = os.path.join(config.dirs.log_dir, "train", "model_{}".format(config.model.model_class), "method_{}".format(config.method.model_kwargs.method_name))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="train_model_{}_method_{}".format(config.model.model_class, config.method.model_kwargs.method_name))

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
if "ibl" in config.data.dataset_name:
    dataset = load_dataset(config.dirs.dataset_dir, cache_dir=config.dirs.dataset_cache_dir)
    # show the columns
    
    bin_size = dataset["train"]["bin_size"][0]
    

    # split the dataset to train and test
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=config.seed)
    # select the train dataset and the spikes_sparse_data column
    data_columns = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape']
    train_dataset = dataset["train"].select_columns(data_columns)
    test_dataset = dataset["test"].select_columns(data_columns)

    if config.data.include_behav:
        dataset = load_from_disk(os.path.join('data', config.dirs.behav_dir))
        dataset = concatenate_datasets([dataset["train"], dataset["val"], dataset["test"]])
        dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)
        bin_size = dataset["train"]["binsize"][0]

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    print(dataset.column_names)
    print(f"bin_size: {bin_size}")

else:
    train_dataset = get_data_from_h5("train", config.dirs.dataset_dir, config=config)
    test_dataset = get_data_from_h5("val", config.dirs.dataset_dir, config=config)
    bin_size = None

# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         batch_size=config.training.train_batch_size, 
                         pad_to_right=True, 
                         patching=config.data.patching,
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         n_neurons_per_patch=config.data.n_neurons_per_patch,
                         dataset_name=config.data.dataset_name,
                         shuffle=True)

test_dataloader = make_loader(test_dataset, 
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         patching=config.data.patching,
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         n_neurons_per_patch=config.data.n_neurons_per_patch,
                         dataset_name=config.data.dataset_name,
                         shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1, "NDT2": NDT2}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs)
model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
lr_scheduler = OneCycleLR(
                optimizer=optimizer,
                total_steps=config.training.num_epochs*len(train_dataloader) //config.optimizer.gradient_accumulation_steps,
                max_lr=config.optimizer.lr,
                pct_start=config.optimizer.warmup_pct,
                div_factor=config.optimizer.div_factor,
            )

trainer_kwargs = {
    "log_dir": log_dir,
    "accelerator": accelerator,
    "lr_scheduler": lr_scheduler,
    "config": config,
}
trainer = make_trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=None,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    **trainer_kwargs
)

# train loop
trainer.train()