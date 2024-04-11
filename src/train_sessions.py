from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
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
log_dir = os.path.join(config.dirs.log_dir, "train", "model_{}".format(config.model.model_class), "method_{}".format(config.method.model_kwargs.method_name), "mask_{}".format(config.encoder.masker.mode))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="train_model_{}_method_{}_mask_{}".format(config.model.model_class, config.method.model_kwargs.method_name,config.encoder.masker.mode))

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
eid = None
train_dataset, test_dataset = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                           config.dirs.huggingface_org,
                           eid=eid,
                           num_sessions=config.data.num_sessions,
                           split_method=config.data.split_method,
                           test_session_eid=config.data.test_session_eid,
                           seed=config.seed)
if config.data.use_aligned_test:
    # aligned dataset
    if eid is None:
        test_dataset = load_from_disk(os.path.join('data', config.dirs.behav_dir))
        data_columns = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape']
        test_dataset = concatenate_datasets([test_dataset["train"], test_dataset["val"], test_dataset["test"]])
        test_dataset = test_dataset.select_columns(data_columns)
    else:
        aligned_dataset = load_from_disk(os.path.join('data', config.dirs.behav_dir))
        aligned_dataset = concatenate_datasets([aligned_dataset["train"], aligned_dataset["val"], aligned_dataset["test"]])
        train_dataset, test_dataset = split_both_dataset(aligned_dataset=aligned_dataset,
                                                         unaligned_dataset=train_dataset,
                                                         seed=config.seed)


# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         batch_size=config.training.train_batch_size, 
                         pad_to_right=True, 
                         patching=config.data.patching,
                         pad_value=-1.,
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