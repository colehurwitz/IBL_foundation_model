from datasets import load_dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5, multi_session_zs_dataset_iTransformer
from models.ndt1 import NDT1
from models.stpatch import STPatch
from models.itransformer_multi import iTransformer
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer

# load config
kwargs = {
    "model": "include:src/configs/itransformer_multi.yaml"
}

EID_PATH = 'data/split_eids'

config = config_from_kwargs(kwargs)
config = update_config("src/configs/trainer_iTransformer_zs.yaml", config)

# make log dir
log_dir = os.path.join(config.dirs.log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb

    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config,
               name=config.wandb.run_name)

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
train_dataset, val_dataset, test_dataset = multi_session_zs_dataset_iTransformer(EID_PATH, config, n_eids_train=30)
try:
    bin_size = train_dataset["binsize"][0]
except:
    bin_size = train_dataset["bin_size"][0]
print(train_dataset.column_names)
print(f"bin_size: {bin_size}")

# make the dataloader
train_dataloader = make_loader(train_dataset,
                               target=config.data.target,
                               load_meta=config.data.load_meta,
                               batch_size=config.training.train_batch_size,
                               pad_to_right=True,
                               pad_value=-1.,
                               bin_size=bin_size,
                               max_time_length=config.data.max_time_length,
                               max_space_length=config.data.max_space_length,
                               dataset_name=config.data.dataset_name,
                               sort_by_depth=config.data.sort_by_depth,
                               sort_by_region=config.data.sort_by_region,
                               shuffle=True)

val_dataloader = make_loader(val_dataset,
                             target=config.data.target,
                             load_meta=config.data.load_meta,
                             batch_size=config.training.test_batch_size,
                             pad_to_right=True,
                             pad_value=-1.,
                             bin_size=bin_size,
                             max_time_length=config.data.max_time_length,
                             max_space_length=config.data.max_space_length,
                             dataset_name=config.data.dataset_name,
                             sort_by_depth=config.data.sort_by_depth,
                             sort_by_region=config.data.sort_by_region,
                             shuffle=False)

test_dataloader = make_loader(test_dataset,
                              target=config.data.target,
                              load_meta=config.data.load_meta,
                              batch_size=config.training.test_batch_size,
                              pad_to_right=True,
                              pad_value=-1.,
                              bin_size=bin_size,
                              max_time_length=config.data.max_time_length,
                              max_space_length=config.data.max_space_length,
                              dataset_name=config.data.dataset_name,
                              sort_by_depth=config.data.sort_by_depth,
                              sort_by_region=config.data.sort_by_region,
                              shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs)
model = accelerator.prepare(model)

print(model)
print(config.model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd,
                              eps=config.optimizer.eps)
lr_scheduler = OneCycleLR(
    optimizer=optimizer,
    total_steps=config.training.num_epochs * len(train_dataloader) // config.optimizer.gradient_accumulation_steps,
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
trainer_ = make_trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    **trainer_kwargs
)



# train loop
trainer_.train()
