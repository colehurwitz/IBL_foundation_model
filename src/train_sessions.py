from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from models.stpatch import STPatch
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer
import threading

# load config
kwargs = {
    "model": "include:src/configs/ndt1_stitching.yaml"
}


config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1_stitching.yaml", config)
config = update_config("src/configs/ssl_sessions_trainer.yaml", config)

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
eid = None
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                           config.dirs.huggingface_org,
                           eid=eid,
                           num_sessions=config.data.num_sessions,
                           split_method=config.data.split_method,
                           test_session_eid=config.data.test_session_eid,
                           batch_size=config.training.train_batch_size,
                           use_re=config.data.use_re,
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
num_sessions = len(meta_data["eids"])
# make log dir
log_dir = os.path.join(config.dirs.log_dir, 
                       "train", 
                       "num_session_{}".format(num_sessions), 
                       "model_{}".format(config.model.model_class), 
                       "method_{}".format(config.method.model_kwargs.method_name), 
                       "mask_{}".format(config.encoder.masker.mode),
                       "stitch_{}".format(config.encoder.stitching))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="train_model_{}_num_session_{}_method_{}_mask_{}_stitch_{}".format(config.model.model_class, num_sessions,config.method.model_kwargs.method_name,config.encoder.masker.mode, config.encoder.stitching))

# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         target=config.data.target,
                         load_meta=config.data.load_meta,
                         batch_size=config.training.train_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         dataset_name=config.data.dataset_name,
                         sort_by_depth=config.data.sort_by_depth,
                         sort_by_region=config.data.sort_by_region,
                         stitching=config.encoder.stitching,
                         shuffle=True)

val_dataloader = make_loader(val_dataset, 
                         target=config.data.target,
                         load_meta=config.data.load_meta,
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         dataset_name=config.data.dataset_name,
                         sort_by_depth=config.data.sort_by_depth,
                         sort_by_region=config.data.sort_by_region,
                         stitching=config.encoder.stitching,
                         shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch}

config = update_config(config, meta_data)
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs, **meta_data)
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
    "stitching": config.encoder.stitching,
}
trainer = make_trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    optimizer=optimizer,
    **trainer_kwargs,
    **meta_data
)
# Shared variable to signal the dummy load to stop
stop_dummy_load = threading.Event()
if config.training.dummy:
    # This is for HPC GPU usage, to avoid the GPU being idle
    print("Running dummy load")
    # Run dummy load in a separate thread
    dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load,))
    dummy_thread.start()
    try:
        # train loop
        trainer.train()
    finally:
        # Signal the dummy load to stop and wait for the thread to finish
        stop_dummy_load.set()
        dummy_thread.join()
else:
    # train loop
    trainer.train()