from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
from models.stpatch import STPatch
from torch.optim.lr_scheduler import OneCycleLR
import torch
import os
from trainer.make import make_trainer

# load config
kwargs = {
    "model": "include:src/configs/ndt1.yaml"
}


config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1.yaml", config)
config = update_config("src/configs/ssl_session_trainer.yaml", config)

# make log dir
log_dir = os.path.join(config.dirs.log_dir, 
                       "train", 
                       "single_session",
                       "model_{}".format(config.model.model_class), 
                       "method_{}".format(config.method.model_kwargs.method_name), 
                       "mask_{}".format(config.encoder.masker.mode))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, 
               entity=config.wandb.entity, 
               config=config, 
               name="train_model_{}_method_{}_mask_{}_finetune_{}".format(config.model.model_class, config.method.model_kwargs.method_name,config.encoder.masker.mode, config.training.finetune))

# set seed for reproducibility
set_seed(config.seed)

eid = "671c7ea7-6726-4fbe-adeb-f89c2c8e489b"
# download dataset from huggingface
train_dataset, val_dataset, test_dataset = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                           config.dirs.huggingface_org,
                           aligned_data_dir=config.dirs.aligned_data_dir,
                           eid=eid,
                           num_sessions=config.data.num_sessions,
                           split_method=config.data.split_method,
                           test_session_eid=config.data.test_session_eid,
                           seed=config.seed)

# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         target=config.data.target,
                         batch_size=config.training.train_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         dataset_name=config.data.dataset_name,
                         load_meta=config.data.load_meta,
                         shuffle=True)

eval_dataloader = make_loader(val_dataset, 
                         target=config.data.target,
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         dataset_name=config.data.dataset_name,
                         load_meta=config.data.load_meta,
                         shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs)
if config.training.finetune:
    print('Loading pretrained model')
    model = torch.load(config.dirs.pretrained_model_path)['model']
model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=config.optimizer.lr, 
                              weight_decay=config.optimizer.wd, 
                              eps=config.optimizer.eps)
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
    eval_dataloader=eval_dataloader,
    optimizer=optimizer,
    **trainer_kwargs
)

# train loop
trainer.train()