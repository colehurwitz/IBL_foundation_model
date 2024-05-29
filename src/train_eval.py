import argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from models.stpatch import STPatch
from models.itransformer import iTransformer
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer
from utils.eval_utils import load_model_data_local, co_smoothing_eval, behavior_decoding
import warnings
warnings.simplefilter("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--model_name", type=str, default="NDT1")
args = ap.parse_args()

# load config
kwargs = {
    "model": "include:src/configs/ndt1.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/trainer.yaml", config)

config.model.encoder.masker.mode = args.mask_mode

# make log dir
log_dir = os.path.join(
    config.dirs.log_dir, "train", "model_{}".format(config.model.model_class),
    "method_{}".format(config.method.model_kwargs.method_name), 
    "mask_{}".format(args.mask_mode),
    "ratio_{}".format(args.mask_ratio)
)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="train_model_{}_method_{}_mask_{}_ratio_{}".format(config.model.model_class, config.method.model_kwargs.method_name, args.mask_mode, args.mask_ratio))

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
if "ibl" in config.data.dataset_name:
    dataset = load_dataset(config.dirs.dataset_dir, cache_dir=config.dirs.dataset_cache_dir)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]
    
    try:
       bin_size = train_dataset["binsize"][0]
    except:
       bin_size = train_dataset["bin_size"][0]

    if config.data.include_behav:
        dataset = load_from_disk(os.path.join(config.dirs.behav_dir))
        #dataset = concatenate_datasets([dataset["train"], dataset["val"], dataset["test"]])
        _dataset = dataset.train_test_split(test_size=0.2, seed=config.seed)['train']
        dataset = _dataset.train_test_split(test_size=0.1, seed=config.seed)
        try:
            bin_size = dataset["train"]["binsize"][0]
        except:
            bin_size = dataset["train"]["bin_size"][0]

        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        test_dataset = _dataset["test"]

    if config.model.model_class == "iTransformer" and config.model.encoder.embed_region:
        config["model"]["encoder"]["neuron_regions"] = list(set(str(b) for a in [row["cluster_regions"] for rows in dataset.values() for row in rows] for b in a))

    print(dataset.column_names)
    print(f"bin_size: {bin_size}")

else:
    train_dataset = get_data_from_h5("train", config.dirs.dataset_dir, config=config)
    test_dataset = get_data_from_h5("val", config.dirs.dataset_dir, config=config)
    bin_size = None

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

model.encoder.masker.mode = args.mask_mode
model.encoder.masker.ratio = args.mask_ratio
model = accelerator.prepare(model)

print("(train) masking mode: ", model.encoder.masker.mode)
print("(train) masking ratio: ", model.encoder.masker.ratio)
print("(train) masking active: ", model.encoder.masker.force_active)
if args.mask_mode == 'causal':
    model.encoder.context_forward = 0
    print("(train) context forward: ", model.encoder.context_forward)

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

#########################

mask_name = f"mask_{args.mask_mode}"
model_name = args.model_name
n_time_steps = 100

co_smooth = True
forward_pred = True
inter_region = True
intra_region = True
choice_decoding = True
continuous_decoding = True

print(mask_name)

base_path = '/mnt/home/yzhang1/ceph'

# Configuration
configs = {
    'model_config': 'src/configs/ndt1.yaml',
    'model_path': f'{base_path}/results/train/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/model_best.pt',
    'trainer_config': 'src/configs/trainer.yaml',
    'dataset_path': None, 
    'test_size': 0.2,
    'seed': 42,
    'mask_name': mask_name,
}  

# import wandb
# wandb.init(project='ibl-ssl-eval', config=configs, 
#            name=f'model_{model_name}_method_ssl_{mask_name}_ratio_{args.mask_ratio}')

# load your model and dataloader
model, accelerator, dataset, dataloader = load_model_data_local(**configs)

# co-smoothing
if co_smooth:
    print('Start co-smoothing:')
    co_smoothing_configs = {
        'subtract': 'task',
        'onset_alignment': [40],
        'method_name': mask_name, 
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/co_smooth',
        'mode': 'per_neuron',
        'n_time_steps': n_time_steps,    
        'is_aligned': True,
        'target_regions': None
    }

    results = co_smoothing_eval(model, 
                    accelerator, 
                    dataloader, 
                    dataset, 
                    **co_smoothing_configs)
    print(results)
    wandb.log(results)


# forward prediction
if forward_pred:
    print('Start forward prediction:')
    results = co_smoothing_configs = {
        'subtract': 'task',
        'onset_alignment': [],
        'method_name': mask_name, 
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/forward_pred',
        'mode': 'forward_pred',
        'n_time_steps': n_time_steps,    
        'held_out_list': list(range(90, 100)), # NLB uses 200 ms for fp
        'is_aligned': True,
        'target_regions': None
    }

    results = co_smoothing_eval(model, 
                    accelerator, 
                    dataloader, 
                    dataset, 
                    **co_smoothing_configs)
    print(results)
    wandb.log(results)
    

# inter-region
if inter_region:
    print('Start inter-region:')
    co_smoothing_configs = {
        'subtract': 'task',
        'onset_alignment': [40],
        'method_name': mask_name,
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/inter_region',
        'mode': 'inter_region',
        'n_time_steps': n_time_steps,    
        'held_out_list': None,
        'is_aligned': True,
        'target_regions': ['all']
    }

    results = co_smoothing_eval(model, 
                    accelerator, 
                    dataloader, 
                    dataset, 
                    **co_smoothing_configs)
    print(results)
    wandb.log(results)


# intra-region
if intra_region:
    print('Start intra-region:')
    co_smoothing_configs = {
        'subtract': 'task',
        'onset_alignment': [40],
        'method_name': mask_name, 
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/intra_region',
        'mode': 'intra_region',
        'n_time_steps': n_time_steps,    
        'held_out_list': None,
        'is_aligned': True,
        'target_regions': ['all']
    }

    results = co_smoothing_eval(model, 
                    accelerator, 
                    dataloader, 
                    dataset, 
                    **co_smoothing_configs)
    print(results)
    wandb.log(results)


if choice_decoding:
    print('Start choice_decoding:')
    configs = {
        'model_config': 'src/configs/ndt1.yaml',
        'model_path': f'{base_path}/results/train/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/model_best.pt',
        'trainer_config': 'src/configs/trainer_sl_choice.yaml',
        'dataset_path': '/home/exouser/Documents/IBL_foundation_model/data/671c7ea7-6726-4fbe-adeb-f89c2c8e489b_aligned',
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/choice_decoding',
        'test_size': 0.2,
        'seed': 42,
        'mask_name': mask_name,
        'metric': 'acc',
        'from_scratch': False,
        'freeze_encoder': False,
        'mask_ratio': args.mask_ratio
    }  
    results = behavior_decoding(**configs)
    print(results)
    wandb.log(results)


if continuous_decoding:
    print('Start continuous_decoding:')
    configs = {
        'model_config': 'src/configs/ndt1.yaml',
        'model_path': f'{base_path}/results/train/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/model_best.pt',
        'trainer_config': 'src/configs/trainer_sl_continuous.yaml',
        'dataset_path': None, 
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/continuous_decoding',
        'test_size': 0.2,
        'seed': 42,
        'mask_name': mask_name,
        'metric': 'r2',
        'from_scratch': False,
        'freeze_encoder': False,
        'mask_ratio': args.mask_ratio
    }  
    results = behavior_decoding(**configs)
    print(results)
    wandb.log(results)
    