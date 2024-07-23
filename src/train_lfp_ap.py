import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle
import argparse
from math import ceil
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
import torch
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_trainer

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="5dcee0eb-b34d-4652-acc3-d10afc6eae68")
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--model_name", type=str, default="NDT1")
ap.add_argument("--modality", type=str, default="lfp", choices=["lfp", "ap"])
ap.add_argument("--task", type=str, default="sl", choices=["sl", "ssl"]) 
ap.add_argument("--train", action="store_true")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--project_name", type=str, default="lfp")
args = ap.parse_args()


"""
-------
CONFIGS
-------
"""
eid = args.eid
base_path = args.base_path
model_acroynm = args.model_name.lower()
modal_name = "LFP" if args.modality == "lfp" else "AP"
task_name = "supervised" if args.task == "sl" else "self-supervised"
config_dir = "lfp" if args.modality == "lfp" else model_acroynm

kwargs = {
    "model": f"include:src/configs/{config_dir}/{model_acroynm}.yaml"
}
config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/{config_dir}/trainer_{args.task}.yaml", config)
set_seed(config.seed)

if config.wandb.use:
    import wandb
    wandb.init(
        project=args.project_name, entity=config.wandb.entity, config=config,
        name="{}_train_model_{}_modal_{}_task_{}_mask_{}_ratio_{}".format(
            eid[:4], model_acroynm, args.modality, args.task, args.mask_mode, args.mask_ratio, 
        )
    )
last_ckpt_path = "model_last.pt"
best_ckpt_path = "model_best.pt"


"""
--------
Training
--------
"""
if args.train:
    
    final_checkpoint = f"{base_path}/results/{eid}/train/model_{model_acroynm}/modal_{args.modality}/task_{args.task}/mask_{args.mask_mode}/ratio_{args.mask_ratio}/{last_ckpt_path}"
    
    if not os.path.exists(final_checkpoint) or args.overwrite:

        print(f"Train {args.model_name} on session {eid} for {task_name} task using {modal_name} data.")
        
        _, _, _, meta_data = load_ibl_dataset(
            config.dirs.dataset_cache_dir, 
            config.dirs.huggingface_org,
            eid=eid,
            num_sessions=1,
            split_method="predefined",
            test_session_eid=[],
            batch_size=config.training.train_batch_size,
            seed=config.seed
        )
    
        print('Start model training.')
        print('=====================')

        log_dir = os.path.join(
            args.base_path,
            config.dirs.log_dir, eid, "train", 
            "model_{}".format(model_acroynm),
            "modal_{}".format(args.modality),
            "task_{}".format(args.task),
            "mask_{}".format(args.mask_mode),
            "ratio_{}".format(args.mask_ratio),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Load data
        dataset = load_dataset(f'neurofm123/{eid}_aligned_4s', cache_dir=config.dirs.dataset_cache_dir)
        train_dataset, val_dataset, test_dataset = dataset["train"], dataset["val"], dataset["test"]
        n_timesteps, n_neurons = np.array(train_dataset['lfp'][0]).shape
        meta_data['max_space_length'] = n_neurons
        meta_data['num_neurons'] = [n_neurons]
        print("Meta data:")
        print(meta_data)

        train_dataloader = make_loader(
            train_dataset, 
            target=config.data.target,
            load_meta=config.data.load_meta,
            batch_size=config.training.train_batch_size, 
            pad_to_right=True, 
            pad_value=-1.,
            max_time_length=n_timesteps,
            max_space_length=n_neurons,
            lfp_only=True if args.modality == "lfp" else False,
            shuffle=True
        )

        val_dataloader = make_loader(
            val_dataset, 
            target=config.data.target,
            load_meta=config.data.load_meta,
            batch_size=config.training.test_batch_size, 
            pad_to_right=True, 
            pad_value=-1.,
            max_time_length=n_timesteps,
            max_space_length=n_neurons,
            lfp_only=True if args.modality == "lfp" else False,
            shuffle=False
        )

        test_dataloader = make_loader(
            test_dataset, 
            target=config.data.target,
            load_meta=config.data.load_meta,
            batch_size=config.training.test_batch_size, 
            pad_to_right=True, 
            pad_value=-1.,
            max_time_length=n_timesteps,
            max_space_length=n_neurons,
            lfp_only=True if args.modality == "lfp" else False,
            shuffle=False
        )

        accelerator = Accelerator()

        # Load model
        NAME2MODEL = {"NDT1": NDT1}
        model_class = NAME2MODEL[config.model.model_class]
        model = model_class(config.model, **config.method.model_kwargs, **meta_data)

        model.encoder.masker.mode = args.mask_mode
        model.encoder.masker.ratio = args.mask_ratio
        if args.mask_mode == "causal":
            model.encoder.context_forward = 0
            print("(train) context forward: ", model.encoder.context_forward)
        print("(train) masking mode: ", model.encoder.masker.mode)
        print("(train) masking ratio: ", model.encoder.masker.ratio)
        print("(train) masking active: ", model.encoder.masker.force_active)
        
        model = accelerator.prepare(model)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps
        )
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=config.training.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
            max_lr=config.optimizer.lr,
            pct_start=config.optimizer.warmup_pct,
            div_factor=config.optimizer.div_factor,
        )
        trainer_kwargs = {
            "log_dir": log_dir, "accelerator": accelerator, "lr_scheduler": lr_scheduler, "config": config,
        }
        trainer_ = make_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            **trainer_kwargs,
            **meta_data
        )
        trainer_.train()
    else:
        print("Skipping training since last checkpoint exists or overwrite is False.")

