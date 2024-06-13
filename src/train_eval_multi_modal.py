import os
import pickle
import argparse
from math import ceil
import numpy as np
import torch
import wandb
import warnings

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from multi_modal.mm import MultiModal
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.decoder_embeddings import DecoderEmbedding


ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='671c7ea7-6726-4fbe-adeb-f89c2c8e489b')
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--cont_target", type=str, default="left-whisker-motion-energy")
ap.add_argument("--train", action='store_true')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
args = ap.parse_args()

base_path = args.base_path

eid = args.eid
    
print(f'Working on EID: {eid} ...')

kwargs = {
    "model": f"include:src/configs/multi_modal/mm.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/multi_modal/trainer_mm.yaml", config)

if config.wandb.use:
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="{}_train_multi_modal_mask_{}_ratio_{}".format(
            eid[:5], args.mask_mode, args.mask_ratio
        )
    )

set_seed(config.seed)

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

if args.train:
    
    final_checkpoint = f'{base_path}/results/{eid}/train/multi_modal/mask_{args.mask_mode}/ratio_{args.mask_ratio}/{last_ckpt_path}'
    
    if not os.path.exists(final_checkpoint) or args.overwrite:
        
        _, _, _, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                           config.dirs.huggingface_org,
                           eid=eid,
                           num_sessions=1,
                           split_method="predefined",
                           test_session_eid=[],
                           batch_size=config.training.train_batch_size,
                           seed=config.seed)
        print(meta_data)
        
        print('Start model training.')
        print('=====================')

        log_dir = os.path.join(
            args.base_path,
            config.dirs.log_dir, eid, "train", "multi_modal",
            "mask_{}".format(args.mask_mode),
            "ratio_{}".format(args.mask_ratio),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
        train_dataset = dataset["train"]
        val_dataset = dataset["val"]
        test_dataset = dataset["test"]
        print(dataset.column_names)

        n_behaviors = 1  # CHANGE LATER
        n_neurons = len(train_dataset['cluster_regions'][0])
        meta_data['num_neurons'] = [n_neurons]
        print(meta_data)

        train_dataloader = make_loader(train_dataset, 
                                 target=args.cont_target,
                                 load_meta=config.data.load_meta,
                                 batch_size=config.training.train_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=n_neurons,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 shuffle=True)

        val_dataloader = make_loader(val_dataset, 
                                 target=args.cont_target,
                                 load_meta=config.data.load_meta,
                                 batch_size=config.training.test_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=n_neurons,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 shuffle=False)

        test_dataloader = make_loader(test_dataset, 
                                 target=args.cont_target,
                                 load_meta=config.data.load_meta,
                                 batch_size=config.training.test_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=n_neurons,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 shuffle=False)

        avail_mod = ['ap', 'behavior']
        
        encoder_embeddings, decoder_embeddings = {}, {}
        for mod in avail_mod:
            encoder_embeddings[mod] = EncoderEmbedding(
                hidden_size=config.model.encoder.transformer.hidden_size,
                n_channel=n_neurons if mod == 'ap' else n_behaviors,
                config=config.model.encoder,
            )
            decoder_embeddings[mod] = DecoderEmbedding(
                hidden_size=config.model.decoder.transformer.hidden_size,
                n_channel=n_neurons if mod == 'ap' else n_behaviors,
                config=config.model.decoder,
            )

        accelerator = Accelerator()

        NAME2MODEL = {"MultiModal": MultiModal}
        model_class = NAME2MODEL[config.model.model_class]
        model = model_class(
            encoder_embeddings,
            decoder_embeddings,
            avail_mod=avail_mod,
            config=config.model, 
            share_modality_embeddings=True,
            use_session=False,
            **config.method.model_kwargs, 
            **meta_data
        )

        model.masker.mode = args.mask_mode
        model.masker.ratio = args.mask_ratio
        print("(train) masking mode: ", model.masker.mode)
        print("(train) masking ratio: ", model.masker.ratio)
        print("(train) masking active: ", model.masker.force_active)
        if args.mask_mode == 'causal':
            model.context_forward = 0
            print("(train) context forward: ", model.context_forward)
        
        model = accelerator.prepare(model)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.optimizer.lr, 
            weight_decay=config.optimizer.wd, 
            eps=config.optimizer.eps
        )
        
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=config.training.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
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
        
        trainer_ = make_multimodal_trainer(
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
        print("skipping training since last checkpoint exists or overwrite is False")


