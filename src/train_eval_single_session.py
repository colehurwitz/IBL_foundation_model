import pickle
import argparse
from math import ceil
from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
import argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
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
import threading
import warnings

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='671c7ea7-6726-4fbe-adeb-f89c2c8e489b')
ap.add_argument("--eid_idx", type=int, default=-1)
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--model_name", type=str, default="NDT1")
ap.add_argument("--tokenize_binary_mask", action='store_true')
ap.add_argument("--prompting", action='store_true')
ap.add_argument("--use_nemo", action='store_true')
ap.add_argument("--embed_nemo", action='store_true')
ap.add_argument("--no_channel_embed", action='store_true')
ap.add_argument("--cont_target", type=str, default="whisker-motion-energy")
ap.add_argument("--train", action='store_true')
ap.add_argument("--eval", action='store_true')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
args = ap.parse_args()

base_path = args.base_path

eid = args.eid
    
print(f'Working on EID: {eid} ...')


model_acroynm = args.model_name.lower()

# load config
if (args.mask_mode == 'causal') & (args.model_name != 'iTransformer'):
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_causal.yaml"
    }
elif (args.prompting) & (args.model_name != 'iTransformer'):
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_prompting.yaml"
    }
elif args.tokenize_binary_mask:
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_mask_token.yaml"
    }
elif args.no_channel_embed:
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_no_channel.yaml"
    }
elif args.embed_nemo:
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_nemo.yaml"
    }
else:
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}.yaml"
    }

config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/{model_acroynm}/trainer_{model_acroynm}.yaml", config)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="{}_train_model_{}_method_{}_mask_{}_ratio_{}_mask_token_{}_prompt_{}_NEMO_{}_no_channel_{}".format(
            eid[:5],
            config.model.model_class, config.method.model_kwargs.method_name, 
            args.mask_mode, args.mask_ratio, args.tokenize_binary_mask, args.prompting, args.embed_nemo, args.no_channel_embed
        )
    )

# set seed for reproducibility
set_seed(config.seed)

last_ckpt_path = 'last' if config.model.model_class == 'iTransformer' else 'model_last.pt'
best_ckpt_path = 'best' if config.model.model_class == 'iTransformer' else 'model_best.pt'

if args.train:
    final_checkpoint = f'{base_path}/results/{eid}/train/model_{args.model_name}/method_ssl/mask_{args.mask_mode}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/{last_ckpt_path}'
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
            config.dirs.log_dir, eid, "train", "model_{}".format(config.model.model_class),
            "method_{}".format(config.method.model_kwargs.method_name), 
            "mask_{}".format(args.mask_mode),
            "ratio_{}".format(args.mask_ratio),
            "mask_token_{}".format(args.tokenize_binary_mask),
            "prompt_{}".format(args.prompting),
            "NEMO_{}".format(args.embed_nemo),
            "no_channel_{}".format(args.no_channel_embed),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if "ibl" in config.data.dataset_name:
            dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
            train_dataset = dataset["train"]
            val_dataset = dataset["val"]
            test_dataset = dataset["test"]

            if args.use_nemo:
                neuron_uuids = np.array(train_dataset['cluster_uuids'][0]).astype('str')
                with open('data/MtM_unit_embed.pkl','rb') as file:
                    nemo_data = pickle.load(file)
                nemo_uuids = nemo_data['uuids']
                include_uuids = np.intersect1d(neuron_uuids, nemo_uuids)
                n_neurons = len(include_uuids)
                print('Use NEMO cell-type embeddings.')
                print('Num of neurons with NEMO embeddings: ', n_neurons)
            else:
                n_neurons = len(train_dataset['cluster_regions'][0])

            if config.model.model_class == "iTransformer" and config.model.encoder.embed_region:
                config["model"]["encoder"]["neuron_regions"] = list(
                    set(str(b) for a in [row["cluster_regions"] for rows in dataset.values() for row in rows] for b in a)
                )
            print(dataset.column_names)
        else:
            train_dataset = get_data_from_h5("train", config.dirs.dataset_dir, config=config)
            test_dataset = get_data_from_h5("val", config.dirs.dataset_dir, config=config)

        if args.model_name in ["NDT1", "iTransformer"]:
            max_space_length = n_neurons  
        elif args.model_name in ["NDT2", "STPatch"]:
            max_space_F = config.model.encoder.embedder.max_space_F
            max_space_length = ceil(n_neurons/max_space_F) * max_space_F
        else:
            max_space_length = config.data.max_space_length

        print('encoder max space length:', max_space_length)

        meta_data['max_space_length'] = max_space_length
        
        meta_data['num_neurons'] = [n_neurons]
        print(meta_data)

        # make the dataloader
        train_dataloader = make_loader(train_dataset, 
                                 target=config.data.target,
                                 load_meta=config.data.load_meta,
                                 use_nemo=args.use_nemo,
                                 batch_size=config.training.train_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=max_space_length,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 shuffle=True)

        val_dataloader = make_loader(val_dataset, 
                                 target=config.data.target,
                                 load_meta=config.data.load_meta,
                                 use_nemo=args.use_nemo,
                                 batch_size=config.training.test_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=max_space_length,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 shuffle=False)

        test_dataloader = make_loader(test_dataset, 
                                 target=config.data.target,
                                 load_meta=config.data.load_meta,
                                 use_nemo=args.use_nemo,
                                 batch_size=config.training.test_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=max_space_length,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 shuffle=False)

        # Initialize the accelerator
        accelerator = Accelerator()

        # load model
        NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer}
        model_class = NAME2MODEL[config.model.model_class]
        model = model_class(config.model, **config.method.model_kwargs, **meta_data)

        if config.model.model_class == 'iTransformer':
            model.masker.mode = args.mask_mode
            model.masker.ratio = args.mask_ratio
            print("(train) masking mode: ", model.masker.mode)
            print("(train) masking ratio: ", model.masker.ratio)
            print("(train) masking active: ", model.masker.force_active)
        else:
            model.encoder.masker.mode = args.mask_mode
            model.encoder.masker.ratio = args.mask_ratio
            print("(train) masking mode: ", model.encoder.masker.mode)
            print("(train) masking ratio: ", model.encoder.masker.ratio)
            print("(train) masking active: ", model.encoder.masker.force_active)
            if args.mask_mode == 'causal':
                model.encoder.context_forward = 0
                print("(train) context forward: ", model.encoder.context_forward)
        
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
        print("skipping training since last checkpoint exists or overwrite is False")

#########################

if args.eval:
    
    print('Start model evaluation.')
    print('=======================')
    
    mask_name = f"mask_{args.mask_mode}"
    if args.model_name == "NDT2":
        model_name = "STPatch"
    else:
        model_name = args.model_name
        
    n_time_steps = 100
    
    co_smooth = True
    forward_pred = True
    inter_region = True
    intra_region = True
    choice_decoding = True
    continuous_decoding = True
    
    print(mask_name)

    if (args.mask_mode == 'causal') & (args.model_name != 'iTransformer'):
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_causal_eval.yaml"
    elif (args.prompting) & (args.model_name != 'iTransformer'):
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_prompting_eval.yaml"
    elif args.tokenize_binary_mask:
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_mask_token_eval.yaml"
    elif args.no_channel_embed:
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_no_channel_eval.yaml"
    elif args.embed_nemo:
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_nemo_eval.yaml"
    else:
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_eval.yaml"
    
    # Configuration
    configs = {
        'model_config': model_config,
        'model_path': f'{base_path}/results/{eid}/train/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/{best_ckpt_path}',
        'trainer_config': f'src/configs/{model_acroynm}/trainer_{model_acroynm}.yaml',
        'dataset_path': None, 
        'test_size': 0.2,
        'seed': 42,
        'mask_name': mask_name,
        'eid': eid,
        'stitching': False,
        'num_sessions': 1,
        'use_nemo': args.use_nemo
    }  
    
    # load your model and dataloader
    model, accelerator, dataset, dataloader = load_model_data_local(**configs)
    
    # co-smoothing
    if co_smooth:
        co_smooth_bps_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/co_smooth/bps.npy'
        co_smooth_r2_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/co_smooth/r2.npy'
        if not os.path.exists(co_smooth_bps_file) or not os.path.exists(co_smooth_r2_file) or args.overwrite:
            print('Start co-smoothing:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/co_smooth',
                'mode': 'per_neuron',
                'n_time_steps': n_time_steps,    
                'is_aligned': True,
                'target_regions': None,
                'n_jobs': 1
            }
    
            results = co_smoothing_eval(model, 
                            accelerator, 
                            dataloader, 
                            dataset, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
        else:
            print("skipping co_smoothing since files exist or overwrite is False")
    
    
    # forward prediction
    if forward_pred:
        forward_pred_bps_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/forward_pred/bps.npy'
        forward_pred_r2_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/forward_pred/r2.npy'
        if not os.path.exists(forward_pred_bps_file) or not os.path.exists(forward_pred_r2_file) or args.overwrite:
            print('Start forward prediction:')
            results = co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/forward_pred',
                'mode': 'forward_pred',
                'n_time_steps': n_time_steps,    
                'held_out_list': list(range(90, 100)), # NLB uses 200 ms for fp
                'is_aligned': True,
                'target_regions': None,
                'n_jobs': 1
            }
    
            results = co_smoothing_eval(model, 
                            accelerator, 
                            dataloader, 
                            dataset, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
        else:
            print("skipping forward_pred since files exist or overwrite is False")
        
    
    # inter-region
    if inter_region:
        inter_region_bps_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/inter_region/bps.npy'
        inter_region_r2_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/inter_region/r2.npy'
        if not os.path.exists(inter_region_bps_file) or not os.path.exists(inter_region_r2_file) or args.overwrite:
            print('Start inter-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name,
                'save_path': f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/inter_region',
                'mode': 'inter_region',
                'n_time_steps': n_time_steps,    
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 1
            }
    
            results = co_smoothing_eval(model, 
                            accelerator, 
                            dataloader, 
                            dataset, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
        else:
            print("skipping inter_region since files exist or overwrite is False")
    
    
    # intra-region
    if intra_region:
        intra_region_bps_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/intra_region/bps.npy'
        intra_region_r2_file = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/intra_region/r2.npy'
        if not os.path.exists(intra_region_bps_file) or not os.path.exists(intra_region_r2_file) or args.overwrite:
            print('Start intra-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/intra_region',
                'mode': 'intra_region',
                'n_time_steps': n_time_steps,    
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 1
            }
    
            results = co_smoothing_eval(model, 
                            accelerator, 
                            dataloader, 
                            dataset, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
        else:
            print("skipping intra_region since files exist or overwrite is False")
    
    
    if choice_decoding:
        choice_results_dir = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/choice_decoding'
        if not os.path.exists(choice_results_dir) or args.overwrite:
            print('Start choice_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/{eid}/train/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/{best_ckpt_path}',
                'trainer_config': f'src/configs/{model_acroynm}/trainer_{model_acroynm}.yaml',
                'dataset_path': '/home/exouser/Documents/IBL_foundation_model/data/671c7ea7-6726-4fbe-adeb-f89c2c8e489b_aligned',
                'save_path': f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/choice_decoding',
                'test_size': 0.2,
                'seed': 42,
                'mask_name': mask_name,
                'metric': 'acc',
                'from_scratch': False,
                'freeze_encoder': True,
                'mask_ratio': args.mask_ratio,
                'eid': eid,
                'num_sessions': 1,
                'target': 'choice',
                'use_trial_filter': False,
                'use_nemo': args.use_nemo
            }  
            results = behavior_decoding(**configs)
            print(results)
            wandb.log(results)
        else:
            print("skipping choice decoding since folder exists or overwrite is False")
    
    
    if continuous_decoding:        
        continuous_results_dir = f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/continuous_decoding'
        if not os.path.exists(continuous_results_dir) or args.overwrite:
            print('Start continuous_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/{eid}/train/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/{best_ckpt_path}',
                'trainer_config': f'src/configs/{model_acroynm}/trainer_{model_acroynm}.yaml',
                'dataset_path': None, 
                'save_path': f'{base_path}/results/{eid}/eval/model_{model_name}/method_ssl/{mask_name}/ratio_{args.mask_ratio}/mask_token_{args.tokenize_binary_mask}/prompt_{args.prompting}/NEMO_{args.embed_nemo}/no_channel_{args.no_channel_embed}/continuous_decoding',
                'test_size': 0.2,
                'seed': 42,
                'mask_name': mask_name,
                'metric': 'rsquared',
                'from_scratch': False,
                'freeze_encoder': True,
                'mask_ratio': args.mask_ratio,
                'eid': eid,
                'num_sessions': 1,
                'target': args.cont_target,
                'use_trial_filter': False,
                'use_nemo': args.use_nemo
            }  
            results = behavior_decoding(**configs)
            print(results)
            wandb.log(results)
        else:
            print("skipping continuous decoding since folder exists or overwrite is False")
    

    
