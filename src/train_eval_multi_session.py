import argparse
from math import ceil
from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
import argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
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
ap.add_argument("--test_eid", type=str, default='824cf03d-4012-4ab1-b499-c83a92c5589e')
ap.add_argument("--mask_ratio", type=float, default=0.3)
ap.add_argument("--mask_mode", type=str, default="all")
ap.add_argument("--model_name", type=str, default="NDT1")
ap.add_argument("--use_nemo", action='store_true')
ap.add_argument("--embed_nemo", action='store_true')
ap.add_argument("--cont_target", type=str, default="whisker-motion-energy")
ap.add_argument("--train", action='store_true')
ap.add_argument("--eval", action='store_true')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--save_plot", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
args = ap.parse_args()

base_path = args.base_path

eid = args.test_eid

model_acroynm = args.model_name.lower()

if args.model_name == 'iTransformer':
    if args.embed_nemo:
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_no_channel.yaml"
        }
    else:
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}/{model_acroynm}.yaml"
        }
    stitching = False
elif args.model_name == 'NDT1':
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_stitching_prompting.yaml"
    }
    stitching = True  
elif args.model_mame == 'NDT2':
    kwargs = {
        "model": f"include:src/configs/{model_acroynm}/{model_acroynm}_prompting.yaml"
    }
    stitching = False

config = config_from_kwargs(kwargs)
config = update_config("src/configs/multi_sessions_trainer.yaml", config)

num_train_sessions = len(config.data.train_session_eid)

if config.wandb.use:
    import wandb
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config, 
        name="train_multi_sessions_model_{}_method_{}_mask_{}_stitch_{}_NEMO_{}_{}_sessions".format(
            config.model.model_class, config.method.model_kwargs.method_name, args.mask_mode, stitching, args.embed_nemo, num_train_sessions
        )
    )

set_seed(config.seed)

last_ckpt_path = 'last' if config.model.model_class == 'iTransformer' else 'model_last.pt'
best_ckpt_path = 'best' if config.model.model_class == 'iTransformer' else 'model_best.pt'

if args.train:
    
    final_checkpoint = f'{base_path}/results/train/multi_sessions/model_{args.model_name}/method_ssl/mask_{args.mask_mode}/stitch_{stitching}/NEMO_{args.embed_nemo}/{num_train_sessions}_sessions/{last_ckpt_path}'
    
    if not os.path.exists(final_checkpoint) or args.overwrite:
        
        log_dir = os.path.join(base_path, config.dirs.log_dir, 
           "train", 
           "multi_sessions", 
           "model_{}".format(config.model.model_class), 
           "method_{}".format(config.method.model_kwargs.method_name), 
           "mask_{}".format(args.mask_mode),
           "stitch_{}".format(stitching),
           "NEMO_{}".format(args.embed_nemo),
           "{}_sessions".format(num_train_sessions)
          )
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
        train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
           config.dirs.dataset_cache_dir, 
           config.dirs.huggingface_org,
           eid=None,
           num_sessions=config.data.num_sessions,
           split_method=config.data.split_method,
           train_session_eid=config.data.train_session_eid,
           test_session_eid=config.data.test_session_eid,
           batch_size=config.training.train_batch_size,
           seed=config.seed,
           use_nemo=args.use_nemo
        )

        print('Start model training.')
        print('=====================')
    
        if args.model_name in ["NDT1", "iTransformer"]:
            max_space_length = config.data.max_space_length  
        elif args.model_name in ["NDT2", "STPatch"]:
            max_space_F = config.model.encoder.embedder.max_space_F
            max_space_length = ceil(n_neurons/max_space_F) * max_space_F

        meta_data['max_space_length'] = max_space_length
        print('encoder max space length:', max_space_length)
        print(meta_data)
        
        train_dataloader = make_loader(train_dataset, 
                                 target=config.data.target,
                                 load_meta=config.data.load_meta,
                                 use_nemo=args.use_nemo,
                                 wvf_only=False,
                                 batch_size=config.training.train_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=max_space_length,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 stitching=stitching,
                                 multi_session=True,
                                 shuffle=True)
        
        val_dataloader = make_loader(val_dataset, 
                                 target=config.data.target,
                                 load_meta=config.data.load_meta,
                                 use_nemo=args.use_nemo,
                                 wvf_only=False,
                                 batch_size=config.training.test_batch_size, 
                                 pad_to_right=True, 
                                 pad_value=-1.,
                                 max_time_length=config.data.max_time_length,
                                 max_space_length=max_space_length,
                                 dataset_name=config.data.dataset_name,
                                 sort_by_depth=config.data.sort_by_depth,
                                 sort_by_region=config.data.sort_by_region,
                                 stitching=stitching,
                                 multi_session=True,
                                 shuffle=False)
        
        # Initialize the accelerator
        accelerator = Accelerator()
        
        # load model
        NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer}
        
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
        
        print(config)
        
        trainer_kwargs = {
            "log_dir": log_dir,
            "accelerator": accelerator,
            "lr_scheduler": lr_scheduler,
            "config": config,
            "stitching": stitching,
        }
        trainer = make_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            optimizer=optimizer,
            **trainer_kwargs,
            **meta_data
        )
        
        trainer.train()

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
    
    if args.model_name == 'iTransformer':
        if args.embed_nemo:
            model_config = f"src/configs/{model_acroynm}/{model_acroynm}_no_channel_eval.yaml"
        else:
            model_config = f"src/configs/{model_acroynm}/{model_acroynm}_eval.yaml"
    elif args.model_name == 'NDT1':
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_stitching_prompting_eval.yaml"
    elif args.model_name == 'NDT2':
        model_config = f"src/configs/{model_acroynm}/{model_acroynm}_prompting_eval.yaml"
    
    configs = {
        'model_config': model_config,
        'model_path': f'{base_path}/results/train/multi_sessions/model_{args.model_name}/method_ssl/mask_{args.mask_mode}/stitch_{stitching}/NEMO_{args.embed_nemo}/{num_train_sessions}_sessions/{best_ckpt_path}',
        'trainer_config': f'src/configs/{model_acroynm}/trainer_{model_acroynm}.yaml',
        'dataset_path': None, 
        'test_size': 0.2,
        'seed': 42,
        'mask_name': mask_name,
        'eid': eid,
        'stitching': True,
        'num_sessions': 1,
        'use_nemo': args.use_nemo,
        'wvf_only': False,
    }  
    
    
    # load your model and dataloader
    model, accelerator, dataset, dataloader = load_model_data_local(**configs)
    
    # co-smoothing
    if co_smooth:
        
        co_smooth_bps_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/co_smooth/bps.npy'
        co_smooth_r2_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/co_smooth/r2.npy'
        
        if not os.path.exists(co_smooth_bps_file) or not os.path.exists(co_smooth_r2_file) or args.overwrite:
            print('Start co-smoothing:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/co_smooth',
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
                            save_plot=args.save_plot,
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
    
    
    # forward prediction
    if forward_pred:
        
        forward_pred_bps_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/forward_pred/bps.npy'
        forward_pred_r2_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/forward_pred/r2.npy'
        
        if not os.path.exists(forward_pred_bps_file) or not os.path.exists(forward_pred_r2_file) or args.overwrite:
            print('Start forward prediction:')
            results = co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/forward_pred',
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
                            save_plot=args.save_plot,
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
        
    
    # inter-region
    if inter_region:

        inter_region_bps_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/inter_region/bps.npy'
        inter_region_r2_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/inter_region/r2.npy'

        if not os.path.exists(inter_region_bps_file) or not os.path.exists(inter_region_r2_file) or args.overwrite:
            print('Start inter-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name,
                'save_path': f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/inter_region',
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
                            save_plot=args.save_plot,
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
    
    
    # intra-region
    if intra_region:

        intra_region_bps_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/intra_region/bps.npy'
        intra_region_r2_file = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/intra_region/r2.npy'
        
        if not os.path.exists(intra_region_bps_file) or not os.path.exists(intra_region_r2_file) or args.overwrite:
            print('Start intra-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/intra_region',
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
                            save_plot=args.save_plot,
                            **co_smoothing_configs)
            print(results)
            wandb.log(results)
    
    
    if choice_decoding:
        
        choice_results_dir = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/choice_decoding'
        if not os.path.exists(choice_results_dir) or args.overwrite:
            
            print('Start choice_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/train/multi_sessions/model_{args.model_name}/method_ssl/mask_{args.mask_mode}/stitch_{stitching}/NEMO_{args.embed_nemo}/{num_train_sessions}_sessions/{best_ckpt_path}',
                'trainer_config': f'src/configs/trainer_sl_choice_{model_acroynm}.yaml',
                'dataset_path': None,
                'save_path': f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/choice_decoding',
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
                'use_nemo': args.use_nemo,
                'wvf_only': False,
            }  
            results = behavior_decoding(**configs)
            print(results)
            wandb.log(results)
    
    
    if continuous_decoding:

        continuous_results_dir = f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/continuous_decoding'
        
        if not os.path.exists(continuous_results_dir) or args.overwrite:
        
            print('Start continuous_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/train/multi_sessions/model_{args.model_name}/method_ssl/mask_{args.mask_mode}/stitch_{stitching}/NEMO_{args.embed_nemo}/{num_train_sessions}_sessions/{best_ckpt_path}',
                'trainer_config': f'src/configs/trainer_sl_continuous_{model_acroynm}.yaml',
                'dataset_path': None, 
                'save_path': f'{base_path}/results/{eid}/eval/multi_sessions/model_{model_name}/method_ssl/{mask_name}/stitch_{stitching}/NEMO_{args.embed_nemo}/continuous_decoding',
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
                'use_nemo': args.use_nemo,
                'wvf_only': False,
            }  
            results = behavior_decoding(**configs)
            print(results)
            wandb.log(results)
    

