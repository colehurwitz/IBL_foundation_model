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
warnings.simplefilter("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--test_eid", type=str, default='51e53aff-1d5d-4182-a684-aba783d50ae5')
ap.add_argument("--mask_ratio", type=float, default=0.3)
ap.add_argument("--mask_mode", type=str, default="all")
ap.add_argument("--model_name", type=str, default="NDT1")
ap.add_argument("--prompting", type=str, default="False")
ap.add_argument("--train", type=str, default="True")
ap.add_argument("--eval", type=str, default="True")
ap.add_argument("--base_path", type=str, default='/mnt/home/yzhang1/ceph')
ap.add_argument("--num_train_sessions", type=int, default=1)
ap.add_argument('--use_dummy', action='store_true')
args = ap.parse_args()

eid = args.test_eid
base_path = args.base_path
model_acroynm = args.model_name.lower()
num_train_sessions = args.num_train_sessions
assert num_train_sessions > 0, 'num_train_sessions should be greater than 0.'

if args.prompting == "True":
    if args.model_name == 'NDT1':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_stitching_prompting.yaml"
        }
    elif args.model_name == 'NDT2':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_prompting.yaml"
        }
else:
    if args.model_name == 'NDT1':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_stitching.yaml"
        }
    elif args.model_name == 'NDT2':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}.yaml"
        }

config = config_from_kwargs(kwargs)
config = update_config("src/configs/finetune_sessions_trainer.yaml", config)

set_seed(config.seed)
# Shared variable to signal the dummy load to stop
stop_dummy_load = threading.Event()
if args.use_dummy:
    print("Running dummy load")
    # Run dummy load in a separate thread
    dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load,80000))
    dummy_thread.start()
try:
    if args.train == "True":
        print('Start model training.')
        print('=====================')
        train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                            config.dirs.huggingface_org,
                            eid=None,
                            num_sessions=config.data.num_sessions,
                            split_method=config.data.split_method,
                            train_session_eid=[eid],
                            test_session_eid=config.data.test_session_eid,
                            batch_size=config.training.train_batch_size,
                            seed=config.seed)

        log_dir = os.path.join(base_path, config.dirs.log_dir, 
                            "finetune", 
                            "num_session_{}".format(num_train_sessions),
                            "model_{}".format(config.model.model_class), 
                            "method_{}".format(config.method.model_kwargs.method_name), 
                            "mask_{}".format(args.mask_mode),
                            "stitch_{}".format(config.model.encoder.stitching),
                            "{}".format(eid)
                            )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print(meta_data)

        if config.wandb.use:
            import wandb
            wandb.init(project=config.wandb.project, 
                    entity=config.wandb.entity, 
                    config=config, 
                    name="finetune_num_session_{}_model_{}_method_{}_mask_{}_stitch_{}_{}".format(num_train_sessions,config.model.model_class, config.method.model_kwargs.method_name,args.mask_mode, config.model.encoder.stitching,eid)
                    )

        if args.model_name in ["NDT1", "iTransformer"]:
            max_space_length = config.data.max_space_length
        elif args.model_name in ["NDT2", "STPatch"]:
            max_space_F = config.model.encoder.embedder.max_space_F
            max_num_neurons = max(meta_data['num_neurons'])
            max_space_length = ceil(max_num_neurons/max_space_F) * max_space_F
        else:
            max_space_length = config.data.max_space_length
        
        meta_data['max_space_length'] = max_space_length

        print('encoder max space length:', max_space_length)
        
        train_dataloader = make_loader(train_dataset, 
                                target=config.data.target,
                                load_meta=config.data.load_meta,
                                batch_size=config.training.train_batch_size, 
                                pad_to_right=True, 
                                pad_value=-1.,
                                max_time_length=config.data.max_time_length,
                                max_space_length=max_space_length,
                                dataset_name=config.data.dataset_name,
                                sort_by_depth=config.data.sort_by_depth,
                                sort_by_region=config.data.sort_by_region,
                                stitching=config.model.encoder.stitching,
                                shuffle=True)
        
        val_dataloader = make_loader(val_dataset, 
                                target=config.data.target,
                                load_meta=config.data.load_meta,
                                batch_size=config.training.test_batch_size, 
                                pad_to_right=True, 
                                pad_value=-1.,
                                max_time_length=config.data.max_time_length,
                                max_space_length=max_space_length,
                                dataset_name=config.data.dataset_name,
                                sort_by_depth=config.data.sort_by_depth,
                                sort_by_region=config.data.sort_by_region,
                                stitching=config.model.encoder.stitching,
                                shuffle=False)
        
        # Initialize the accelerator
        accelerator = Accelerator()
        
        # load model
        NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch}
        
        config = update_config(config, meta_data)
        model_class = NAME2MODEL[config.model.model_class]
        model = model_class(config.model, **config.method.model_kwargs, **meta_data)
        model = accelerator.prepare(model)
        # load pretrain model
        pretrain_model_path = f'{base_path}/results/train/num_session_{num_train_sessions}/model_{config.model.model_class}/method_{config.method.model_kwargs.method_name}/mask_{args.mask_mode}/stitch_{config.model.encoder.stitching}/model_best.pt'
        if num_train_sessions > 1:
            print('Load pretrain model from:', pretrain_model_path)
            # load weights that can be found in the pretrain model
            model.load_state_dict(torch.load(pretrain_model_path)['model'].state_dict(), strict=False)
        else:
            print('Train from scratch.')
        
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
            "stitching": config.model.encoder.stitching,
        }
        trainer = make_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            optimizer=optimizer,
            **trainer_kwargs,
            **meta_data
        )
        
        # train loop
        trainer.train()

    #########################

    if args.eval == "True":
        import wandb
        wandb.init(project=config.wandb.project, 
                entity=config.wandb.entity, 
                config=config, 
                name=f"eval_num_session_{num_train_sessions}_model_{args.model_name}_method_ssl_mask_{args.mask_mode}_stitch_True_{eid}"
                )
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
        
        if args.prompting == "True":
            model_config = f"src/configs/{model_acroynm}_stitching_prompting_eval.yaml"
        else:
            model_config = f"src/configs/{model_acroynm}_stitching_eval.yaml"
        
        configs = {
            'model_config': model_config,
            'model_path': f'{base_path}/results/finetune/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/model_best.pt',
            'trainer_config': f'src/configs/trainer_{model_acroynm}.yaml',
            'dataset_path': None, 
            'test_size': 0.2,
            'seed': 42,
            'mask_name': mask_name,
            'eid': eid,
            'stitching': True,
            'num_sessions': 1 
        }  
        
        
        # load your model and dataloader
        model, accelerator, dataset, dataloader = load_model_data_local(**configs)
        
        # co-smoothing
        if co_smooth:
            print('Start co-smoothing:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name, 
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/co_smooth',
                'mode': 'per_neuron',
                'n_time_steps': n_time_steps,    
                'is_aligned': True,
                'target_regions': None,
                'n_jobs': 8
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
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/forward_pred',
                'mode': 'forward_pred',
                'n_time_steps': n_time_steps,    
                'held_out_list': list(range(90, 100)), # NLB uses 200 ms for fp
                'is_aligned': True,
                'target_regions': None,
                'n_jobs': 8
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
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/inter_region',
                'mode': 'inter_region',
                'n_time_steps': n_time_steps,    
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 8
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
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/intra_region',
                'mode': 'intra_region',
                'n_time_steps': n_time_steps,    
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 8
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
                'model_config': model_config,
                'model_path': f'{base_path}/results/finetune/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/model_best.pt',
                'trainer_config': f'src/configs/ppwang/trainer_sl_choice_{model_acroynm}.yaml',
                'dataset_path': None,
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/choice_decoding',
                'test_size': 0.2,
                'seed': 42,
                'mask_name': mask_name,
                'metric': 'acc',
                'from_scratch': False,
                'freeze_encoder': True,
                'mask_ratio': args.mask_ratio,
                'eid': eid,
                'num_sessions': 1 ,
                'num_train_sessions': num_train_sessions,
                'use_logreg': True,
                'use_trial_filter': True,
            }  
            results = behavior_decoding(**configs)
            print(results)
            wandb.log(results)
        
        
        if continuous_decoding:
            print('Start continuous_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/finetune/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/model_best.pt',
                'trainer_config': f'src/configs/ppwang/trainer_sl_continuous_{model_acroynm}.yaml',
                'dataset_path': None, 
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/continuous_decoding',
                'test_size': 0.2,
                'seed': 42,
                'mask_name': mask_name,
                'metric': 'rsquared',
                'from_scratch': False,
                'freeze_encoder': True,
                'mask_ratio': args.mask_ratio,
                'eid': eid,
                'num_sessions': 1,
                'num_train_sessions': num_train_sessions,
                'use_logreg': True,
                'use_trial_filter': True
            }  
            results = behavior_decoding(**configs)
            print(results)
            wandb.log(results)

finally:
    if args.use_dummy:
        stop_dummy_load.set()
        dummy_thread.join()
    print('Finish model evaluation.')
    print('=======================')
    print('Finish model training.')
    print('=====================')
    

