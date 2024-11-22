import argparse
from math import ceil
import argparse
from accelerate import Accelerator
#from loader.data_loader import *
#from loader.rnn_data_loader import *
from loader.chaotic_rnn_loader import *
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
from models.ndt1_with_region_stitcher import NDT1_with_region_stitcher 
from models.mlp import NeuralMLP
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer
#from utils.eval_utils import load_model_data_local, co_smoothing_eval, behavior_decoding
from utils.eval_utils_rnn import load_model_data_local, co_smoothing_eval

import threading
import warnings
warnings.simplefilter("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--test_eid", type=int, default=[0,1,2])
#ap.add_argument("--test_eid", type=int, default=[0,1,2,3,4,5])
ap.add_argument("--mask_ratio", type=float, default=0.3)
ap.add_argument("--mask_mode", type=str, default="inter-region")
#ap.add_argument("--model_name", type=str, default="NDT1")
ap.add_argument("--model_name", type=str, default="NDT1_with_region_stitcher")
ap.add_argument("--prompting", type=str, default="False")
ap.add_argument("--train", type=str, default="True")
ap.add_argument("--eval", type=str, default="True")
ap.add_argument("--base_path", type=str, default='/mnt/smb/locker/miller-locker/users/jx2484/results/')
ap.add_argument("--num_train_sessions", type=int, default=3)
ap.add_argument('--use_dummy', action='store_true')
ap.add_argument('--region_channel_num', type=int, default=50)
args = ap.parse_args()

eids = args.test_eid
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
    elif args.model_name == 'NDT1_with_region_stitcher':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_stitching.yaml"
        }
    elif args.model_name == 'NDT2':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}.yaml"
        }

config = config_from_kwargs(kwargs)
config = update_config("src/configs/finetune_sessions_trainer.yaml", config)

config['model']['encoder']['embedder']['n_channels_per_region'] = args.region_channel_num

set_seed(config.seed)

max_space_length = config.data.max_space_length
meta_data = {}
meta_data['max_space_length'] = max_space_length
print('encoder max space length:', max_space_length)

#dataloader, num_neurons, datasets, areaoi_ind, area_ind_list_list = make_loader(eids, batch_size=12) # for test, the batch size is the total number of trials 
#dataloader, num_neurons, datasets, area_ind_list_list, record_info_list = make_rnn_loader(eids, batch_size=12)
dataloader, num_neurons, datasets, area_ind_list_list, record_info_list = make_chaotic_rnn_loader(eids, batch_size=12)

areaoi_ind = np.array([0,1,2])

meta_data['area_ind_list_list'] = area_ind_list_list
meta_data['areaoi_ind'] = areaoi_ind

meta_data['num_neurons'] = num_neurons
meta_data['num_sessions'] = len(eids)
meta_data['eids'] = eids

train_dataloader = dataloader['train']
val_dataloader = dataloader['val']
test_dataloader = dataloader['test']
#test_dataloader = dataloader['train']

print('check basic info of dataset')
print(record_info_list)

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

        log_dir = os.path.join(base_path, 
                            "results",
                            "finetune", 
                            "num_session_{}".format(num_train_sessions),
                            "model_{}".format(config.model.model_class), 
                            "method_{}".format(config.method.model_kwargs.method_name), 
                            "mask_{}".format(args.mask_mode),
                            "stitch_{}".format(config.model.encoder.stitching),
                            "{}".format(eids),
                            "region_factors_{}".format(args.region_channel_num)
                            )

        if not os.path.exists(log_dir):
            print(log_dir)
            os.makedirs(log_dir)

        if config.wandb.use:
            import wandb
            wandb.init(project=config.wandb.project, 
                    entity=config.wandb.entity, 
                    config=config, 
                    name="finetune_num_session_{}_model_{}_method_{}_mask_{}_stitch_{}_{}_region_factors_{}".format(num_train_sessions,config.model.model_class, config.method.model_kwargs.method_name,args.mask_mode, config.model.encoder.stitching,eids, args.region_channel_num)
                    )
        
        # Initialize the accelerator
        accelerator = Accelerator()
        
        # load model
        config = update_config(config, meta_data)

        NAME2MODEL = {"NDT1": NDT1, "NDT1_with_region_stitcher": NDT1_with_region_stitcher}

        model_class = NAME2MODEL[config.model.model_class]
        mlp = NeuralMLP(hidden_size=512, inter_size=512, act='relu', use_bias=True, dropout=0.1)
        model = model_class(config.model, **config.method.model_kwargs, **meta_data)
        print(model.encoder.layers)
        model.encoder.layers = nn.ModuleList([mlp])
        model = accelerator.prepare(model)
    
        # load pretrain model
        # pretrain_model_path = f'{base_path}/finetune/num_session_{num_train_sessions}/model_{config.model.model_class}/method_{config.method.model_kwargs.method_name}/mask_{args.mask_mode}/stitch_{config.model.encoder.stitching}/{eids}/region_factors_{args.region_channel_num}/model_best.pt'
        # print('load pretrain model from:', pretrain_model_path)
        # model.load_state_dict(torch.load(pretrain_model_path)['model'].state_dict(), strict=False)

        # if num_train_sessions > 1:
        #     print('Load pretrain model from:', pretrain_model_path)
        #     # load weights that can be found in the pretrain model
        #     model.load_state_dict(torch.load(pretrain_model_path)['model'].state_dict(), strict=False)
        # else:
        #     print('Train from scratch.')
        
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

#############EVALUATION######################

    if args.eval == "True":
        import wandb
        wandb.init(project=config.wandb.project, 
                entity=config.wandb.entity, 
                config=config, 
                name=f"eval_num_session_{num_train_sessions}_model_{args.model_name}_method_ssl_mask_{args.mask_mode}_stitch_True_{eids}_region_channel_{args.region_channel_num}"
                ) if config.wandb.use else None
        print('Start model evaluation.')
        print('=======================')
        
        mask_name = f"mask_{args.mask_mode}"
        if args.model_name == "NDT2":
            model_name = "STPatch"
        else:
            model_name = args.model_name
            
        n_time_steps = 400
        
        co_smooth = True
        forward_pred = False
        inter_region = True
        intra_region = False
        choice_decoding = False
        continuous_decoding = False
        
        print(mask_name)
        
        if args.prompting == "True":
            model_config = f"src/configs/{model_acroynm}_stitching_prompting_eval.yaml"
        else:
            model_config = f"src/configs/{model_acroynm}_stitching_eval.yaml"
        
        configs = {
            'region_channel_num': args.region_channel_num,
            'model_config': model_config,
            'model_path': f'{base_path}/finetune/num_session_{num_train_sessions}/model_{args.model_name}/method_ssl/{mask_name}/stitch_True/{eids}/region_factors_{args.region_channel_num}/model_best.pt',
            'trainer_config': f'src/configs/finetune_sessions_trainer.yaml',
            'dataset_path': None, 
            'test_size': 0.2,
            'seed': 42,
            'mask_name': mask_name,
            'eids': eids,
            'stitching': True,
            'num_sessions': 1,
        }  
        
        # load your model and dataloader
        model, accelerator = load_model_data_local(meta_data, **configs)
        
        # co-smoothing
        if co_smooth:
            print('Start co-smoothing:')
            co_smoothing_configs = {
                'subtract': 'task',
                #'onset_alignment': [50, 115, 235],
                'onset_alignment': [20, 40, 70, 80],
                'method_name': mask_name, 
                'save_path': f'{base_path}/eval/num_session_{num_train_sessions}/model_{args.model_name}/method_ssl/{mask_name}/stitch_True/{eids}/co_smooth/region_channel_{args.region_channel_num}',
                'mode': 'per_neuron',
                'n_time_steps': n_time_steps,    
                'is_aligned': True,
                'target_regions': None,
                'n_jobs': 8,
                'record_info_list': record_info_list, #added for rnn
                'eids': eids #added for rnn
            }
        
            results = co_smoothing_eval(model, 
                            accelerator, 
                            test_dataloader,  
                            **co_smoothing_configs)
            
            print(results)
            wandb.log(results) if config.wandb.use else None
        
        # forward prediction
        if forward_pred:
            print('Start forward prediction:')
            results = co_smoothing_configs = {
                'subtract': 'task',
                #'onset_alignment': [50, 115, 235],
                'onset_alignment': [20, 40, 70, 80],
                'method_name': mask_name, 
                'save_path': f'{base_path}/eval/num_session_{num_train_sessions}/model_{args.model_name}/method_ssl/{mask_name}/stitch_True/{eids}/forward_pred',
                'mode': 'forward_pred',
                'n_time_steps': n_time_steps,    
                'held_out_list': list(range(90, 100)), # NLB uses 200 ms for fp
                'is_aligned': True,
                'target_regions': None,
                'n_jobs': 8
            }
        
            results = co_smoothing_eval(model, 
                            accelerator, 
                            test_dataloader, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results) if config.wandb.use else None
                   
        # inter-region
        if inter_region:
            print('Start inter-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                #'onset_alignment': [50, 115, 235],
                'onset_alignment': [20, 40, 70, 80],
                'method_name': mask_name,
                'save_path': f'{base_path}/eval/num_session_{num_train_sessions}/model_{args.model_name}/method_ssl/{mask_name}/stitch_True/{eids}/inter_region/region_channel_{args.region_channel_num}',
                'mode': 'inter_region',
                'n_time_steps': n_time_steps,    
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 8,
                'record_info_list': record_info_list, #added for rnn
                'eids': eids #added for rnn
            }
        
            results = co_smoothing_eval(model, 
                            accelerator, 
                            test_dataloader, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results) if config.wandb.use else None

        # intra-region
        if intra_region:
            print('Start intra-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [50, 115, 235],
                'method_name': mask_name, 
                'save_path': f'{base_path}/eval/num_session_{num_train_sessions}/model_{args.model_name}/method_ssl/{mask_name}/stitch_True/{eids}/intra_region',
                'mode': 'intra_region',
                'n_time_steps': n_time_steps,    
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 8
            }
        
            results = co_smoothing_eval(model, 
                            accelerator, 
                            test_dataloader, 
                            **co_smoothing_configs)
            print(results)
            wandb.log(results) if config.wandb.use else None
        


finally:
    if args.use_dummy:
        stop_dummy_load.set()
        dummy_thread.join()
    print('Finish model evaluation.')
    print('=======================')
    print('Finish model training.')
    print('=====================')
    

