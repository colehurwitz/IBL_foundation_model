from utils.eval_utils import load_model_data_local, co_smoothing_eval, behavior_decoding
import warnings
warnings.simplefilter("ignore")

mask_name = 'mask_all'
model_name = 'NDT1'
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
    'model_path': f'{base_path}/results/train/model_{model_name}/method_ssl/{mask_name}/model_best.pt',
    'trainer_config': 'src/configs/trainer.yaml',
    'dataset_path': None, 
    'test_size': 0.2,
    'seed': 42,
    'mask_name': mask_name,
}  

# init wandb
import wandb
wandb.init(project='ibl-ssl-eval', config=configs, name=f'model_{model_name}_method_ssl_{mask_name}')

# load your model and dataloader
model, accelerator, dataset, dataloader = load_model_data_local(**configs)

# co-smoothing
if co_smooth:
    print('Start co-smoothing:')
    co_smoothing_configs = {
        'subtract': 'task',
        'onset_alignment': [40],
        'method_name': mask_name, 
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/co_smooth',
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
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/forward_pred',
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
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/inter_region',
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
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/intra_region',
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
        'model_path': f'{base_path}/results/train/model_{model_name}/method_ssl/{mask_name}/model_best.pt',
        'trainer_config': 'src/configs/trainer_sl_choice.yaml',
        'dataset_path': '/home/exouser/Documents/IBL_foundation_model/data/671c7ea7-6726-4fbe-adeb-f89c2c8e489b_aligned',
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/choice_decoding',
        'test_size': 0.2,
        'seed': 42,
        'mask_name': mask_name,
        'metric': 'acc',
        'from_scratch': False,
        'freeze_encoder': False,
        'mask_ratio': 0.1
    }  
    results = behavior_decoding(**configs)
    print(results)
    wandb.log(results)


if continuous_decoding:
    print('Start continuous_decoding:')
    configs = {
        'model_config': 'src/configs/ndt1.yaml',
        'model_path': f'{base_path}/results/train/model_{model_name}/method_ssl/{mask_name}/model_best.pt',
        'trainer_config': 'src/configs/trainer_sl_continuous.yaml',
        'dataset_path': None, 
        'save_path': f'{base_path}/results/eval/model_{model_name}/method_ssl/{mask_name}/continuous_decoding',
        'test_size': 0.2,
        'seed': 42,
        'mask_name': mask_name,
        'metric': 'r2',
        'from_scratch': False,
        'freeze_encoder': False,
        'mask_ratio': 0.1
    }  
    results = behavior_decoding(**configs)
    print(results)
    wandb.log(results)
    
