import sys
import os
import importlib

# set your working dir
work_dir = '/home/exouser/Documents/IBL_foundation_model'
os.chdir(work_dir)
print('working dir: ', work_dir)

path = '/home/exouser/Documents/IBL_foundation_model'
sys.path.append(str(path))

from src.utils.eval_utils import load_model_data_local, co_smoothing_r2, compare_R2_scatter, co_smoothing_bps

mask_name = 'mask_temporal'

print(mask_name)

# Configuration
configs = {
    'model_config': 'src/configs/ndt1.yaml',
    'model_path': f'results/train/model_NDT1/method_ssl/{mask_name}/model_best.pt',
    'trainer_config': 'src/configs/trainer.yaml',
    'dataset_path': 'data/671c7ea7-6726-4fbe-adeb-f89c2c8e489b_aligned', 
    'test_size': 0.2,
    'seed': 42,
}  

# load your model and dataloader
model, accelerator, dataset, dataloader = load_model_data_local(**configs)

# co-smoothing
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': mask_name, # used for file name of figures
    'save_path': f'figs/model_NDT1/method_ssl/{mask_name}/co_smooth',
    'mode': 'per_neuron',
    'n_time_steps': 100,    
    'is_aligned': True,
    'target_regions': None
}

#co_smoothing_r2(model, accelerator, dataloader, dataset, **co_smoothing_configs)

co_smoothing_bps(model, accelerator, dataloader, dataset, mode='per_neuron', save_path=f'figs/model_NDT1/method_ssl/{mask_name}/co_smooth')

# forward prediction
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [],
    'method_name': mask_name, # used for file name of figures
    'save_path': f'figs/model_NDT1/method_ssl/{mask_name}/forward_pred',
    'mode': 'forward_pred',
    'n_time_steps': 100,    
    'held_out_list': list(range(90, 100)),
    'is_aligned': True,
    'target_regions': None
}

#co_smoothing_r2(model, accelerator, dataloader, dataset, **co_smoothing_configs)

co_smoothing_bps(model, accelerator, dataloader, dataset, mode = 'forward_pred', 
        held_out_list = list(range(90, 100)), save_path=f'figs/model_NDT1/method_ssl/{mask_name}/forward_pred')

# inter-region
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': mask_name,
    'save_path': f'figs/model_NDT1/method_ssl/{mask_name}/inter_region',
    'mode': 'inter_region',
    'n_time_steps': 100,    
    'held_out_list': None,
    'is_aligned': True,
    'target_regions': ['GRN']
}

#co_smoothing_r2(model, accelerator, dataloader, dataset, **co_smoothing_configs)

co_smoothing_bps(model, accelerator, dataloader, dataset, mode = 'inter_region', held_out_list = None, target_regions = ['GRN'],
    save_path=f'figs/model_NDT1/method_ssl/{mask_name}/inter_region'
)

# intra-region
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': mask_name, 
    'save_path': f'figs/model_NDT1/method_ssl/{mask_name}/intra_region',
    'mode': 'intra_region',
    'n_time_steps': 100,    
    'held_out_list': [5,10,15,20,25,30,35,40,45,50],
    'is_aligned': True,
    'target_regions': ['GRN']
}

#co_smoothing_r2(model, accelerator, dataloader, dataset, **co_smoothing_configs)

co_smoothing_bps(model, accelerator, dataloader, dataset, mode = 'intra_region', held_out_list = [5,10,15,20,25,30,35,40,45,50],
    target_regions = ['GRN'], save_path = f'figs/model_NDT1/method_ssl/{mask_name}/intra_region'
)


