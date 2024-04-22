from utils.eval_utils import load_model_data_local, co_smoothing_r2, compare_R2_scatter, co_smoothing_bps

mask_name = 'mask_random_causal'
model_name = 'NDT2'
n_time_steps = 50

print(mask_name)

# Configuration
configs = {
    'model_config': 'src/configs/ndt2.yaml',
    'model_path': f'/home/ppwang/IBL_foundation_model/checkpoints/train/single/model_{model_name}/method_ssl/{mask_name}/model_best.pt',
    'trainer_config': 'src/configs/ssl_session_trainer.yaml',
    'dataset_path': 'data/671c7ea7-6726-4fbe-adeb-f89c2c8e489b_aligned', 
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
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': mask_name, # used for file name of figures
    'save_path': f'results/eval/model_{model_name}/method_ssl/{mask_name}/co_smooth',
    'mode': 'per_neuron',
    'n_time_steps': n_time_steps,    
    'is_aligned': True,
    'target_regions': None
}

results = co_smoothing_r2(model, 
                accelerator, 
                dataloader, 
                dataset, 
                **co_smoothing_configs)
print(results)
wandb.log(results)


results = co_smoothing_bps(model, 
                 accelerator, 
                 dataloader, 
                 dataset, 
                 mode='per_neuron', 
                 save_path=f'results/eval/model_{model_name}/method_ssl/{mask_name}/co_smooth')
print(results)
wandb.log(results)

# forward prediction
results = co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [],
    'method_name': mask_name, # used for file name of figures
    'save_path': f'results/eval/model_{model_name}/method_ssl/{mask_name}/forward_pred',
    'mode': 'forward_pred',
    'n_time_steps': n_time_steps,    
    'held_out_list': list(range(n_time_steps//2 -10,n_time_steps//2)),
    'is_aligned': True,
    'target_regions': None
}

results = co_smoothing_r2(model, 
                accelerator, 
                dataloader, 
                dataset, 
                **co_smoothing_configs)
print(results)
wandb.log(results)

results = co_smoothing_bps(model, 
                 accelerator, 
                 dataloader, 
                 dataset, 
                 mode = 'forward_pred', 
                 held_out_list = list(range(n_time_steps//2 -10, n_time_steps//2)), 
                 save_path=f'results/eval/model_{model_name}/method_ssl/{mask_name}/forward_pred')
print(results)
wandb.log(results)

# inter-region
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': mask_name,
    'save_path': f'results/eval/model_{model_name}/method_ssl/{mask_name}/inter_region',
    'mode': 'inter_region',
    'n_time_steps': n_time_steps,    
    'held_out_list': None,
    'is_aligned': True,
    'target_regions': ['GRN']
}

results = co_smoothing_r2(model, 
                accelerator, 
                dataloader, 
                dataset, 
                **co_smoothing_configs)
print(results)
wandb.log(results)

results = co_smoothing_bps(model, 
                 accelerator, 
                 dataloader, 
                 dataset, 
                 mode = 'inter_region', 
                 held_out_list = None, 
                 target_regions = ['GRN'],
                 save_path=f'results/eval/model_{model_name}/method_ssl/{mask_name}/inter_region'
)
print(results)
wandb.log(results)

# intra-region
co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': mask_name, 
    'save_path': f'results/eval/model_{model_name}/method_ssl/{mask_name}/intra_region',
    'mode': 'intra_region',
    'n_time_steps': n_time_steps,    
    'held_out_list': [5,10,15,20,25,30,35,40,45,50],
    'is_aligned': True,
    'target_regions': ['GRN']
}

results = co_smoothing_r2(model, 
                accelerator, 
                dataloader, 
                dataset, 
                **co_smoothing_configs)
print(results)
wandb.log(results)

results = co_smoothing_bps(model, 
                 accelerator, 
                 dataloader, 
                 dataset, 
                 mode = 'intra_region', 
                 held_out_list = [5,10,15,20,25,30,35,40,45,50],
                 target_regions = ['GRN'], 
                 save_path = f'results/eval/model_{model_name}/method_ssl/{mask_name}/intra_region'
)
print(results)
wandb.log(results)


