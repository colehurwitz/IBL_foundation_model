from utils.eval_utils import load_model_data_local, co_smoothing_eval, behavior_decoding
import warnings
warnings.simplefilter("ignore")

configs = {
    'model_config': 'src/configs/itransformer_multi.yaml',
    'model_path': None,
    'trainer_config': 'src/configs/trainer_iTransformer_zs.yaml',
    'mask_name': 'None', # no need for iTransformer
    'seed': 42,
    'eid': '71e55bfe-5a3a-4cba-bdc7-f085140d798e',
    'zero_shot': True
}

model, accelerator, dataset, dataloader = load_model_data_local(**configs)
print(model.encoder.attn_mode, model.masker.force_active)

####################################################
# Co-Smoothing
####################################################

co_smoothing_configs = {
    'subtract': 'task',
    'onset_alignment': [40],
    'method_name': 'test', # used for file name of figures
    'save_path': '/expanse/lustre/scratch/zwang34/temp_project/iTransformer/results/test_zs_30/co_smooth', # manually
    'mode': 'per_neuron',
    'n_time_steps': 100,
    'is_aligned': True,
    'target_regions': None
}

results = co_smoothing_eval(model, 
                    accelerator, 
                    dataloader, 
                    dataset, 
                    **co_smoothing_configs)
print(results)
    