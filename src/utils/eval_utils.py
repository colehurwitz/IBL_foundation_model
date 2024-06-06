import pickle
from math import ceil
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.dataset_utils import load_ibl_dataset
from utils.utils import set_seed, move_batch_to_device, plot_gt_pred, metrics_list, plot_avg_rate_and_spike, \
    plot_rate_and_spike
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
from models.stpatch import STPatch
from models.itransformer import iTransformer
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import r2_score
from scipy.special import gammaln
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.colors as colors
import os
from trainer.make import make_trainer
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer}

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Model/Dataset Loading and Configuration
# --------------------------------------------------------------------------------------------------

def load_model_data_local(**kwargs):
    model_config = kwargs['model_config']
    trainer_config = kwargs['trainer_config']
    model_path = kwargs['model_path']
    dataset_path = kwargs['dataset_path']
    test_size = kwargs['test_size']
    seed = kwargs['seed']
    mask_name = kwargs['mask_name']
    mask_mode = mask_name.split("_")[1]
    eid = kwargs['eid']
    stitching = kwargs['stitching']
    num_sessions = kwargs['num_sessions']
    use_nemo = kwargs['use_nemo']

    # set seed
    set_seed(seed)

    # load the model
    config = config_from_kwargs({"model": f"include:{model_config}"})
    config = update_config(model_config, config)
    config = update_config(trainer_config, config)
    config.model.encoder.masker.mode = mask_mode

    accelerator = Accelerator()

    _,_,_, meta_data = load_ibl_dataset(
                            cache_dir=config.dirs.dataset_cache_dir,
                            user_or_org_name=config.dirs.huggingface_org,
                            num_sessions=1,
                            split_method="predefined",
                            test_session_eid=[],
                            batch_size=config.training.train_batch_size,
                            seed=seed,
                            eid=eid
                        )
    print(meta_data)

    # load the dataset
    r_dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
    dataset = r_dataset["test"]

    if use_nemo:
        neuron_uuids = np.array(dataset['cluster_uuids'][0]).astype('str')
        with open('data/MtM_unit_embed.pkl','rb') as file:
            nemo_data = pickle.load(file)
        nemo_uuids = nemo_data['uuids']
        include_uuids = np.intersect1d(neuron_uuids, nemo_uuids)
        n_neurons = len(include_uuids)
        print('Use NEMO cell-type embeddings.')
    else:
        n_neurons = len(dataset['cluster_regions'][0])

    if config.model.model_class == "iTransformer" and config.model.encoder.embed_region:
        config["model"]["encoder"]["neuron_regions"] = list(
            set(str(b) for a in [row["cluster_regions"] for rows in r_dataset.values() for row in rows] for b in a)
        )
        
    if config.model.model_class in ["NDT1", "iTransformer"]:
        max_space_length = n_neurons  
    elif config.model.model_class == "STPatch":
        max_space_F = config.model.encoder.embedder.max_space_F
        max_space_length = ceil(n_neurons/max_space_F) * max_space_F
    else:
        max_space_length = config.data.max_space_length

    print('encoder max space length:', max_space_length)

    meta_data['max_space_length'] = max_space_length
    meta_data['num_neurons'] = [n_neurons]
    print(meta_data)

    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model, **config.method.model_kwargs, **meta_data)    

    if config.model.model_class == 'iTransformer':
        model.load_checkpoint(model_path)
        model.masker.mode = mask_mode
        model.masker.ratio = 0
        model.masker.force_active = False
        print("(eval) masking mode: ", model.masker.mode)
        print("(eval) masking ratio: ", model.masker.ratio)
        print("(eval) masking active: ", model.masker.force_active)
    else:
        model = torch.load(model_path)['model']
        model.encoder.masker.mode = mask_mode
        model.encoder.masker.force_active = False
        print("(eval) masking mode: ", model.encoder.masker.mode)
        print("(eval) masking ratio: ", model.encoder.masker.ratio)
        print("(eval) masking active: ", model.encoder.masker.force_active)
        if 'causal' in mask_name:
            model.encoder.context_forward = 0
            print("(behave decoding) context forward: ", model.encoder.context_forward)
    
    model = accelerator.prepare(model)

    dataloader = make_loader(
        dataset,
        target=config.data.target,
        batch_size=len(dataset),
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False,
    )

    # check the shape of the dataset
    for batch in dataloader:
        print('spike data shape: {}'.format(batch['spikes_data'].shape))
        break

    return model, accelerator, dataset, dataloader

# --------------------------------------------------------------------------------------------------
# Evaluation
# 1. Co-smoothing_eval (R2, co-bps, and shuqi's plot) 
# 2. Behavior_decoding (choice, wheel speed)
# 3. R2 scatter plot
# --------------------------------------------------------------------------------------------------

def co_smoothing_eval(
        model,
        accelerator,
        test_dataloader,
        test_dataset,
        n=1,
        **kwargs
):
    assert n == 1, 'only support n=1 now'

    for batch in test_dataloader:
        break

    method_name = kwargs['method_name']
    mode = kwargs['mode']
    is_aligned = kwargs['is_aligned']
    target_regions = kwargs['target_regions']
    n_jobs = kwargs['n_jobs']

    # hack to accommodate NDT2 - fix later 
    if sum(batch['space_attn_mask'][0] == 0) == 0:
        tot_num_neurons = batch['space_attn_mask'].size()[-1]
    else:
        tot_num_neurons = (batch['space_attn_mask'][0] == 0).nonzero().min().item() 
    uuids_list = np.array(test_dataset['cluster_uuids'][0])[:tot_num_neurons]
    region_list = np.array(test_dataset['cluster_regions'])[0][:tot_num_neurons]

    T = kwargs['n_time_steps']
    N = uuids_list.shape[0]

    if is_aligned:
        
        # prepare the condition matrix
        b_list = []
    
        # choice
        choice = np.array(test_dataset['choice'])
        choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, T))
        b_list.append(choice)
    
        # reward
        reward = np.array(test_dataset['reward'])
        reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, T))
        b_list.append(reward)
    
        # block
        block = np.array(test_dataset['block'])
        block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, T))
        b_list.append(block)
    
        behavior_set = np.stack(b_list, axis=-1)
    
        var_name2idx = {'block': [2],
                        'choice': [0],
                        'reward': [1],
                        'wheel': [3],
                        }
        var_value2label = {'block': {(0.2,): "p(left)=0.2",
                                     (0.5,): "p(left)=0.5",
                                     (0.8,): "p(left)=0.8", },
                           'choice': {(-1.0,): "right",
                                      (1.0,): "left"},
                           'reward': {(0.,): "no reward",
                                      (1.,): "reward", }}
        var_tasklist = ['block', 'choice', 'reward']
        var_behlist = []

    if mode == 'per_neuron':
        
        bps_result_list, r2_result_list = [float('nan')] * tot_num_neurons, [np.array([np.nan, np.nan])] * N
        # loop through all the neurons
        counter = 0
        for n_i in tqdm(range(0, tot_num_neurons+n_jobs, n_jobs)):    
            if counter >= tot_num_neurons:
                break
            gt_spikes_lst, mask_spikes_lst, eval_mask_lst = [], [], []
            time_attn_mask_lst, space_attn_mask_lst, spikes_timestamps_lst, spikes_spacestamps_lst, targets_lst, neuron_regions_lst, nemo_rep_lst = [], [], [], [], [], [], []
            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    gt_spike_data = batch['spikes_data'].clone()
                    for i in range(n_jobs):
                        counter += 1
                        if counter <= tot_num_neurons:
                            mask_result = heldout_mask(
                                batch['spikes_data'].clone(),
                                mode='manual',
                                heldout_idxs=np.array([n_i+i])
                            )
                            mask_spikes_lst.append(mask_result['spikes'])
                            eval_mask_lst.append(mask_result['eval_mask'])
                            gt_spikes_lst.append(gt_spike_data)
                            time_attn_mask_lst.append(batch['time_attn_mask'])
                            space_attn_mask_lst.append(batch['space_attn_mask'])
                            spikes_timestamps_lst.append(batch['spikes_timestamps'])
                            spikes_spacestamps_lst.append(batch['spikes_spacestamps'])
                            targets_lst.append(batch['target'])
                            neuron_regions_lst.append(batch['neuron_regions'])
                            nemo_rep_lst.append(batch['nemo_rep'])
                        else:
                            break

                    try:
                        masking_mode = 'neuron' if model.use_prompt else model.encoder.masker.mode
                        model.encoder.mask = False
                    except AttributeError:
                        masking_mode = 'neuron' if model.use_prompt else model.masker.mode
                        model.mask = False
                    
                    outputs = model(
                        torch.cat(mask_spikes_lst, 0),
                        time_attn_mask=torch.cat(time_attn_mask_lst, 0),
                        space_attn_mask=torch.cat(space_attn_mask_lst, 0),
                        spikes_timestamps=torch.cat(spikes_timestamps_lst, 0), 
                        spikes_spacestamps=torch.cat(spikes_spacestamps_lst, 0), 
                        targets = torch.cat(targets_lst, 0),
                        neuron_regions=np.stack(neuron_regions_lst, axis=-1).squeeze(),
                        eval_mask=torch.cat(eval_mask_lst, 0),
                        masking_mode = masking_mode,
                        num_neuron=batch['spikes_data'].shape[2],
                        eid=batch['eid'][0],
                        nemo_rep=torch.cat(nemo_rep_lst, 0)
                    )
            outputs.preds = torch.exp(outputs.preds)
    
            gt_spikes = torch.cat(gt_spikes_lst, 0).detach().cpu().numpy()
            pred_spikes = outputs.preds.detach().cpu().numpy()
            tot_num_trials = len(batch['spikes_data'])

            # compute co-bps
            for i in range(n_jobs):
                if n_i+i < tot_num_neurons:
                    gt_held_out = gt_spikes[i*tot_num_trials:(i+1)*tot_num_trials, :, [n_i+i]]
                    pred_held_out = pred_spikes[i*tot_num_trials:(i+1)*tot_num_trials, :, [n_i+i]]
        
                    bps = bits_per_spike(pred_held_out, gt_held_out)
                    if np.isinf(bps):
                        bps = np.nan
                    bps_result_list[n_i+i] = bps
        
                    # compute R2
                    if is_aligned:
                        X = behavior_set  # [#trials, #timesteps, #variables]
                        _r2_psth, _r2_trial = viz_single_cell(X, gt_held_out.squeeze(), pred_held_out.squeeze(),
                                                              var_name2idx, var_tasklist, var_value2label, var_behlist,
                                                              subtract_psth=kwargs['subtract'],
                                                              aligned_tbins=kwargs['onset_alignment'],
                                                              neuron_idx=uuids_list[n_i+i][:4],
                                                              neuron_region=region_list[n_i+i],
                                                              method=method_name, save_path=kwargs['save_path'])
                        r2_result_list[n_i+i] = np.array([_r2_psth, _r2_trial])
                    else:
                        r2 = viz_single_cell_unaligned(
                            gt_held_out.squeeze(), pred_held_out.squeeze(), 
                            neuron_idx=uuids_list[n_i+i][:4],
                            neuron_region=region_list[n_i+i],
                            method=method_name, save_path=kwargs['save_path']
                        )
                        r2_result_list[n_i+i] = r2
                else:
                    break

    elif mode == 'forward_pred':

        held_out_list = kwargs['held_out_list']

        assert held_out_list is not None, 'forward_pred requires specific target time points to predict'
        target_regions = neuron_regions = None
        held_out_list = [held_out_list]

        bps_result_list, r2_result_list = [float('nan')] * tot_num_neurons, [np.array([np.nan, np.nan])] * N
        for hd_idx in held_out_list:
           
            hd = np.array([hd_idx])

            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    gt_spike_data = batch['spikes_data'].clone()
                    mask_result = heldout_mask(
                        batch['spikes_data'].clone(),
                        mode=mode,
                        heldout_idxs=hd,
                        target_regions=target_regions,
                        neuron_regions=region_list
                    )  
                    
                    try:
                        masking_mode = 'causal' if model.use_prompt else model.encoder.masker.mode
                        model.encoder.mask = False
                    except AttributeError:
                        masking_mode = 'causal' if model.use_prompt else model.masker.mode
                        model.mask = False
                    
                    outputs = model(
                        mask_result['spikes'],
                        time_attn_mask=batch['time_attn_mask'],
                        space_attn_mask=batch['space_attn_mask'],
                        spikes_timestamps=batch['spikes_timestamps'], 
                        spikes_spacestamps=batch['spikes_spacestamps'], 
                        targets = batch['target'],
                        neuron_regions=batch['neuron_regions'],
                        eval_mask=mask_result['eval_mask'],
                        masking_mode=masking_mode,
                        num_neuron=batch['spikes_data'].shape[2],
                        eid=batch['eid'][0],
                        nemo_rep=batch['nemo_rep']
                    )
            outputs.preds = torch.exp(outputs.preds)
        
            gt_spikes = gt_spike_data.detach().cpu().numpy()
            pred_spikes = outputs.preds.detach().cpu().numpy()
    
            target_neuron_idxs = np.arange(tot_num_neurons)
            target_time_idxs = held_out_list[0]

            # compute co-bps
            gt_held_out = gt_spikes[:, target_time_idxs][:,:,target_neuron_idxs]
            pred_held_out = pred_spikes[:, target_time_idxs][:,:,target_neuron_idxs]
    
            for n_i in tqdm(range(len(target_neuron_idxs)), desc='co-bps'): 
                bps = bits_per_spike(pred_held_out[:,:,[n_i]], gt_held_out[:,:,[n_i]])
                if np.isinf(bps):
                    bps = np.nan
                bps_result_list[target_neuron_idxs[n_i]] = bps

            # compute R2
            ys = gt_spikes[:, target_time_idxs]
            y_preds = pred_spikes[:, target_time_idxs]
    
            # choose the neuron to plot
            idxs = target_neuron_idxs
    
            for i in tqdm(range(idxs.shape[0]), desc='R2'):
                if is_aligned:
                    X = behavior_set[:, target_time_idxs, :]  # [#trials, #timesteps, #variables]
                    _r2_psth, _r2_trial = viz_single_cell(X, ys[:, :, idxs[i]], y_preds[:, :, idxs[i]],
                                                          var_name2idx, var_tasklist, var_value2label, var_behlist,
                                                          subtract_psth=kwargs['subtract'],
                                                          aligned_tbins=[],
                                                          neuron_idx=uuids_list[idxs[i]][:4],
                                                          neuron_region=region_list[idxs[i]],
                                                          method=method_name, save_path=kwargs['save_path']);
                    r2_result_list[idxs[i]] = np.array([_r2_psth, _r2_trial])
                else:
                    r2 = viz_single_cell_unaligned(
                        ys[:, :, idxs[i]], y_preds[:, :, idxs[i]], 
                        neuron_idx=uuids_list[idxs[i]][:4],
                        neuron_region=region_list[idxs[i]],
                        method=method_name, save_path=kwargs['save_path']
                    )
                    r2_result_list[idxs[i]] = r2

    elif mode == 'inter_region':

        if 'all' in target_regions:
            target_regions = list(np.unique(region_list))
            
        held_out_list = kwargs['held_out_list']

        assert held_out_list is None, 'inter_region does LOO for all neurons in the target region'

        bps_result_list, r2_result_list = [float('nan')] * tot_num_neurons, [np.array([np.nan, np.nan])] * N
        for region in tqdm(target_regions, desc='region'):
            print(region)
            hd = np.argwhere(region_list==region).flatten() 
            held_out_list = np.arange(len(hd))
            held_out_list = [held_out_list]   
            hd = np.array([held_out_list]).flatten()

            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    gt_spike_data = batch['spikes_data'].clone()
                    mask_result = heldout_mask(
                        batch['spikes_data'].clone(),
                        mode=mode,
                        heldout_idxs=hd,
                        target_regions=[region],
                        neuron_regions=region_list
                    )              

                    try:
                        masking_mode = 'inter-region' if model.use_prompt else model.encoder.masker.mode
                        model.encoder.mask = False
                    except AttributeError:
                        masking_mode = 'inter-region' if model.use_prompt else model.masker.mode
                        model.mask = False
                    
                    outputs = model(
                        mask_result['spikes'],
                        time_attn_mask=batch['time_attn_mask'],
                        space_attn_mask=batch['space_attn_mask'],
                        spikes_timestamps=batch['spikes_timestamps'], 
                        spikes_spacestamps=batch['spikes_spacestamps'], 
                        targets = batch['target'],
                        neuron_regions=batch['neuron_regions'],
                        eval_mask=mask_result['eval_mask'],
                        masking_mode=masking_mode,
                        num_neuron=batch['spikes_data'].shape[2],
                        eid=batch['eid'][0],
                        nemo_rep=batch['nemo_rep']
                    )
            outputs.preds = torch.exp(outputs.preds)
        
            gt_spikes = gt_spike_data.detach().cpu().numpy()
            pred_spikes = outputs.preds.detach().cpu().numpy()
    
            target_neuron_idxs = mask_result['heldout_idxs']
            target_time_idxs = np.arange(gt_spikes.shape[1])

            # compute co-bps
            gt_held_out = gt_spikes[:, target_time_idxs][:,:,target_neuron_idxs]
            pred_held_out = pred_spikes[:, target_time_idxs][:,:,target_neuron_idxs]
    
            for n_i in range(len(target_neuron_idxs)): 
                bps = bits_per_spike(pred_held_out[:,:,[n_i]], gt_held_out[:,:,[n_i]])
                if np.isinf(bps):
                    bps = np.nan
                bps_result_list[target_neuron_idxs[n_i]] = bps

            # compute R2
            ys = gt_spikes[:, target_time_idxs]
            y_preds = pred_spikes[:, target_time_idxs]
    
            # choose the neuron to plot
            idxs = target_neuron_idxs
            for i in range(idxs.shape[0]):
                if is_aligned:
                    X = behavior_set[:, target_time_idxs, :]  # [#trials, #timesteps, #variables]
                    _r2_psth, _r2_trial = viz_single_cell(X, ys[:, :, idxs[i]], y_preds[:, :, idxs[i]],
                                                          var_name2idx, var_tasklist, var_value2label, var_behlist,
                                                          subtract_psth=kwargs['subtract'],
                                                          aligned_tbins=[],
                                                          neuron_idx=uuids_list[idxs[i]][:4],
                                                          neuron_region=region_list[idxs[i]],
                                                          method=method_name, save_path=kwargs['save_path']);
                    r2_result_list[idxs[i]] = np.array([_r2_psth, _r2_trial])
                else:
                    r2 = viz_single_cell_unaligned(
                        ys[:, :, idxs[i]], y_preds[:, :, idxs[i]], 
                        neuron_idx=uuids_list[idxs[i]][:4],
                        neuron_region=region_list[idxs[i]],
                        method=method_name, save_path=kwargs['save_path']
                    )
                    r2_result_list[idxs[i]] = r2
                        
    elif mode == 'intra_region':

        if 'all' in target_regions:
            target_regions = list(np.unique(region_list))
            
        held_out_list = kwargs['held_out_list']
        assert held_out_list is None, 'intra_region does LOO for all neurons in the target region'

        bps_result_list, r2_result_list = [float('nan')] * tot_num_neurons, [np.array([np.nan, np.nan])] * N
        for region in tqdm(target_regions, desc='region'):
            print(region)
            target_neuron_idxs = np.argwhere(region_list==region).flatten() 
            held_out_list = list(range(0, len(target_neuron_idxs)+n_jobs, n_jobs))

            for hd_idx in held_out_list:
                
                if hd_idx >= len(target_neuron_idxs):
                    break

                gt_spikes_lst, mask_spikes_lst, eval_mask_lst, heldout_idxs_lst = [], [], [], []
                time_attn_mask_lst, space_attn_mask_lst, spikes_timestamps_lst, spikes_spacestamps_lst, targets_lst, neuron_regions_lst, nemo_rep_lst = [], [], [], [], [], [], []
    
                model.eval()
                with torch.no_grad():
                    for batch in test_dataloader:
                        batch = move_batch_to_device(batch, accelerator.device)
                        gt_spike_data = batch['spikes_data'].clone()
                        for i in range(n_jobs):
                            if hd_idx+i < len(target_neuron_idxs):
                                mask_result = heldout_mask(
                                    batch['spikes_data'].clone(),
                                    mode=mode,
                                    heldout_idxs=np.array([hd_idx+i]).flatten(),
                                    target_regions=[region],
                                    neuron_regions=region_list
                                )   
                                mask_spikes_lst.append(mask_result['spikes'])
                                eval_mask_lst.append(mask_result['eval_mask'])
                                heldout_idxs_lst.append(mask_result['heldout_idxs'])
                                gt_spikes_lst.append(gt_spike_data)
                                time_attn_mask_lst.append(batch['time_attn_mask'])
                                space_attn_mask_lst.append(batch['space_attn_mask'])
                                spikes_timestamps_lst.append(batch['spikes_timestamps'])
                                spikes_spacestamps_lst.append(batch['spikes_spacestamps'])
                                targets_lst.append(batch['target'])
                                neuron_regions_lst.append(batch['neuron_regions'])
                                nemo_rep_lst.append(batch['nemo_rep'])
                            else:
                                break

                        try:
                            masking_mode = 'intra-region' if model.use_prompt else model.encoder.masker.mode
                            model.encoder.mask = False
                        except AttributeError:
                            masking_mode = 'intra-region' if model.use_prompt else model.masker.mode
                            model.mask = False
                        
                        outputs = model(
                            torch.cat(mask_spikes_lst, 0),
                            time_attn_mask=torch.cat(time_attn_mask_lst, 0),
                            space_attn_mask=torch.cat(space_attn_mask_lst, 0),
                            spikes_timestamps=torch.cat(spikes_timestamps_lst, 0), 
                            spikes_spacestamps=torch.cat(spikes_spacestamps_lst, 0), 
                            targets = torch.cat(targets_lst, 0),
                            neuron_regions=np.stack(neuron_regions_lst, axis=-1).squeeze(),
                            eval_mask=torch.cat(eval_mask_lst, 0),
                            masking_mode=masking_mode,
                            num_neuron=batch['spikes_data'].shape[2],
                            eid=batch['eid'][0],
                            nemo_rep=torch.cat(nemo_rep_lst, 0)
                        )
                outputs.preds = torch.exp(outputs.preds)
            
                gt_spikes = torch.cat(gt_spikes_lst, 0).detach().cpu().numpy()
                pred_spikes = outputs.preds.detach().cpu().numpy()
                tot_num_trials = len(batch['spikes_data'])

                heldout_idxs = np.stack(heldout_idxs_lst).flatten()
                
                for i in range(len(heldout_idxs)):
                    gt_held_out = gt_spikes[i*tot_num_trials:(i+1)*tot_num_trials, :, [heldout_idxs[i]]]
                    pred_held_out = pred_spikes[i*tot_num_trials:(i+1)*tot_num_trials, :, [heldout_idxs[i]]]
    
                    bps = bits_per_spike(pred_held_out, gt_held_out)
                    if np.isinf(bps):
                        bps = np.nan
                    bps_result_list[heldout_idxs[i]] = bps

                    if is_aligned:
                        X = behavior_set  # [#trials, #timesteps, #variables]
                        _r2_psth, _r2_trial = viz_single_cell(X, gt_held_out.squeeze(), pred_held_out.squeeze(),
                                                              var_name2idx, var_tasklist, var_value2label, var_behlist,
                                                              subtract_psth=kwargs['subtract'],
                                                              aligned_tbins=[],
                                                              neuron_idx=uuids_list[heldout_idxs[i]][:4],
                                                              neuron_region=region_list[heldout_idxs[i]],
                                                              method=method_name, save_path=kwargs['save_path']);
                        r2_result_list[heldout_idxs[i]] = np.array([_r2_psth, _r2_trial])
                    else:
                        r2 = viz_single_cell_unaligned(
                            gt_held_out.squeeze(), pred_held_out.squeeze(),
                            neuron_idx=uuids_list[heldout_idxs[i]][:4],
                            neuron_region=region_list[heldout_idxs[i]],
                            method=method_name, save_path=kwargs['save_path']
                        )
                        r2_result_list[heldout_idxs[i]] = r2
    else:
        raise NotImplementedError('mode not implemented')

    # save co-bps
    os.makedirs(kwargs['save_path'], exist_ok=True)
    bps_all = np.array(bps_result_list)
    bps_mean = np.nanmean(bps_all)
    bps_std = np.nanstd(bps_all)
    plt.hist(bps_all, bins=30, alpha=0.75, color='red', edgecolor='black')
    plt.xlabel('bits per spike')
    plt.ylabel('count')
    plt.title('Co-bps distribution\n mean: {:.2f}, std: {:.2f}\n # non-zero neuron: {}'.format(bps_mean, bps_std, len(bps_all)));
    plt.savefig(os.path.join(kwargs['save_path'], f'bps.png'), dpi=200)
    np.save(os.path.join(kwargs['save_path'], f'bps.npy'), bps_all)
    
    # save R2
    r2_all = np.array(r2_result_list)
    np.save(os.path.join(kwargs['save_path'], f'r2.npy'), r2_all)

    return {
        f"{mode}_mean_bps": bps_mean,
        f"{mode}_std_bps": bps_std,
        f"{mode}_mean_r2_psth": np.nanmean(r2_all[:, 0]),
        f"{mode}_std_r2_psth": np.nanstd(r2_all[:, 0]),
        f"{mode}_mean_r2_trial": np.nanmean(r2_all[:, 1]),
        f"{mode}_std_r2_trial": np.nanstd(r2_all[:, 1])
    }


def draw_threshold_table(
        mask_methods: list,
        eval_methods: list,
        load_path: str,
        firing_rate_ts: list,  # firing rate threshold
        quality_ts: list,  # quality threshold
):
    # clear_output(wait=True)

    save_path = Path(load_path)

    # there should be a mean_rates file.
    fr = np.load(save_path / 'mean_rates.npy')
    ts_idx = (firing_rate_ts[0] <= fr) & (fr <= firing_rate_ts[1])

    print(sum(ts_idx))

    metrics_dict = {}
    for mask in mask_methods:
        metrics_dict[mask] = {}
        for eval in eval_methods:
            metrics_dict[mask][eval] = {}
            try:
                r_r2 = np.load(save_path / mask / eval / 'r2.npy')
            except:
                r_r2 = np.zeros((1, 2))
            try:
                r_bps = np.load(save_path / mask / eval / 'bps.npy')
            except:
                r_bps = 0

            r2 = r_r2[ts_idx, :]
            bps = r_bps[ts_idx]
            # print(r2.shape, bps.shape)

            metrics_dict[mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) if np.nanmean(r2.T[0]) > -10 else -5
            metrics_dict[mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) if np.nanmean(r2.T[1]) > -10 else -5
            metrics_dict[mask][eval]['bps'] = np.nanmean(bps) if np.nanmean(bps) > -10 else -5
            # print(metrics_dict[mask][eval]['r2_psth'], metrics_dict[mask][eval]['r2_per_trial'], metrics_dict[mask][eval]['bps'])


    N = len(mask_methods)
    K = len(eval_methods)
    r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
    for i, mask in enumerate(mask_methods):
        for j, eval in enumerate(eval_methods):
            r2_psth_mat[i, j] = metrics_dict[mask][eval]['r2_psth']
            r2_per_trial_mat[i, j] = metrics_dict[mask][eval]['r2_per_trial']
            bps_mat[i, j] = metrics_dict[mask][eval]['bps']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    mat = bps_mat
    im0 = axes[0].imshow(mat, cmap='RdYlGn')
    axes[0].set_title("bps")

    for i in range(len(mask_methods)):
        for j in range(len(eval_methods)):
            color = 'w' if mat[i, j] < 0.5 else 'k'
            text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                                ha="center", va="center", color=color, fontsize=12)

    mat = r2_psth_mat
    im1 = axes[1].imshow(mat, cmap='RdYlGn')
    axes[1].set_title("r2 psth")

    for i in range(len(mask_methods)):
        for j in range(len(eval_methods)):
            color = 'w' if mat[i, j] < 0.5 else 'k'
            text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                                ha="center", va="center", color=color, fontsize=12)

    mat = r2_per_trial_mat
    # print(mat)
    im2 = axes[2].imshow(mat, cmap='RdYlGn')
    axes[2].set_title("r2 per trial")

    for i in range(len(mask_methods)):
        for j in range(len(eval_methods)):
            color = 'w' if mat[i, j] < 0.5 else 'k'
            text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                                ha="center", va="center", color=color, fontsize=12)

    for ax in axes:
        ax.set_yticks(np.arange(N),
                      labels=mask_methods)  # local
        ax.set_xticks(np.arange(K), labels=eval_methods)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    fig.tight_layout()
    # display(fig)
    plt.savefig('figs/table/metrics.png')


def behavior_decoding(**kwargs):
    model_config = kwargs['model_config']
    trainer_config = kwargs['trainer_config']
    model_path = kwargs['model_path']
    dataset_path = kwargs['dataset_path']
    test_size = kwargs['test_size']
    seed = kwargs['seed']
    mask_name = kwargs['mask_name']
    metric = kwargs['metric']
    mask_ratio = kwargs['mask_ratio']
    mask_mode = mask_name.split("_")[1]
    eid = kwargs['eid']
    num_sessions = kwargs['num_sessions']
    target = kwargs['target']
    use_trial_filter = kwargs['use_trial_filter']
    use_nemo = kwargs['use_nemo']

    # set seed
    set_seed(seed)

    # load the model
    config = config_from_kwargs({"model": f"include:{model_config}"})
    config = update_config(model_config, config)
    config = update_config(trainer_config, config)
    config.model.encoder.masker.mode = mask_mode

    accelerator = Accelerator()

    _,_,_, meta_data = load_ibl_dataset(
                        cache_dir=config.dirs.dataset_cache_dir,
                        user_or_org_name=config.dirs.huggingface_org,
                        num_sessions=1,
                        split_method="predefined",
                        test_session_eid=[],
                        batch_size=config.training.train_batch_size,
                        seed=seed,
                        eid=eid
                    )
    print(meta_data)

    # load the dataset
    dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    if use_nemo:
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

    if use_trial_filter:
        # load the trial filter
        print(os.path.join('data', 'trials_mask', f'{eid}.npy'))
        trial_filter = np.load(os.path.join('data', 'trials_mask', f'{eid}.npy'), allow_pickle=True).item()
        train_filter,val_filter, test_filter = trial_filter['train'], trial_filter['val'], trial_filter['test']
        assert len(train_filter) == len(train_dataset), f"eid {eid} train filter length mismatch, expected {len(train_dataset)}, got {len(train_filter)}"
        assert len(val_filter) == len(val_dataset), f"eid {eid} val filter length mismatch, expected {len(val_dataset)}, got {len(val_filter)}"
        assert len(test_filter) == len(test_dataset), f"eid {eid} test filter length mismatch, expected {len(test_dataset)}, got {len(test_filter)}"
        train_filter_idx = np.where(train_filter==1)[0]
        val_filter_idx = np.where(val_filter==1)[0]
        test_filter_idx = np.where(test_filter==1)[0]
        train_dataset = train_dataset.select(train_filter_idx)
        val_dataset = val_dataset.select(val_filter_idx)
        test_dataset = test_dataset.select(test_filter_idx)
        print(f"Filtered trials: train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}")

    if config.model.model_class in ["NDT1", "iTransformer"]:
        max_space_length = n_neurons  
    elif config.model.model_class == "STPatch":
        max_space_F = config.model.encoder.embedder.max_space_F
        max_space_length = ceil(n_neurons/max_space_F) * max_space_F
    else:
        max_space_length = config.data.max_space_length

    print('encoder max space length:', max_space_length)

    meta_data['max_space_length'] = max_space_length 
    meta_data['num_neurons'] = [n_neurons]
    print(meta_data)

    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model, **config.method.model_kwargs, **meta_data)

    if config.model.model_class == 'iTransformer':
        model.load_checkpoint(model_path)
        model.masker.mode = mask_mode
        model.masker.ratio = mask_ratio
        print("(behave decoding) masking mode: ", model.masker.mode)
        print("(behave decoding) masking ratio: ", model.masker.ratio)
        print("(behave decoding) masking active: ", model.masker.force_active)
    else:
        model = torch.load(model_path)['model']
        model.encoder.masker.mode = mask_mode
        model.encoder.masker.ratio = mask_ratio
        print("(behave decoding) masking mode: ", model.encoder.masker.mode)
        print("(behave decoding) masking ratio: ", model.encoder.masker.ratio)
        print("(behave decoding) masking active: ", model.encoder.masker.force_active)
        if args.mask_mode == 'causal':
            model.encoder.context_forward = 0
            print("(train) context forward: ", model.encoder.context_forward)

    model = accelerator.prepare(model)

    train_dataloader = make_loader(
        train_dataset,
        target=target,
        batch_size=config.training.train_batch_size,
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False
    )

    val_dataloader = make_loader(
        val_dataset,
        target=target,
        batch_size=config.training.test_batch_size,
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False
    )

    test_dataloader = make_loader(
        test_dataset,
        target=target,
        batch_size=10000,
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False
    )

    train_y, train_x = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            model.encoder.mask = False
            outputs = model(
                batch['spikes_data'],
                time_attn_mask=batch['time_attn_mask'],
                space_attn_mask=batch['space_attn_mask'],
                spikes_timestamps=batch['spikes_timestamps'], 
                spikes_spacestamps=batch['spikes_spacestamps'], 
                neuron_regions=batch['neuron_regions'],
                eval_mask=torch.ones_like(batch['spikes_data']),
                masking_mode = 'causal' if target == 'choice' else 'causal',
                num_neuron=batch['spikes_data'].shape[2],
                eid=batch['eid'][0],  
                nemo_rep=batch['nemo_rep']
            )
            train_y.append(batch['target'].clone())
            train_x.append(outputs.preds.clone())

        for batch in val_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)   
            model.encoder.mask = False 
            outputs = model(
                batch['spikes_data'],
                time_attn_mask=batch['time_attn_mask'],
                space_attn_mask=batch['space_attn_mask'],
                spikes_timestamps=batch['spikes_timestamps'], 
                spikes_spacestamps=batch['spikes_spacestamps'], 
                neuron_regions=batch['neuron_regions'],
                eval_mask=torch.ones_like(batch['spikes_data']),
                masking_mode = 'causal' if target == 'choice' else 'causal',
                num_neuron=batch['spikes_data'].shape[2],
                eid=batch['eid'][0],
                nemo_rep=batch['nemo_rep']
            )
            train_y.append(batch['target'].clone())
            train_x.append(outputs.preds.clone())
    train_y = torch.cat(train_y, dim=0)
    train_x = torch.cat(train_x, dim=0)

    test_y, test_x = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            
            model.encoder.mask = False
            
            outputs = model(
                batch['spikes_data'],
                time_attn_mask=batch['time_attn_mask'],
                space_attn_mask=batch['space_attn_mask'],
                spikes_timestamps=batch['spikes_timestamps'], 
                spikes_spacestamps=batch['spikes_spacestamps'], 
                neuron_regions=batch['neuron_regions'],
                eval_mask=torch.ones_like(batch['spikes_data']),
                masking_mode = 'causal' if target == 'choice' else 'causal',
                num_neuron=batch['spikes_data'].shape[2],
                eid=batch['eid'][0],
                nemo_rep=batch['nemo_rep']
            )
            test_y.append(batch['target'].clone())
            test_x.append(outputs.preds.clone())
    test_y = torch.cat(test_y, dim=0)
    test_x = torch.cat(test_x, dim=0)

    
    train_y = train_y.cpu().detach().numpy()
    train_x = train_x.cpu().detach().numpy().reshape((len(train_x), -1))
    test_y = test_y.cpu().detach().numpy()
    test_x = test_x.cpu().detach().numpy().reshape((len(test_x), -1))

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score

    if target == 'choice':
        train_y, test_y = train_y.argmax(1).flatten(), test_y.argmax(1).flatten()
        grid={"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4], "penalty":["l2"]}
        logreg=LogisticRegression(random_state=seed)
        logreg_cv=GridSearchCV(logreg, grid, cv=4)
        logreg_cv.fit(train_x, train_y)
        test_pred = logreg_cv.predict(test_x)
        acc = accuracy_score(test_y, test_pred)
        results = {
            metric: acc
        }
    elif target in ['whisker-motion-energy', 'wheel-speed', 'left-whisker-motion-energy']:
        grid={"alpha": [0, 1e1, 1e2, 1e3, 1e4]}
        reg=Ridge(random_state=seed)
        reg_cv=GridSearchCV(reg, grid, cv=4)
        reg_cv.fit(train_x, train_y)
        test_pred = reg_cv.predict(test_x)
        r2 = r2_score(test_y.flatten(), test_pred.flatten())
        results = {
            metric: r2
        }


    if not os.path.exists(kwargs['save_path']):
        os.makedirs(kwargs['save_path'])
    np.save(os.path.join(kwargs['save_path'], f'{target}_results.npy'), results)
    
    return {
        f"{target}_{metric}": results[metric],
    }


def region_wise_behavior_decoding(**kwargs):
    model_config = kwargs['model_config']
    trainer_config = kwargs['trainer_config']
    model_path = kwargs['model_path']
    dataset_path = kwargs['dataset_path']
    test_size = kwargs['test_size']
    seed = kwargs['seed']
    mask_name = kwargs['mask_name']
    metric = kwargs['metric']
    mask_ratio = kwargs['mask_ratio']
    mask_mode = mask_name.split("_")[1]
    eid = kwargs['eid']
    num_sessions = kwargs['num_sessions']
    target = kwargs['target']
    use_nemo = kwargs['use_nemo']

    # set seed
    set_seed(seed)

    # load the model
    config = config_from_kwargs({"model": f"include:{model_config}"})
    config = update_config(model_config, config)
    config = update_config(trainer_config, config)
    config.model.encoder.masker.mode = mask_mode

    accelerator = Accelerator()

    _,_,_, meta_data = load_ibl_dataset(
                        cache_dir=config.dirs.dataset_cache_dir,
                        user_or_org_name=config.dirs.huggingface_org,
                        num_sessions=1,
                        split_method="predefined",
                        test_session_eid=[],
                        batch_size=config.training.train_batch_size,
                        seed=seed,
                        eid=eid
                    )
    print(meta_data)

    # load the dataset
    dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    cluster_regions = np.array(test_dataset['cluster_regions'])[0].astype('str')

    if use_nemo:
        neuron_uuids = np.array(train_dataset['cluster_uuids'][0]).astype('str')
        with open('data/MtM_unit_embed.pkl','rb') as file:
            nemo_data = pickle.load(file)
        nemo_uuids = nemo_data['uuids']
        include_uuids = np.intersect1d(neuron_uuids, nemo_uuids)
        include_neuron_ids = np.argwhere(np.array([1 if uuid in include_uuids else 0 for uuid in neuron_uuids]).flatten() == 1).astype(np.int64)
        cluster_regions = cluster_regions[include_neuron_ids].squeeze()
        print('Use NEMO cell-type embeddings.')
        print('Num of neurons with NEMO embeddings: ', n_neurons)

    if config.model.model_class == "iTransformer" and config.model.encoder.embed_region:
        config["model"]["encoder"]["neuron_regions"] = list(
            set(str(b) for a in [row["cluster_regions"] for rows in dataset.values() for row in rows] for b in a)
        )

    n_neurons = len(cluster_regions)
    brain_regions = []
    for region in np.unique(cluster_regions):
        if region not in ['root']:
            brain_regions.append(region)

    if config.model.model_class in ["NDT1", "iTransformer"]:
        max_space_length = n_neurons  
    elif config.model.model_class == "STPatch":
        max_space_F = config.model.encoder.embedder.max_space_F
        max_space_length = ceil(n_neurons/max_space_F) * max_space_F
    else:
        max_space_length = config.data.max_space_length

    print('encoder max space length:', max_space_length)

    meta_data['max_space_length'] = max_space_length
    meta_data['num_neurons'] = [n_neurons]

    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model, **config.method.model_kwargs, **meta_data)

    if config.model.model_class == 'iTransformer':
        model.load_checkpoint(model_path)
        model.masker.mode = mask_mode
        model.masker.ratio = mask_ratio
        print("(behave decoding) masking mode: ", model.masker.mode)
        print("(behave decoding) masking ratio: ", model.masker.ratio)
        print("(behave decoding) masking active: ", model.masker.force_active)
    else:
        model = torch.load(model_path)['model']
        model.encoder.masker.mode = mask_mode
        model.encoder.masker.ratio = mask_ratio
        model.encoder.masker.force_active = False
        print("(behave decoding) masking mode: ", model.encoder.masker.mode)
        print("(behave decoding) masking ratio: ", model.encoder.masker.ratio)
        print("(behave decoding) masking active: ", model.encoder.masker.force_active)
        if 'causal' in mask_name:
            model.encoder.context_forward = 0
            print("(behave decoding) context forward: ", model.encoder.context_forward)

    model = accelerator.prepare(model)

    train_dataloader = make_loader(
        train_dataset,
        target=target,
        batch_size=config.training.train_batch_size,
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False
    )

    val_dataloader = make_loader(
        val_dataset,
        target=target,
        batch_size=config.training.test_batch_size,
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False
    )

    test_dataloader = make_loader(
        test_dataset,
        target=target,
        batch_size=10000,
        pad_to_right=True,
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        use_nemo=use_nemo,
        shuffle=False
    )

    results = {}
    for region in brain_regions:

        print(f'Start {region}:')
        if len(np.argwhere(cluster_regions == region).flatten()) < 1:
            print(f'Skip {region} due to too few neurons.')
            continue

        train_y, train_x = [], []
        model.eval()
        with torch.no_grad():
            for batch in train_dataloader:
                batch = move_batch_to_device(batch, accelerator.device)
                model.encoder.mask = False

                mask_result = heldout_mask(
                    batch['spikes_data'].clone(),
                    mode='intra_region',
                    heldout_idxs=np.array([]),
                    target_regions=[region],
                    neuron_regions=cluster_regions
                )   
                heldout_idxs = mask_result['heldout_idxs']

                outputs = model(
                    mask_result['spikes'],
                    time_attn_mask=batch['time_attn_mask'],
                    space_attn_mask=batch['space_attn_mask'],
                    spikes_timestamps=batch['spikes_timestamps'], 
                    spikes_spacestamps=batch['spikes_spacestamps'], 
                    neuron_regions=batch['neuron_regions'],
                    eval_mask=mask_result['eval_mask'],
                    masking_mode = 'intra-region',
                    num_neuron=batch['spikes_data'].shape[2],
                    eid=batch['eid'][0],
                    nemo_rep=batch['nemo_rep']
                )
                train_y.append(batch['target'].clone())
                train_x.append(torch.exp(outputs.preds.clone()[:,:,heldout_idxs]))

            for batch in val_dataloader:
                batch = move_batch_to_device(batch, accelerator.device)   
                model.encoder.mask = False 
                mask_result = heldout_mask(
                    batch['spikes_data'].clone(),
                    mode='intra_region',
                    heldout_idxs=np.array([]),
                    target_regions=[region],
                    neuron_regions=cluster_regions
                )   
                heldout_idxs = mask_result['heldout_idxs']

                outputs = model(
                    mask_result['spikes'],
                    time_attn_mask=batch['time_attn_mask'],
                    space_attn_mask=batch['space_attn_mask'],
                    spikes_timestamps=batch['spikes_timestamps'], 
                    spikes_spacestamps=batch['spikes_spacestamps'], 
                    neuron_regions=batch['neuron_regions'],
                    eval_mask=mask_result['eval_mask'],
                    masking_mode = 'intra-region',
                    num_neuron=batch['spikes_data'].shape[2],
                    eid=batch['eid'][0],
                    nemo_rep=batch['nemo_rep']
                )
                train_y.append(batch['target'].clone())
                train_x.append(torch.exp(outputs.preds.clone()[:,:,heldout_idxs]))
        train_y = torch.cat(train_y, dim=0)
        train_x = torch.cat(train_x, dim=0)

        test_y, test_x = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                batch = move_batch_to_device(batch, accelerator.device)
                
                model.encoder.mask = False
                
                mask_result = heldout_mask(
                    batch['spikes_data'].clone(),
                    mode='intra_region',
                    heldout_idxs=np.array([]),
                    target_regions=[region],
                    neuron_regions=cluster_regions
                )   
                heldout_idxs = mask_result['heldout_idxs']

                outputs = model(
                    mask_result['spikes'],
                    time_attn_mask=batch['time_attn_mask'],
                    space_attn_mask=batch['space_attn_mask'],
                    spikes_timestamps=batch['spikes_timestamps'], 
                    spikes_spacestamps=batch['spikes_spacestamps'], 
                    neuron_regions=batch['neuron_regions'],
                    eval_mask=mask_result['eval_mask'],
                    masking_mode = 'intra-region',
                    num_neuron=batch['spikes_data'].shape[2],
                    eid=batch['eid'][0],
                    nemo_rep=batch['nemo_rep']
                )
                test_y.append(batch['target'].clone())
                test_x.append(torch.exp(outputs.preds.clone()[:,:,heldout_idxs]))
        test_y = torch.cat(test_y, dim=0)
        test_x = torch.cat(test_x, dim=0)

        print(train_x.shape)
        
        train_y = train_y.cpu().detach().numpy()
        train_x = train_x.cpu().detach().numpy().reshape((len(train_x), -1))
        test_y = test_y.cpu().detach().numpy()
        test_x = test_x.cpu().detach().numpy().reshape((len(test_x), -1))

        if target == 'choice':
            train_y, test_y = train_y.argmax(1).flatten(), test_y.argmax(1).flatten()
            grid={"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4], "penalty":["l2"]}
            logreg=LogisticRegression(random_state=seed)
            logreg_cv=GridSearchCV(logreg, grid, cv=4)
            logreg_cv.fit(train_x, train_y)
            test_pred = logreg_cv.predict(test_x)
            acc = accuracy_score(test_y, test_pred)
            results[region] = acc
        elif target in ['whisker-motion-energy', 'wheel-speed', 'left-whisker-motion-energy']:
            grid={"alpha": [0, 1e1, 1e2, 1e3, 1e4]}
            reg=Ridge(random_state=seed)
            reg_cv=GridSearchCV(reg, grid, cv=4)
            reg_cv.fit(train_x, train_y)
            test_pred = reg_cv.predict(test_x)
            r2 = r2_score(test_y.flatten(), test_pred.flatten())
            results[region] = r2

    if not os.path.exists(kwargs['save_path']):
        os.makedirs(kwargs['save_path'])
    np.save(os.path.join(kwargs['save_path'], f'{target}_region_results.npy'), results)
        
    return results
    


def compare_R2_scatter(**kwargs):
    A_path = kwargs['A_path'],
    B_path = kwargs['B_path'],
    A_name = kwargs['A_name'],
    B_name = kwargs['B_name'],

    A_r2 = np.load(os.path.join(A_path, 'r2.npy'))
    B_r2 = np.load(os.path.join(B_path, 'r2.npy'))

    A_psth = A_r2[:, 0]
    B_psth = B_r2[:, 0]
    A_psth[A_psth < 0] = 0
    B_psth[B_psth < 0] = 0

    A_r2 = A_r2[:, 1]
    B_r2 = B_r2[:, 1]
    A_r2[A_r2 < 0] = 0
    B_r2[B_r2 < 0] = 0

    line_x = np.linspace(0, 1, 100)
    line_y = line_x

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(A_psth, B_psth, alpha=0.9, s=1)
    axes[0].plot(line_x, line_y, color='black', lw=1)
    axes[0].set_xlabel(A_name)
    axes[0].set_ylabel(B_name)
    axes[0].set_title('R2_PSTH')

    axes[1].scatter(A_r2, B_r2, alpha=0.9, s=1)
    axes[1].plot(line_x, line_y, color='black', lw=1)
    axes[1].set_xlabel(A_name)
    axes[1].set_ylabel(B_name)
    axes[1].set_title('R2')

    
# --------------------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------------------

def heldout_mask(
        spike_data,                     # (K, T, N)
        mode='manual',                  # manual / active / per_neuron / forward_pred / inter_region / etc (TODO)
        heldout_idxs=np.array([]),      # list for manual mode
        n_active=1,                     # n_neurons for most-active mode
        target_regions=None,            # list for region mode
        neuron_regions=None,            # list for region mode
):
    mask = torch.ones(spike_data.shape).to(torch.int64).to(spike_data.device)
    
    if mode == 'manual':
        hd = heldout_idxs
        mask[:, :, hd] = 0

    elif mode == 'most':
        _act = spike_data.detach().cpu().numpy()
        _act = np.mean(_act, axis=(0, 1))
        act_idx = np.argsort(_act)
        hd = np.array(act_idx[-n_active:])
        mask[:, :, hd] = 0

    elif mode == 'inter_region':
        hd = []
        for region in target_regions:
            region_idxs = np.argwhere(neuron_regions == region).flatten()
            mask[:, :, region_idxs] = 0 
            target_idxs = region_idxs[heldout_idxs]
            hd.append(target_idxs)
        hd = np.stack(hd).flatten()

    elif mode == 'intra_region':
        mask *= 0
        hd = []
        for region in target_regions:
            region_idxs = np.argwhere(neuron_regions == region).flatten()
            mask[:, :, region_idxs] = 1 
            if len(heldout_idxs) == 0:
                target_idxs = region_idxs
            else:
                target_idxs = region_idxs[heldout_idxs]
                mask[:, :, target_idxs] = 0
            hd.append(target_idxs)
        hd = np.stack(hd).flatten()
            
    elif mode == 'forward_pred':
        hd = heldout_idxs
        mask[:, hd, :] = 0
        
    else:
        raise NotImplementedError('mode not implemented')

    spike_data_masked = spike_data * mask

    return {"spikes": spike_data_masked, "heldout_idxs": hd, "eval_mask": 1-mask}


# --------------------------------------------------------------------------------------------------
# copied from NLB repo
# standard evaluation metrics
# --------------------------------------------------------------------------------------------------

def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    # print('nll_score', np.sum(result))
    # print('rate', rates.reshape(-1, rates.shape[1]*rates.shape[2]), '\nspikes', spikes.reshape(-1, spikes.shape[1]*spikes.shape[2]), '\nresult', result.reshape(-1, result.shape[1]*result.shape[2]))
    # print(rates.shape, spikes.shape, result.shape)
    return np.sum(result)


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    # print(np.nansum(spikes))
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


# --------------------------------------------------------------------------------------------------
# single neuron plot functions
# --------------------------------------------------------------------------------------------------

"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: for each task variable in var_tasklists, compute PSTH
:var_name2idx: for each task variable in var_tasklists, the corresponding index of X
:var_value2label:
:aligned_tbins: reference time steps to annotate. 
"""


def plot_psth(X, y, y_pred, var_tasklist, var_name2idx, var_value2label,
              aligned_tbins=[],
              axes=None, legend=False, neuron_idx='', neuron_region='', save_plot=False):
    
    if save_plot:
        if axes is None:
            nrows = 1;
            ncols = len(var_tasklist)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

        for ci, var in enumerate(var_tasklist):
            ax = axes[ci]
            psth_xy = compute_all_psth(X, y, var_name2idx[var])
            psth_pred_xy = compute_all_psth(X, y_pred, var_name2idx[var])
            
            for _i, _x in enumerate(psth_xy.keys()):
                psth = psth_xy[_x]
                psth_pred = psth_pred_xy[_x]
                ax.plot(psth,
                        color=plt.get_cmap('tab10')(_i),
                        linewidth=3, alpha=0.3, label=f"{var_value2label[var][tuple(_x)]}")
                ax.plot(psth_pred,
                        color=plt.get_cmap('tab10')(_i),
                        linestyle='--')
                ax.set_xlabel("Time bin")
                if ci == 0:
                    ax.set_ylabel("Neural activity")
                else:
                    ax.sharey(axes[0])
            _add_baseline(ax, aligned_tbins=aligned_tbins)
            if legend:
                ax.legend()
                ax.set_title(f"{var}")

    # compute PSTH for task_contingency
    idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
    psth_xy = compute_all_psth(X, y, idxs_psth)
    psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
    r2_psth = compute_R2_psth(psth_xy, psth_pred_xy, clip=False)
    r2_single_trial = compute_R2_main(y.reshape(-1, 1), y_pred.reshape(-1, 1), clip=False)[0]
    
    if save_plot:
        axes[0].set_ylabel(
            f'Neuron: #{neuron_idx[:4]} \n PSTH R2: {r2_psth:.2f} \n Avg_SingleTrial R2: {r2_single_trial:.2f}')

        for ax in axes:
            # ax.axis('off')
            ax.spines[['right', 'top']].set_visible(False)
            # ax.set_frame_on(False)
            # ax.tick_params(bottom=False, left=False)
        plt.tight_layout()

    return r2_psth, r2_single_trial


"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: variables used for computing the task-condition-averaged psth if subtract_psth=='task'
:var_name2idx:
:var_tasklist: variables to be plotted in the single-trial behavior
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:aligned_tbins: reference time steps to annotate. 
:nclus, n_neighbors: hyperparameters for spectral_clustering
:cmap, vmax_perc, vmin_perc: parameters used when plotting the activity and behavior
"""


def plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_behlist,
                               var_tasklist, subtract_psth="task",
                               aligned_tbins=[],
                               n_clus=8, n_neighbors=5, n_pc=32, clusby='y_pred',
                               cmap='bwr', vmax_perc=90, vmin_perc=10,
                               axes=None):
    if axes is None:
        ncols = 1;
        nrows = 2 + len(var_behlist) + 1 + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows))

    ### get the psth-subtracted y
    if subtract_psth is None:
        pass
    elif subtract_psth == "task":
        idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
        psth_xy = compute_all_psth(X, y, idxs_psth)
        psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
        y_psth = np.asarray(
            [psth_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y_predpsth = np.asarray(
            [psth_pred_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    elif subtract_psth == "global":
        y_psth = np.mean(y, 0)
        y_predpsth = np.mean(y_pred, 0)
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. \'\'"
    y_residual = (y_pred - y)  # (K, T), residuals of prediction
    idxs_behavior = np.concatenate(([var_name2idx[var] for var in var_behlist])) if len(var_behlist) > 0 else []
    X_behs = X[:, :, idxs_behavior]

    ### plot single-trial activity
    # arange the trials by unsupervised clustering labels
    # model = Rastermap(n_clusters=n_clus, # None turns off clustering and sorts single neurons
    #               n_PCs=n_pc, # use fewer PCs than neurons
    #               locality=0.15, # some locality in sorting (this is a value from 0-1)
    #               time_lag_window=15, # use future timepoints to compute correlation
    #               grid_upsample=0, # 0 turns off upsampling since we're using single neurons
    #             )
    # if clusby == 'y_pred':
    #     clustering = model.fit(y_pred)
    # elif clusby == 'y':
    #     clustering = model.fit(y)
    # else:
    #     assert False, "invalid clusby"
    # t_sort = model.isort

    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0)
    if clusby == 'y_pred':
        clustering = clustering.fit(y_pred)
    elif clusby == 'y':
        clustering = clustering.fit(y)
    else:
        assert False, "invalid clusby"
    t_sort = np.argsort(clustering.labels_)

    for ri, (toshow, label, ax) in enumerate(zip([y, y_pred, X_behs, y_residual],
                                                 [f"obs. act. \n (subtract_psth={subtract_psth})",
                                                  f"pred. act. \n (subtract_psth={subtract_psth})",
                                                  var_behlist,
                                                  "residual act."],
                                                 [axes[0], axes[1], axes[2:-2], axes[-2]])):
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(y_pred, vmax_perc)
            vmin = np.percentile(y_pred, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)
        elif ri == 2:
            # plot behavior
            for bi in range(len(var_behlist)):
                ts_ = toshow[:, :, bi][t_sort]
                vmax = np.percentile(ts_, vmax_perc)
                vmin = np.percentile(ts_, vmin_perc)
                raster_plot(ts_, vmax, vmin, True, label[bi], ax[bi],
                            cmap=cmap,
                            aligned_tbins=aligned_tbins)
        elif ri == 3:
            # plot residual activity
            vmax = np.percentile(toshow, vmax_perc)
            vmin = np.percentile(toshow, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)

    ### plot single-trial activity
    # re-arrange the trials
    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    # model = Rastermap(n_clusters=n_clus, n_PCs=n_pc, locality=0.15, time_lag_window=15, grid_upsample=0,).fit(y_residual)
    # t_sort_rd = model.isort
    raster_plot(y_residual[t_sort_rd], np.percentile(y_residual, vmax_perc), np.percentile(y_residual, vmin_perc), True,
                "residual act. (re-clustered)", axes[-1])

    plt.tight_layout()


"""
This script generates a plot to examine the (single-trial) fitting of a single neuron.
:X: behavior matrix of the shape [n_trials, n_timesteps, n_variables]. 
:y: true neural activity matrix of the shape [n_trials, n_timesteps] 
:ypred: predicted activity matrix of the shape [n_trials, n_timesteps] 
:var_name2idx: dictionary mapping feature names to their corresponding index of the 3-rd axis of the behavior matrix X. e.g.: {"choice": [0], "wheel": [1]}
:var_tasklist: *static* task variables used to form the task condition and compute the psth. e.g.: ["choice"]
:var_value2label: dictionary mapping values in X to their corresponding readable labels (only required for static task variables). e.g.: {"choice": {1.: "left", -1.: "right"}}
:var_behlist: *dynamic* behavior variables. e.g., ["wheel"]
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:algined_tbins: reference time steps to annotate in the plot. 
"""


def viz_single_cell(X, y, y_pred, var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[], clusby='y_pred', neuron_idx='', neuron_region='', method='',
                    save_path='figs', save_plot=False):
    
    if save_plot:
        nrows = 8
        plt.figure(figsize=(8, 2 * nrows))
        axes_psth = [plt.subplot(nrows, len(var_tasklist), k + 1) for k in range(len(var_tasklist))]
        axes_single = [plt.subplot(nrows, 1, k) for k in range(2, 2 + 2 + len(var_behlist) + 2)]
    else:
        axes_psth = None
        axes_single = None


    ### plot psth
    r2_psth, r2_trial = plot_psth(X, y, y_pred,
                                  var_tasklist=var_tasklist,
                                  var_name2idx=var_name2idx,
                                  var_value2label=var_value2label,
                                  aligned_tbins=aligned_tbins,
                                  axes=axes_psth, legend=True, neuron_idx=neuron_idx, neuron_region=neuron_region,
                                  save_plot=save_plot)

    ### plot the psth-subtracted activity
    if save_plot:
        plot_single_trial_activity(X, y, y_pred,
                                   var_name2idx,
                                   var_behlist,
                                   var_tasklist, subtract_psth=subtract_psth,
                                   aligned_tbins=aligned_tbins,
                                   clusby=clusby,
                                   axes=axes_single)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if save_plot:
        plt.savefig(os.path.join(save_path, f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2_trial:.2f}_{method}.png"))
        plt.tight_layout();

    return r2_psth, r2_trial
    

def viz_single_cell_unaligned(
    gt, pred, neuron_idx, neuron_region, method, save_path, 
    n_clus=8, n_neighbors=5, save_plot=False
):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    r2 = 0
    for _ in range(len(gt)):
        r2 += r2_score(gt, pred)
    r2 /= len(gt)

    if save_plot:
        y = gt - gt.mean(0)
        y_pred = pred - pred.mean(0)
        y_resid = y - y_pred

        clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                            affinity='nearest_neighbors',
                                            assign_labels='discretize',
                                            random_state=0)

        clustering = clustering.fit(y_pred)
        t_sort = np.argsort(clustering.labels_)
        
        vmin_perc, vmax_perc = 10, 90 
        vmax = np.percentile(y_pred, vmax_perc)
        vmin = np.percentile(y_pred, vmin_perc)
        
        toshow = [y, y_pred, y_resid]
        resid_vmax = np.percentile(toshow, vmax_perc)
        resid_vmin = np.percentile(toshow, vmin_perc)
        
        N = len(y)
        y_labels = ['obs.', 'pred.', 'resid.']

        fig, axes = plt.subplots(3, 1, figsize=(8, 7))
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im1 = axes[0].imshow(y[t_sort], aspect='auto', cmap='bwr', norm=norm)
        cbar = plt.colorbar(im1, pad=0.02, shrink=.6)
        cbar.ax.tick_params(rotation=90)
        axes[0].set_title(f' R2: {r2:.3f}')
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im2 = axes[1].imshow(y_pred[t_sort], aspect='auto', cmap='bwr', norm=norm)
        cbar = plt.colorbar(im2, pad=0.02, shrink=.6)
        cbar.ax.tick_params(rotation=90)
        norm = colors.TwoSlopeNorm(vmin=resid_vmin, vcenter=0, vmax=resid_vmax)
        im3 = axes[2].imshow(y_resid[t_sort], aspect='auto', cmap='bwr', norm=norm)
        cbar = plt.colorbar(im3, pad=0.02, shrink=.6)
        cbar.ax.tick_params(rotation=90)
        
        for i, ax in enumerate(axes):
            ax.set_ylabel(f"{y_labels[i]}"+f"\n(#trials={N})")
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.spines[['left','bottom', 'right', 'top']].set_visible(False)
        
        plt.savefig(os.path.join(save_path, f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2:.2f}_{method}.png"))
        plt.tight_layout()

    return r2


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin - 1, c='k', alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
                    xy=(tbin - 1, N),
                    xytext=(tbin - 1, N + 10),
                    ha='center',
                    va='center',
                    arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}" + f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        pass
    else:
        ax.axis('off')


"""
- X, y should be nparray with
    - X: [K,T,ncoef]
    - y: [K,T,N] or [K,T]
- axis and value should be list
- return: nparray [T, N] or [T]
"""


def compute_PSTH(X, y, axis, value):
    trials = np.all(X[:, 0, axis] == value, axis=-1)
    return y[trials].mean(0)


def compute_all_psth(X, y, idxs_psth):
    uni_vs = np.unique(X[:, 0, idxs_psth], axis=0)  # get all the unique task-conditions
    psth_vs = {};
    for v in uni_vs:
        # compute separately for true y and predicted y
        _psth = compute_PSTH(X, y,
                             axis=idxs_psth, value=v)  # (T)
        psth_vs[tuple(v)] = _psth
    return psth_vs


"""
psth_xy/ psth_pred_xy: {tuple(x): (T) or (T,N)}
return a float or (N) array
"""


def compute_R2_psth(psth_xy, psth_pred_xy, clip=True):
    psth_xy_array = np.array([psth_xy[x] for x in psth_xy])
    psth_pred_xy_array = np.array([psth_pred_xy[x] for x in psth_xy])
    K, T = psth_xy_array.shape[:2]
    psth_xy_array = psth_xy_array.reshape((K * T, -1))
    psth_pred_xy_array = psth_pred_xy_array.reshape((K * T, -1))
    r2s = [r2_score(psth_xy_array[:, ni], psth_pred_xy_array[:, ni]) for ni in range(psth_xy_array.shape[1])]
    r2s = np.array(r2s)
    # # compute r2 along dim 0
    # r2s = [r2_score(psth_xy[x], psth_pred_xy[x], multioutput='raw_values') for x in psth_xy]
    if clip:
        r2s = np.clip(r2s, 0., 1.)
    # r2s = np.mean(r2s, 0)
    if len(r2s) == 1:
        r2s = r2s[0]
    return r2s


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
        

