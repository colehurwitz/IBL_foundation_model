
import numpy as np
import pandas as pd
import torch

from nlb_tools.make_tensors import (
        make_train_input_tensors,
        make_eval_input_tensors,
        make_eval_target_tensors,
        PARAMS,
        _prep_mask,
        _prep_behavior,
        make_stacked_array,
        make_jagged_array
    )
from nlb_tools.nwb_interface import NWBDataset
# ## Load dataset
# dataset = NWBDataset("data/000129/sub-Indy", "*train", split_heldout=False)
# # resample to 20 ms
# dataset.resample(6)
params = PARAMS["mc_rtt"].copy()
spike_params = {'align_field': 'start_time', 'align_range': (0, 600), 'allow_overlap': True}
make_params = params['make_params'].copy()

behavior_source = params['behavior_source']
behavior_fields = ["cursor_pos", "finger_pos","finger_vel","target_pos"]

def create_rtt_dataset(dataset, trial_mask, behavior_make_params):
    # Retrieve neural data from indicated source
    data_dict = make_stacked_array(dataset, ["spikes", "heldout_spikes"], spike_params, trial_mask)
    # Retrieve behavior data from indicated source
    data_behavior_dict = {}
    if behavior_source == 'data':
        for behavior_field in behavior_fields:
            data_behavior_dict[behavior_field] = make_jagged_array(dataset, [behavior_field], behavior_make_params, trial_mask)[0][behavior_field]

    data_dict = {**data_dict, **data_behavior_dict}
    data_dict['spikes'] = np.concatenate([data_dict['spikes'], data_dict['heldout_spikes']], axis=2)
    trial_id = np.arange(data_dict['spikes'].shape[0])
    data_dict['trial_id'] = trial_id
    return data_dict

def load_nlb_dataset(dataset_path, bin_size_ms=6):
    dataset = NWBDataset(dataset_path, "*train", split_heldout=True)
    # resample to bin_size_ms
    dataset.resample(bin_size_ms)
    # Prep mask
    train_trial_mask = _prep_mask(dataset, "train")
    val_trial_mask = _prep_mask(dataset, "val")
    # select 10% of val trials for test
    # test_trial_mask = np.zeros_like(val_trial_mask)
    # test_trial_mask[np.random.choice(np.where(val_trial_mask)[0], int(val_trial_mask.sum() * 0.1), replace=False)] = True
    # val_trial_mask[test_trial_mask] = False
    
    # Prep behavior
    behavior_make_params = _prep_behavior(dataset, params.get('lag', None), make_params)

    # Create datasets
    train_dataset = create_rtt_dataset(dataset, train_trial_mask, behavior_make_params)
    val_dataset = create_rtt_dataset(dataset, val_trial_mask, behavior_make_params)
    # test_dataset = create_rtt_dataset(dataset, test_trial_mask, behavior_make_params)

    print(f"trials number: train {train_trial_mask.sum()}, val {val_trial_mask.sum()}")
    meta_data = {
        "num_neurons":[train_dataset['spikes'].shape[2]],
        "num_sessions":1,
        "eids":{"nlb-rtt"},
    }
    return train_dataset, val_dataset, meta_data

def load_nlb_dataset_test(dataset_path, bin_size_ms=6):
    dataset = NWBDataset(dataset_path, split_heldout=True)

    train_split = ['train', 'val']
    # resample to bin_size_ms
    dataset.resample(bin_size_ms)
    train_dict = make_train_input_tensors(dataset, dataset_name='mc_rtt', trial_split=train_split, save_file=False)
    # Unpack data
    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']

    # Print 3d array shape: trials x time x channel
    train_spikes = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=2)
    trial_id = np.arange(train_spikes_heldin.shape[0])
    print(train_spikes.shape)
    train_data_dict = {
        "spikes": train_spikes,
        "trial_id": trial_id
    }

    ## Make eval data
    # Split for evaluation is same as phase name
    eval_split = 'test'
    # Make data tensors
    eval_dict = make_eval_input_tensors(dataset, dataset_name='mc_rtt', trial_split=eval_split, save_file=False)

    eval_spikes_heldin = eval_dict['eval_spikes_heldin']
    # fill in eval_spikes_heldout
    # eval_spikes_heldout.shape[0], train_spikes_heldout.shape[1], train_spikes_heldout.shape[2]
    eval_spikes_heldout = np.zeros((eval_spikes_heldin.shape[0], train_spikes_heldout.shape[1], train_spikes_heldout.shape[2]))
    eval_spikes = np.concatenate([eval_spikes_heldin, eval_spikes_heldout], axis=2)
    trial_id = np.arange(eval_spikes_heldin.shape[0])
    eval_data_dict = {
        "spikes": eval_spikes,
        "trial_id": trial_id
    }

    meta_data = {
        "num_neurons":[train_spikes.shape[2]],
        "num_sessions":1,
        "eids":{"nlb-rtt"},
    }
    return train_data_dict, eval_data_dict, meta_data

