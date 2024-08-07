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
import numpy as np 
from utils.nlb_data_utils import load_nlb_dataset_test

train_dataset, eval_dataset, meta_data = load_nlb_dataset_test('data/000129/sub-Indy', 20)

dataset = NWBDataset('data/000129/sub-Indy', "*train", split_heldout=True)
# # resample to bin_size_ms
# dataset.resample(20)
# print(dataset.data['heldout_spikes'].shape)
# print(dataset.data['spikes'].shape)
# # exit()
# # Prep mask
# train_trial_mask = _prep_mask(dataset, "train")
# val_trial_mask = _prep_mask(dataset, "val")
# test_trial_mask = _prep_mask(dataset, "eval")
# print(f"trials number: train {train_trial_mask.sum()}, val {val_trial_mask.sum()}, test {test_trial_mask.sum()}")
# exit()
# # select 10% of val trials for test
# test_trial_mask = np.zeros_like(val_trial_mask)
# test_trial_mask[np.random.choice(np.where(val_trial_mask)[0], int(val_trial_mask.sum() * 0.1), replace=False)] = True
# val_trial_mask[test_trial_mask] = False

## Dataset preparation

# Choose the phase here, either 'val' for the Validation phase or 'test' for the Test phase
# Note terminology overlap with 'train', 'val', and 'test' data splits -
# the phase name corresponds to the data split that predictions are evaluated on
phase = 'val'

# Choose bin width and resample
bin_width = 20
dataset.resample(bin_width)

# Create suffix for group naming later
suffix = '' if (bin_width == 5) else f'_{int(bin_width)}'
print(suffix)

## Make train data

# Create input tensors, returned in dict form
train_split = 'train' if (phase == 'val') else ['train', 'val']
train_dict = make_train_input_tensors(dataset, dataset_name='mc_rtt', trial_split=train_split, save_file=False)

# Show fields of returned dict
print(train_dict.keys())

# Unpack data
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']

# Print 3d array shape: trials x time x channel
print(train_spikes_heldin.shape)
print(train_spikes_heldout.shape)

## Make eval data

# Split for evaluation is same as phase name
eval_split = phase
# Make data tensors
eval_dict = make_eval_input_tensors(dataset, dataset_name='mc_rtt', trial_split=eval_split, save_file=False)
print(eval_dict.keys()) # only includes 'eval_spikes_heldout' if available
eval_spikes_heldin = eval_dict['eval_spikes_heldin']

print(eval_spikes_heldin.shape)