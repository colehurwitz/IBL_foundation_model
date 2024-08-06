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
dataset = NWBDataset('data/000129/sub-Indy', "*train", split_heldout=True)
# resample to bin_size_ms
dataset.resample(20)
print(dataset.data['heldout_spikes'].shape)
print(dataset.data['spikes'].shape)
exit()
# Prep mask
train_trial_mask = _prep_mask(dataset, "train")
val_trial_mask = _prep_mask(dataset, "val")
# select 10% of val trials for test
test_trial_mask = np.zeros_like(val_trial_mask)
test_trial_mask[np.random.choice(np.where(val_trial_mask)[0], int(val_trial_mask.sum() * 0.1), replace=False)] = True
val_trial_mask[test_trial_mask] = False