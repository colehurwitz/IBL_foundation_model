'''Cache spike and behavior data for decoding with pre-defined train, val, test splits.'''

import os
import sys
from pathlib import Path
path_root = '..'
sys.path.append(str(path_root))

import numpy as np
import pandas as pd

from one.api import ONE

from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    create_intervals, 
    bin_spiking_data,
    bin_behaviors,
    align_spike_behavior
)
from datasets import DatasetDict
from utils.dataset_utils import create_dataset, upload_dataset

np.random.seed(42)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True,
    cache_dir = '/expanse/lustre/scratch/yzhang39/temp_project/'
)

freeze_file = Path(path_root)/'data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

# select 100 eids to upload to Hugging Face
n_sub = 100
subjects = np.unique(bwm_df.subject)
selected_subs = np.random.choice(subjects, n_sub, replace=False)
by_subject = bwm_df.groupby('subject')
include_eids = np.array([bwm_df.eid[by_subject.groups[sub][0]] for sub in selected_subs])
# include_eids = np.insert(include_eids, -1, '671c7ea7-6726-4fbe-adeb-f89c2c8e489b')

# setup
params = {
    'interval_len': 2, 'binsize': 0.02, 'single_region': False,
    'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)
}

error_eids = []
for eid_idx, eid in enumerate(include_eids):

    try: 
        print('======================')
        print(f'Process session {eid} from subject {selected_subs[eid_idx]}:')
        
        neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, bwm_df, params, n_workers=2)
        regions, beryl_reg = list_brain_regions(neural_dict, **params)
        region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
        binned_spikes, clusters_used_in_bins = bin_spiking_data(
            region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], n_workers=10, **params
        )
        binned_behaviors, behavior_masks = bin_behaviors(
            one, eid, trials_df=trials_data['trials_df'], allow_nans=True, **params
        )
        
        # IMPORTANT: ensure spiking data and behavior match in each trial
        aligned_binned_spikes, aligned_binned_behaviors = align_spike_behavior(
            binned_spikes, binned_behaviors, trials_data['trials_mask']
        )
    
        # train: 0.7, val: 0.1, test: 0.2
        max_num_trials = len(aligned_binned_spikes)
        trial_idxs = np.random.choice(np.arange(max_num_trials), max_num_trials, replace=False)
        train_idxs = trial_idxs[:int(0.7*max_num_trials)]
        val_idxs = trial_idxs[int(0.7*max_num_trials):int(0.8*max_num_trials)]
        test_idxs = trial_idxs[int(0.8*max_num_trials):]
    
        train_beh, val_beh, test_beh = {}, {}, {}
        for beh in aligned_binned_behaviors.keys():
            train_beh.update({beh: aligned_binned_behaviors[beh][train_idxs]})
            val_beh.update({beh: aligned_binned_behaviors[beh][val_idxs]})
            test_beh.update({beh: aligned_binned_behaviors[beh][test_idxs]})
        
        train_dataset = create_dataset(
            aligned_binned_spikes[train_idxs], bwm_df, eid, params, 
            binned_behaviors=train_beh, meta_data=meta_data
        )
        val_dataset = create_dataset(
            aligned_binned_spikes[val_idxs], bwm_df, eid, params, 
            binned_behaviors=val_beh, meta_data=meta_data
        )
        test_dataset = create_dataset(
            aligned_binned_spikes[test_idxs], bwm_df, eid, params, 
            binned_behaviors=test_beh, meta_data=meta_data
        )
        partitioned_dataset = DatasetDict({
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset}
        )
        print(partitioned_dataset)
        # partitioned_dataset.save_to_disk(f'/burg/stats/users/yz4123/cached_ibl_data/ibl-fm/aligned/{eid}')
        upload_dataset(partitioned_dataset, org='neurofm123', eid=f'{eid}_aligned')
    
        print('======================')
        print(f'Cached session {eid}.')
        print(f'Finished {eid_idx+1} / {len(include_eids)} sessions.')
            
    except Exception as e:
        error_eids.append(eid)
        print(f'Skipped session {eid} due to unexpected error: ', e)

print(error_eids)

