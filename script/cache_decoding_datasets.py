'''Cache spike and behavior data for decoding with pre-defined train, val, test splits.'''

import os
import sys
from pathlib import Path
path_root = '/home/yizi/IBL_foundation_model'
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
from utils.dataset import create_dataset

np.random.seed(0)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True
)

freeze_file = Path(path_root)/'data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

# select 100 eids to upload to Hugging Face
n_sub = 100
subjects = np.unique(bwm_df.subject)
selected_subs = np.random.choice(subjects, n_sub, replace=False)
by_subject = bwm_df.groupby('subject')
include_eids = np.array([bwm_df.eid[by_subject.groups[sub][0]] for sub in selected_subs])

# setup
params = {
    'interval_len': 2, 'binsize': 0.02, 'single_region': False,
    'align_time': 'firstMovement_times', 'time_window': (0., 2.)
}

for eid_idx, eid in enumerate(include_eids):

    try: 
        print('======================')
        print(f'Process session {eid} from subject {selected_subs[eid_idx]}:')
        
        neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, bwm_df, params, n_workers=4)
        regions, beryl_reg = list_brain_regions(neural_dict, **params)
        region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
        binned_spikes, clusters_used_in_bins = bin_spiking_data(
            region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], n_workers=4, **params
        )
        binned_behaviors, behavior_masks = bin_behaviors(
            one, eid, trials_df=trials_data['trials_df'], allow_nans=True, **params
        )
        # IMPORTANT: ensure spiking data and behavior match in each trial
        aligned_binned_spikes, aligned_binned_behaviors = align_spike_behavior(binned_spikes, binned_behaviors)
        dataset = create_dataset(
            aligned_binned_spikes, bwm_df, eid, params, 
            binned_behaviors=aligned_binned_behaviors, meta_data=meta_data
        )
        # train: 0.7, val: 0.1, test: 0.2
        train_test = dataset.train_test_split(test_size=0.2)
        train_val = train_test['train'].train_test_split(test_size=0.1)
        partitioned_dataset = DatasetDict({
            'train': train_val['train'],
            'val': train_val['test'],
            'test': train_test['test']}
        )
        print(partitioned_dataset)
        partitioned_dataset.save_to_disk(f'/mnt/3TB/yizi/huggingface/decoding_datasets/{eid}')
    
        print('======================')
        print(f'Cached session {eid}.')
        print(f'Finished {eid_idx+1} / {len(include_eids)} sessions.')
        
    except Exception as e:
        print(f'Skipped session {eid} due to unexpected error: ', e)
    
    break
    