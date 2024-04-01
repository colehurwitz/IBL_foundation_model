'''Upload spike data for self-supervised learning.'''


import os
import sys
from pathlib import Path
path_root = '../'
sys.path.append(str(path_root))

import numpy as np
import pandas as pd

from one.api import ONE

from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    create_intervals, 
    bin_spiking_data
)
from utils.dataset import (
    create_dataset, 
    upload_dataset, 
    download_dataset, 
    get_binned_spikes_from_sparse
)

np.random.seed(0)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True
)

freeze_file = '../data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

# select 100 eids to upload to Hugging Face
n_sub = 100
subjects = np.unique(bwm_df.subject)
selected_subs = np.random.choice(subjects, n_sub, replace=False)
by_subject = bwm_df.groupby('subject')
include_eids = np.array([bwm_df.eid[by_subject.groups[sub][0]] for sub in selected_subs])

# setup
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False}
for eid_idx, eid in enumerate(include_eids):

    try: 
        print('======================')
        print(f'Process session {eid} from subject {selected_subs[eid_idx]}:')
        
        neural_dict, _, meta_data, _ = prepare_data(one, eid, bwm_df, params, n_workers=3)
        regions, beryl_reg = list_brain_regions(neural_dict, **params)
        region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
        intervals = create_intervals(
            start_time=0, end_time=neural_dict['spike_times'].max(), interval_len=params['interval_len']
        )
        binned_spikes, clusters_used_in_bins = bin_spiking_data(
            region_cluster_ids, neural_dict, intervals=intervals, n_workers=3, **params
        )
        dataset = create_dataset(
            binned_spikes, bwm_df, eid, params, meta_data=meta_data, binned_behaviors=None
        )
        upload_dataset(dataset, org='neurofm123', eid=eid)
        dataset = download_dataset(org='neurofm123', eid=eid, cache_dir='/mnt/3TB/yizi/huggingface/datasets')
    
        print('======================')
        print(f'Uploaded session {eid} to Hugging Face.')
        print(f'Finished {eid_idx+1} / {len(include_eids)} sessions.')
        
        spikes_sparse_data_list = dataset['spikes_sparse_data']
        spikes_sparse_indices_list = dataset['spikes_sparse_indices']
        spikes_sparse_indptr_list = dataset['spikes_sparse_indptr']
        spikes_sparse_shape_list = dataset['spikes_sparse_shape']
        
        downloaded_binned_spikes = get_binned_spikes_from_sparse(
            spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list
        )
    
        assert downloaded_binned_spikes.shape == binned_spikes.shape, f'Downloaded spike data shape cannot match to original data shape.'
    except Exception as e:
        print(f'Skipped session {eid} due to unexpected error: ', e)
    




