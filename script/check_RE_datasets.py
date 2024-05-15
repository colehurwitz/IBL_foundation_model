'''Cache spike and behavior data for decoding with pre-defined train, val, test splits.'''

import os
import sys
from pathlib import Path
path_root = '/home/yizi/IBL_foundation_model/'
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
)

freeze_file = Path(path_root)/'data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

# select 40 eids to upload to Hugging Face
with open(Path(path_root)/'data/ibl_repro_ephys_eids.txt') as file:
    include_eids = [line.rstrip() for line in file]

with open(Path(path_root)/'data/target_eids.txt') as file:
    processed_eids = [line.rstrip() for line in file]

# setup
params = {
    'interval_len': 2, 'binsize': 0.02, 'single_region': False,
    'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)
}

good_eids, all_regions = [], []
for eid_idx, eid in enumerate(include_eids):

    if eid in processed_eids:
        continue

    print(eid_idx)
    
    neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, bwm_df, params, n_workers=1)
    regions, beryl_reg = list_brain_regions(neural_dict, **params)

    if (len(meta_data['uuids']) > 600) and (len(meta_data['uuids']) < 1000):
        good_eids.append(eid)
        all_regions.append(np.array(regions).flatten())
        print(eid, len(meta_data['uuids']))
        print(good_eids)
        print(np.unique(np.concatenate(all_regions)))
        print(len(np.unique(np.concatenate(all_regions))))

for eid in good_eids:
    print(eid)
