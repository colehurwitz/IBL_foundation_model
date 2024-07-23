import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from datasets import DatasetDict
from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_spike_behavior
)
from utils.dataset_utils import create_dataset, upload_dataset
from lfp.preprocess_lfp import prepare_lfp, featurize_lfp

ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--datasets", type=str, default="reproducible-ephys", choices=["reproducible-ephys", "brain-wide-map"])
ap.add_argument("--huggingface_org", type=str, default="neurofm123")
ap.add_argument("--n_sessions", type=int, default=1)
ap.add_argument("--query_session", nargs="+", default=[]) # --query_session EID_1 EID_2 EID_3 
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

SEED = 42

np.random.seed(SEED)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True,
    cache_dir = args.base_path
)

freeze_file = 'data/bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

if len(args.query_session) == 0:
    if args.datasets == "brain-wide-map":
        n_sub = args.n_sessions
        subjects = np.unique(bwm_df.subject)
        selected_subs = np.random.choice(subjects, n_sub, replace=False)
        by_subject = bwm_df.groupby('subject')
        include_eids = np.array([bwm_df.eid[by_subject.groups[sub][0]] for sub in selected_subs])
    else:
        with open('data/repro_ephys_release.txt') as file:
            include_eids = [line.rstrip() for line in file]
        include_eids = include_eids[:args.n_sessions]
else:
    include_eids = args.query_session

# Trial setup
params = {
    'interval_len': 4, 'binsize': 0.02, 'single_region': False,
    'align_time': 'stimOn_times', 'time_window': (-1., 3.)
}

beh_names = [
    'choice', 'reward', 'block', 
    'wheel-speed', 'whisker-motion-energy', 
    #'pupil-diameter', # Some sessions do not have pupil traces
]

for eid_idx, eid in enumerate(include_eids):

    # try: 
    print('==========================')
    print(f'Preprocess session {eid}:')

    # Load and preprocess AP and behavior
    neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, bwm_df, params, n_workers=args.n_workers)
    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
    binned_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], n_workers=args.n_workers, **params
    )
    binned_behaviors, behavior_masks = bin_behaviors(
        one, eid, beh_names[3:], trials_df=trials_data['trials_df'], 
        allow_nans=True, n_workers=args.n_workers, **params
    )

    # Load and preprocess LFP
    lfp_prec = prepare_lfp(one, eid, **params)
    # bin_size in second; n_split = interval_len / bin_size, e.g., n_split = 2s/0.4s = 5
    binned_lfp = featurize_lfp(lfp_prec, n_split=10) 
    
    # Ensure neural and behavior data match for each trial
    aligned_binned_spikes, aligned_binned_lfp, aligned_binned_behaviors = align_spike_behavior(
        binned_spikes, binned_lfp, binned_behaviors, beh_names, trials_data['trials_mask']
    )

    # Partition dataset (train: 0.7 val: 0.1 test: 0.2)
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
        binned_behaviors=train_beh, meta_data=meta_data,
        binned_lfp=aligned_binned_lfp[train_idxs]
    )
    val_dataset = create_dataset(
        aligned_binned_spikes[val_idxs], bwm_df, eid, params, 
        binned_behaviors=val_beh, meta_data=meta_data,
        binned_lfp=aligned_binned_lfp[val_idxs]
    )
    test_dataset = create_dataset(
        aligned_binned_spikes[test_idxs], bwm_df, eid, params, 
        binned_behaviors=test_beh, meta_data=meta_data,
        binned_lfp=aligned_binned_lfp[test_idxs]
    )

    # Create dataset
    partitioned_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset}
    )
    print(partitioned_dataset)

    # Upload dataset
    upload_dataset(partitioned_dataset, org=args.huggingface_org, eid=f"{eid}_aligned_4s")

    print(f'Uploaded session {eid}.')
    print(f'Progress: {eid_idx+1} / {len(include_eids)} sessions uploaded.')
            
    # except Exception as e:
    #     print(f'Skipped session {eid} due to unexpected error: ', e)

