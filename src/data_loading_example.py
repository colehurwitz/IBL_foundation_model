import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from utils.ibl_data_utils import (
    load_spiking_data, load_trials_and_mask, merge_probes,
    load_trial_behaviors, load_anytime_behaviors,
    prepare_data, 
    select_brain_regions, list_brain_regions, 
    bin_spiking_data, 
)
from utils.dataset import create_dataset, upload_dataset, download_dataset, get_binned_spikes_from_sparse

# Instantiate ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', 
          silent=True, 
          cache_dir='../checkpoints/one_cache',
)
one = ONE(password='international')
ba = AllenAtlas()

# List all publicly available sessions
# sessions = one.search()

# List all sessions from the brain-wide-map project
# Note: Not all sessions will have behaviors recorded so we can use sessions 
#       w/ behaviors for supervised tasks and those w/o for SSL
# sessions = one.search(project='brainwide')

# List brainwide map sessions that pass the most important quality controls
# Note: Let's first work with these good quality sessions, and 
#       transition to using more sessions later on
freeze_file = '../data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

# Load an example session
idx = 400
pid = bwm_df.pid[idx]
eid, probe = one.pid2eid(pid)
print(f"Session {idx} has PID {pid} and EID {eid}")

# Load spike sorting data
spikes, clusters = load_spiking_data(one, pid)

print("Spikes Keys:", spikes.keys())
# for k, v in spikes.items():
#     print(k, v.shape)
#     print(v[:10])
# exit()

# We may not want to train the model with data within trials, but we may need it for eval purposes.
# Load trials data and mask. Trials are excluded in the mask if reaction time is too long or too short,
# or if critical trial events are missing.
trials, mask = load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default')

# Load behaviors for any-time decoding
anytime_behaviors = load_anytime_behaviors(one, eid)

params = {
    # setup for trial decoding:
    'align_time': 'stimOn_times',
    'time_window': (-.5, 1.5),
    'binsize': 0.02,
    'single_region': False # use all available regions
}

neural_dict, behave_dict, metadata = prepare_data(one, eid, bwm_df, params)

print("neural_dict keys:", neural_dict.keys())
print("behave_dict keys:", behave_dict.keys())
print("metadata keys:", metadata.keys())

regions, beryl_reg = list_brain_regions(neural_dict, **params)
print("Regions:", regions)
# Use spikes from brain regions:  ['CA1' 'CP' 'DG' 'LP' 'MOp' 'OT' 'PIR' 'PO' 'PoT' 'SI' 'VISa' 'root']
region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

# 'binned_spikes' is very sparse - how to handle the tokenization more efficiently?
binned_spikes, clusters_used_in_bins = bin_spiking_data(region_cluster_ids, neural_dict, trials, **params)

trial_idx = 0
plt.figure(figsize=(4,3))
plt.imshow(binned_spikes[trial_idx].T, aspect="auto", cmap="binary")
plt.title(f"spike count in trial {trial_idx}")
plt.xlabel("time bin")
plt.ylabel("unit")
plt.colorbar()
plt.savefig("spike_count_trial_{}.png".format(trial_idx))

total_elems = binned_spikes[trial_idx].shape[0] * binned_spikes[trial_idx].shape[1]
nonzero_elems = np.count_nonzero(binned_spikes[trial_idx])
print(f"sparsity: {nonzero_elems / total_elems:.2f}")

binned_behaviors = load_trial_behaviors(one, eid, trials, allow_nans=True, **params)
binned_behaviors.keys()

behave = "wheel-speed"
plt.figure(figsize=(4,3))
plt.plot(binned_behaviors[behave][trial_idx])
plt.title(f"{behave} in trial {trial_idx}")
plt.xlabel("time bin")
plt.ylabel(behave)
plt.savefig(f"{behave}_trial_{trial_idx}.png")

dataset = create_dataset(binned_spikes, bwm_df, idx, eid, probe, region_cluster_ids, beryl_reg, params["binsize"], binned_behaviors=None, metadata=None)
# upload_dataset(dataset)
# dataset = download_dataset()

spikes_sparse_data_list = dataset['train']['spikes_sparse_data']
spikes_sparse_indices_list = dataset['train']['spikes_sparse_indices']
spikes_sparse_indptr_list = dataset['train']['spikes_sparse_indptr']
spikes_sparse_shape_list = dataset['train']['spikes_sparse_shape']

downloaded_binned_spikes = get_binned_spikes_from_sparse(spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list)

print(f"Shape of downloaded binned spikes: {downloaded_binned_spikes.shape}. Shape of original binned spikes: {binned_spikes.shape}")