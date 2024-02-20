import numpy as np
from iblutil.numerical import bincount2D
from brainbox.population.decode import get_spike_counts_in_bins

from utils.ibl_data_loading import (
    load_spiking_data, load_trials_and_mask, merge_probes,
    load_discrete_behaviors, load_continuous_behaviors
)

def prepare_data(one, idx, bwm_df, params):
    
    # When merging probes we are interested in eids, not pids
    if params['merged_probes']:
        eid = bwm_df['eid'].unique()[idx]
        tmp_df = bwm_df.set_index(['eid', 'subject']).xs(eid, level='eid')
        subject = tmp_df.index[0]
        pids = tmp_df['pid'].to_list()  # Select all probes of this session
        probe_names = tmp_df['probe_name'].to_list()
        print(f"Running merged probes for session eid: {eid}")
    else:
        eid = bwm_df.iloc[idx]['eid']
        subject = bwm_df.iloc[idx]['subject']
        pid = bwm_df.iloc[idx]['pid']
        probe_name = bwm_df.iloc[idx]['probe_name']
        print(f"Running probe pid: {pid}")

    if params['merged_probes']:
        clusters_list = []
        spikes_list = []
        for pid, probe_name in zip(pids, probe_names):
            tmp_spikes, tmp_clusters = load_spiking_data(one, pid, eid=eid, pname=probe_name)
            tmp_clusters['pid'] = pid
            spikes_list.append(tmp_spikes)
            clusters_list.append(tmp_clusters)
        spikes, clusters = merge_probes(spikes_list, clusters_list)
    else:
        spikes, clusters = load_spiking_data(one, pid, eid=eid, pname=probe_name)

    trials, trials_mask = load_trials_and_mask(one=one, eid=eid)
    discrete_behaviors = load_discrete_behaviors(trials)
    continuous_behaviors = load_continuous_behaviors(one, eid)
    
    neural_dict = {
        'spike_times': spikes['times'],
        'spike_clusters': spikes['clusters'],
        'cluster_regions': clusters['acronym'],
        'cluster_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
        'cluster_df': clusters
    }
    
    behave_dict = {
        'discrete': discrete_behaviors,
        'continuous': continuous_behaviors
    }
    
    metadata = {
        'subject': subject,
        'eid': eid,
        'probe_name': probe_name,
        'trials': trials,
        'trials_mask': trials_mask
    }

    return neural_dict, behave_dict, metadata



    