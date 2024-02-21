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


def select_brain_regions(regressors, beryl_reg, region, **kwargs):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Select units based on brain region.
    """
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def get_spike_data_per_trial(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Select spiking data for specified interval on each trial.

    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    interval_len : float
    binsize : float
        width of each bin in seconds

    Returns
    -------
    tuple
        - (list): time in seconds for each trial; timepoints refer to the start/left edge of a bin
        - (list): data for each trial of shape (n_clusters, n_bins)

    """
    n_trials = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_trials, n_clusters_in_region, n_bins))
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)

        # update data block
        binned_spikes[tr, idxs_tmp, :] += binned_spikes_tmp[:, :n_bins]
        spike_times_list.append(t_idxs[:n_bins])

    return spike_times_list, binned_spikes


def bin_spiking_data(reg_clu_ids, neural_df, trials_df, **kwargs):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Format a single session-wide array of spikes into a list of trial-based arrays.
    The ordering of clusters used in the output are also returned.

    Parameters
    ----------
    reg_clu_ids : array-like
        array of cluster ids for each spike
    neural_df : pd.DataFrame
        keys: 'spike_times', 'spike_clusters', 'cluster_regions', 'cluster_qc', 'cluster_df'
    trials_df : pd.DataFrame
        columns: 'choice', 'feedback', 'pLeft', 'firstMovement_times', 'stimOn_times',
        'feedback_times'
    kwargs
        align_time : str
            event in trial on which to align intervals
            'firstMovement_times' | 'stimOn_times' | 'feedback_times'
        time_window : tuple
            (window_start, window_end), relative to align_time
        binsize : float, optional
            size of bins in seconds for multi-bin decoding
            
    Returns
    -------
    list
        each element is a 2D numpy.ndarray for a single trial of shape (n_bins, n_clusters)
    array
        cluster ids that account for axis 1 of the above 2D arrays.
    """

    # compute time intervals for each trial
    intervals = np.vstack([
        trials_df[kwargs['align_time']] + kwargs['time_window'][0],
        trials_df[kwargs['align_time']] + kwargs['time_window'][1]
    ]).T

    # subselect spikes for this region
    spikemask = np.isin(neural_df['spike_clusters'], reg_clu_ids)
    regspikes = neural_df['spike_times'][spikemask]
    regclu = neural_df['spike_clusters'][spikemask]
    clusters_used_in_bins = np.unique(regclu)

    # for each trial, put spiking data into a 2D array; collect trials in a list
    trial_len = kwargs['time_window'][1] - kwargs['time_window'][0]
    binsize = kwargs.get('binsize', trial_len)
    # TODO: can likely combine get_spike_counts_in_bins and get_spike_data_per_trial
    # added second condition in if statement if n_bins_lag is None, gives error otherwise (bensonb)
    if trial_len / binsize == 1.0:
        # one vector of neural activity per trial
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]

    else:
        # multiple vectors of neural activity per trial
        # moved interval_len definintion into this condition so that when n_bins_lag is None it doesn't cause error
        interval_len = (
            kwargs['time_window'][1] - kwargs['time_window'][0]
        )
        spike_times_list, binned_array = get_spike_data_per_trial(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs['binsize'])
        binned_list = [x.T for x in binned_array]

    return binned_list, clusters_used_in_bins


def bin_behaviors():
    pass


    