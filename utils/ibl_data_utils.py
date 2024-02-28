import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d

from iblutil.numerical import ismember, bincount2D
import brainbox.behavior.dlc as dlc
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.regions import BrainRegions
from brainbox.population.decode import get_spike_counts_in_bins


def load_spiking_data(one, pid, compute_metrics=False, qc=None, **kwargs):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Function to load the cluster information and spike trains for clusters that may or may not pass certain quality metrics.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    pid: str
        A probe insertion UUID
    compute_metrics: bool
        If True, force SpikeSortingLoader.merge_clusters to recompute the cluster metrics. Default is False
    qc: float
        Quality threshold to be used to select good clusters. Default is None.
        If use all available clusters, set qc to None. If use good clusters, set qc to 1.
    kwargs:
        Keyword arguments passed to SpikeSortingLoader upon initiation. Specifically, if one instance offline,
        you need to pass 'eid' and 'pname' here as they cannot be inferred from pid in offline mode.

    Returns
    -------
    selected_spikes: dict
        Spike trains associated with clusters. Dictionary with keys ['depths', 'times', 'clusters', 'amps']
    selected_clusters: pandas.DataFrame
        Information of clusters for this pid 
    """
    eid = kwargs.pop('eid', '')
    pname = kwargs.pop('pname', '')
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics).to_df()
    if qc is None:
        selected_clusters = clusters_labeled
    else:
        iok = clusters_labeled['label'] >= qc
        selected_clusters = clusters_labeled[iok]

    spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
    selected_clusters.reset_index(drop=True, inplace=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)

    return selected_spikes, selected_clusters


def merge_probes(spikes_list, clusters_list):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Merge spikes and clusters information from several probes as if they were recorded from the same probe.
    This can be used to account for the fact that data from the probes recorded in the same session are not
    statistically independent as they have the same underlying behaviour.

    NOTE: The clusters dataframe will be re-indexed to avoid duplicated indices. Accordingly, spikes['clusters']
    will be updated. To unambiguously identify clusters use the column 'uuids'

    Parameters
    ----------
    spikes_list: list of dicts
        List of spike dictionaries as loaded by SpikeSortingLoader or brainwidemap.load_good_units
    clusters_list: list of pandas.DataFrames
        List of cluster dataframes as loaded by SpikeSortingLoader.merge_clusters or brainwidemap.load_good_units

    Returns
    -------
    merged_spikes: dict
        Merged and time-sorted spikes in single dictionary, where 'clusters' is adjusted to index into merged_clusters
    merged_clusters: pandas.DataFrame
        Merged clusters in single dataframe, re-indexed to avoid duplicate indices.
        To unambiguously identify clusters use the column 'uuids'
    """

    assert (len(clusters_list) == len(spikes_list)), 'clusters_list and spikes_list must have the same length'
    assert all([isinstance(s, dict) for s in spikes_list]), 'spikes_list must contain only dictionaries'
    assert all([isinstance(c, pd.DataFrame) for c in clusters_list]), 'clusters_list must contain only pd.DataFrames'

    merged_spikes = []
    merged_clusters = []
    cluster_max = 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes['clusters'] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()}
    # Sort spikes by spike time
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}

    return merged_spikes, merged_clusters


def load_trials_and_mask(
        one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', min_trial_len=None,
        max_trial_len=None, exclude_unbiased=False, exclude_nochoice=False, sess_loader=None):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Function to load all trials for a given session and create a mask to exclude all trials that have a reaction time
    shorter than min_rt or longer than max_rt or that have NaN for one of the specified events.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    eid: str
        A session UUID
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is 2. If None, don't apply.
    nan_exclude: list or 'default'
        List of trial events that cannot be NaN for a trial to be included. If set to 'default' the list is
        ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']
    min_trial_len: float or None
        Minimum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    max_trial_len: float or Nona
        Maximum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    exclude_unbiased: bool
        True to exclude trials that fall within the unbiased block at the beginning of session.
        Default is False.
    exclude_nochoice: bool
        True to exclude trials where the animal does not respond. Default is False.
    sess_loader: brainbox.io.one.SessionLoader or NoneType
        Optional SessionLoader object; if None, this object will be created internally

    Returns
    -------
    trials: pandas.DataFrame
        Trials table containing all trials for this session. If complete with columns:
        ['stimOff_times','goCueTrigger_times','feedbackType','contrastLeft','contrastRight','rewardVolume',
        'goCue_times','choice','feedback_times','stimOn_times','response_times','firstMovement_times',
        'probabilityLeft', 'intervals_0', 'intervals_1']
    mask: pandas.Series
        Boolean Series to mask trials table for trials that pass specified criteria. True for all trials that should be
        included, False for all trials that should be excluded.
    """

    if nan_exclude == 'default':
        nan_exclude = [
            'stimOn_times',
            'choice',
            'feedback_times',
            'probabilityLeft',
            'firstMovement_times',
            'feedbackType'
        ]

    if sess_loader is None:
        sess_loader = SessionLoader(one, eid)

    if sess_loader.trials.empty:
        sess_loader.load_trials()

    # Create a mask for trials to exclude
    # Remove trials that are outside the allowed reaction time range
    if min_rt is not None:
        query = f'(firstMovement_times - stimOn_times < {min_rt})'
    else:
        query = ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    # Remove trials that are outside the allowed trial duration range
    if min_trial_len is not None:
        query += f' | (feedback_times - goCue_times < {min_trial_len})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    # Remove trials with nan in specified events
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    # Remove trials in unbiased block at beginning
    if exclude_unbiased:
        query += ' | (probabilityLeft == 0.5)'
    # Remove trials where animal does not respond
    if exclude_nochoice:
        query += ' | (choice == 0)'
    # If min_rt was None we have to clean up the string
    if min_rt is None:
        query = query[3:]

    # Create mask
    mask = ~sess_loader.trials.eval(query)

    return sess_loader.trials, mask


def list_brain_regions(neural_dict, **kwargs):
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict['cluster_regions'], mapping='Beryl')
    regions = ([[k] for k in np.unique(beryl_reg)] if kwargs['single_region'] else [np.unique(beryl_reg)])
    print(f"Use spikes from brain regions: ", regions[0])
    return regions, beryl_reg


def select_brain_regions(regressors, beryl_reg, region, **kwargs):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Select units based on brain region.
    """
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def create_intervals(start_time, end_time, interval_len):
    interval_begs = np.arange(
        start_time, end_time-interval_len, interval_len
    )
    interval_ends = np.arange(
        start_time+interval_len, end_time, interval_len
    )
    return np.c_[interval_begs, interval_ends]


def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Select spiking data for specified interval in each recording.

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
        - (list): time in seconds for each interval; timepoints refer to the start/left edge of a bin
        - (list): data for each interval of shape (n_clusters, n_bins)

    """
    n_intervals = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_intervals, n_clusters_in_region, n_bins))
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(tqdm(zip(interval_begs, interval_ends), total=n_intervals)):
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


def bin_spiking_data(reg_clu_ids, neural_df, intervals=None, trials_df=None, **kwargs):
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
    intervals : 
        array of time intervals for each recording chunk including trials and non-trials
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
        each element is a 2D numpy.ndarray for a single interval of shape (n_bins, n_clusters)
    array
        cluster ids that account for axis 1 of the above 2D arrays.
    """

    if trials_df is not None:
        # compute time intervals for each trial
        intervals = np.vstack([
            trials_df[kwargs['align_time']] + kwargs['time_window'][0],
            trials_df[kwargs['align_time']] + kwargs['time_window'][1]
        ]).T
        chunk_len = kwargs['time_window'][1] - kwargs['time_window'][0]
        interval_len = (
            kwargs['time_window'][1] - kwargs['time_window'][0]
        )
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        chunk_len = intervals[0,1] - intervals[0,0]
        interval_len = (
            intervals[0,1] - intervals[0,0]
        )

    # subselect spikes for this region
    spikemask = np.isin(neural_df['spike_clusters'], reg_clu_ids)
    regspikes = neural_df['spike_times'][spikemask]
    regclu = neural_df['spike_clusters'][spikemask]
    clusters_used_in_bins = np.unique(regclu)

    binsize = kwargs.get('binsize', chunk_len)
    
    if chunk_len / binsize == 1.0:
        # one vector of neural activity per interval
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        spike_times_list, binned_array = get_spike_data_per_interval(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs['binsize'])
        binned_list = [x.T for x in binned_array]

    return np.array(binned_list), clusters_used_in_bins


def load_target_behavior(one, eid, target):
    """

    Parameters
    ----------
    target : str
        'wheel-position' | 'wheel-velocity' | 'wheel-speed' | 
        'left-whisker-motion-energy' | 'right-whisker-motion-energy' | 
        'left-pupil-diameter' | 'right-pupil-diameter' |
        'left-camera-left-paw-speed' | 'left-camera-right-paw-speed' | 
        'right-camera-left-paw-speed' | 'right-camera-right-paw-speed' |
        'left-nose-speed' | 'right-nose-speed'
    one : 
    eid : str

    Returns
    -------
    dict
        'times': timestamps for behavior signal
        'values': associated values
        'skip': bool, True if there was an error loading data
    """

    # To load wheel and motion energy, we just use the SessionLoader, e.g.
    sess_loader = SessionLoader(one, eid)
    
    # wheel is a dataframe that contains wheel times and position interpolated to a uniform sampling rate, velocity and
    # acceleration computed using Gaussian smoothing
    try:
        if target == 'wheel-position':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['position'].to_numpy()
            }
        elif target == 'wheel-velocity':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['velocity'].to_numpy()
            }
        elif target == 'wheel-speed':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': np.abs(sess_loader.wheel['velocity'].to_numpy())
            }
    
        # motion_energy is a dictionary of dataframes, each containing the times and the motion energy for each view
        # for the side views, they contain columns ['times', 'whiskerMotionEnergy'] for the body view it contains
        # ['times', 'bodyMotionEnergy']
        elif target == 'left-whisker-motion-energy':
            sess_loader.load_motion_energy(views=['left'])
            beh_dict = {
                'times': sess_loader.motion_energy['leftCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy'].to_numpy()
            }
        elif target == 'right-whisker-motion-energy':
            sess_loader.load_motion_energy(views=['right'])
            beh_dict = {
                'times': sess_loader.motion_energy['rightCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['rightCamera']['whiskerMotionEnergy'].to_numpy()
            }
    
        # To load pose (DLC) data, e.g.
        # TO DO: Add pupil traces from lightning pose when they become available in the IBL database, e.g.,
        #        sessions = one.search(dataset='lightningPose', details=False)
        #        pupil_data = one.load_object(eid, f'leftCamera', attribute=['lightningPose', 'times'])
        # TO DO: Sometimes some traces are unavailable. Right now we still load them as 'nan' but need to handle it later.
        # TO DO: Different cameras have very different traces for the same behavior. Treat them as independent? 
        elif target == 'left-pupil-diameter':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc_left.features.pupilDiameter_smooth
            }
        elif target == 'right-pupil-diameter':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc_right.features.pupilDiameter_smooth
            }
        elif target == 'left-camera-left-paw-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_l")
            }
        elif target == 'left-camera-right-paw-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_r")
            }
        elif target == 'right-camera-left-paw-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_l")
            }
        elif target == 'right-camera-right-paw-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_r")
            }
        elif target == 'left-nose-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="nose_tip")
            }
        elif target == 'right-nose-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="nose_tip")
            }
        else:
            raise NotImplementedError
    except BaseException as e:
        print('Error loading %s data' % target)
        print(e)
        beh_dict = {'times': None, 'values': None, 'skip': True}
 
    return beh_dict


def get_behavior_per_interval(target_times, target_vals, intervals=None, trials_df=None, allow_nans=False, **kwargs):
    """
    (Code adapted from: https://github.com/int-brain-lab/paper-brain-wide-map)
    Format a single session-wide array of target data into a list of interval-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `align_interval`; for example if `align_interval=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `align_interval=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_vals : array-like
        data samples
    intervals : 
        array of time intervals for each recording chunk including trials and non-trials
    trials_df : pd.DataFrame
        requires a column that matches `align_event`
    align_event : str
        event to align interval to
        firstMovement_times | stimOn_times | feedback_times
    align_interval : tuple
        (align_begin, align_end); time in seconds relative to align_event
    binsize : float
        size of individual bins in interval
    allow_nans : bool, optional
        False to skip intervals with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each interval
        - (list): data for each interval
        - (array-like): mask of good intervals (True) and bad intervals (False)

    """

    binsize = kwargs['binsize']
    align_interval = kwargs['time_window']
    interval_len = align_interval[1] - align_interval[0]

    if trials_df is not None:
        align_event = kwargs['align_time']
        align_times = trials_df[align_event].values
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        interval_begs, interval_ends = intervals.T

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print('interval times all nan')
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        target_times_list = []
        target_vals_list = []
        return target_times_list, target_vals_list, good_interval

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = []
    target_vals_list = []
    good_interval = [None for _ in range(len(target_times_og_list))]

    def skip_interval(index, skip_reason):
        # print(skip_reason)
        good_interval[index] = False
        target_times_list.append(None)
        target_vals_list.append(None)
        return good_interval, target_times_list, target_vals_list
    
    for i, (target_time, target_vals) in enumerate(zip(target_times_og_list, target_vals_og_list)):

        if len(target_vals) == 0:
            skip_reason = 'target data not present'
            good_interval, target_times_list, target_vals_list = skip_interval(i, skip_reason)
            continue
        if np.sum(np.isnan(target_vals)) > 0 and not allow_nans:
            skip_reason = 'nans in target data'
            good_interval, target_times_list, target_vals_list = skip_interval(i, skip_reason)
            continue
        if np.isnan(interval_begs[i]) or np.isnan(interval_ends[i]):
            skip_reason = 'bad interval data'
            good_interval, target_times_list, target_vals_list = skip_interval(i, skip_reason)
            continue
        if np.abs(interval_begs[i] - target_time[0]) > binsize:
            skip_reason = 'target data starts too late'
            good_interval, target_times_list, target_vals_list = skip_interval(i, skip_reason)
            continue
        if np.abs(interval_ends[i] - target_time[-1]) > binsize:
            # print('target data ends too early on interval %i; skipping' % i)
            skip_reason = 'target data ends too early'
            good_interval, target_times_list, target_vals_list = skip_interval(i, skip_reason)
            continue

        # resample signal in desired bins
        # using `interval_begs[i] + binsize` forces the interpolation to sample the continuous
        # signal at the *end* (or right side) of each bin; this way the spikes in a given bin will
        # fully precede the corresponding target sample for that same bin.
        x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)
        if len(target_vals.shape) > 1 and target_vals.shape[1] > 1:
            n_dims = target_vals.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(interp1d(
                    target_time, target_vals[:, n], kind='linear',
                    fill_value='extrapolate')(x_interp))
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(
                target_time, target_vals, kind='linear', fill_value='extrapolate')(x_interp)

        target_times_list.append(x_interp)
        target_vals_list.append(y_interp)
        good_interval[i] = True

    return target_times_list, target_vals_list, np.array(good_interval)
    


def load_anytime_behaviors(one, eid):

    behaviors = [
        'wheel-position', 'wheel-velocity', 'wheel-speed',
        'left-whisker-motion-energy', 'right-whisker-motion-energy',
        'left-pupil-diameter', 'right-pupil-diameter',
        'left-camera-left-paw-speed', 'left-camera-right-paw-speed', 
        'right-camera-left-paw-speed', 'right-camera-right-paw-speed',
        'left-nose-speed', 'right-nose-speed'
    ]
    
    behave_dict = {}
    for beh in behaviors:
        behave_dict.update({beh: load_target_behavior(one, eid, beh)})
        
    return behave_dict


def bin_behaviors(one, eid, intervals=None, trials_only=False, trials_df=None, mask=None, allow_nans=True, **kwargs):

    behaviors = [
        'wheel-position', 'wheel-velocity', 'wheel-speed',
        'left-whisker-motion-energy', 'right-whisker-motion-energy',
        'left-pupil-diameter', 'right-pupil-diameter',
        'left-camera-left-paw-speed', 'left-camera-right-paw-speed', 
        'right-camera-left-paw-speed', 'right-camera-right-paw-speed',
        'left-nose-speed', 'right-nose-speed'
    ]

    behave_dict = {}
    
    if mask is not None:
        trials_df = trials_df[mask]

    if trials_only:
        assert trials_df is not None, 'Require trials data frame to segment the recording into trials only.'   
        
        choice = trials_df['choice'].to_numpy()
        block = trials_df['probabilityLeft'].to_numpy()
        reward = (trials_df['rewardVolume'] > 1).astype(int).to_numpy()
        contrast = np.c_[trials_df['contrastLeft'], trials_df['contrastRight']]
        contrast = (-1 * np.nan_to_num(contrast, 0)).sum(1)

        behave_dict.update(
            {'choice': choice, 'block': block, 'reward': reward, 'contrast': contrast}
        )
        behave_mask = np.ones(len(trials_df)) 
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        behave_mask = np.ones(len(intervals)) 
    
    # Bin anytime behaviors for each trial
    for beh in behaviors:
        target_dict = load_target_behavior(one, eid, beh)
        target_times, target_vals = target_dict['times'], target_dict['values']
        target_times_list, target_vals_list, target_mask = get_behavior_per_interval(
            target_times, target_vals, intervals=intervals, trials_only=trials_only,
            trials_df=trials_df, allow_nans=allow_nans, **kwargs
        )
        behave_dict.update({beh: np.array(target_vals_list, dtype=object)})
        behave_mask = np.logical_and(behave_mask, target_mask)

    if not allow_nans:
        for k, v in behave_dict.items():
            behave_dict[k] = behave_dict[beh][behave_mask]
    
    return behave_dict


def prepare_data(one, eid, bwm_df, params):
    
    # When merging probes we are interested in eids, not pids
    idx = (bwm_df.eid.unique() == eid).argmax()
    eid = bwm_df.eid.unique()[idx]
    tmp_df = bwm_df.set_index(['eid', 'subject']).xs(eid, level='eid')
    subject = tmp_df.index[0]
    lab = tmp_df.lab.iloc[0]
    
    pids = tmp_df['pid'].to_list()  # Select all probes of this session
    probe_names = tmp_df['probe_name'].to_list()

    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters = load_spiking_data(one, pid, eid=eid, pname=probe_name)
        tmp_clusters['pid'] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)
    print(f"Merged {len(probe_names)} probes for session eid: {eid}")

    trials_df, trials_mask = load_trials_and_mask(one=one, eid=eid)
        
    anytime_behaviors = load_anytime_behaviors(one, eid)
    
    neural_dict = {
        'spike_times': spikes['times'],
        'spike_clusters': spikes['clusters'],
        'cluster_regions': clusters['acronym'].to_numpy(),
        # We don't need details about the cluster QC. Only include if good units for now.
        # 'cluster_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
        # 'cluster_df': clusters
    }
    
    behave_dict = anytime_behaviors
    
    meta_data = {
        'subject': subject,
        'eid': eid,
        'probe_name': probe_name,
        'lab': lab,
        'cluster_channels': list(clusters['channels']),
        'cluster_regions': list(clusters['acronym']),
        'good_clusters': list((clusters['label'] >= 1).astype(int))
        # TO DO: Add sampling frequency
    }

    trials_data = {
        'trials_df': trials_df,
        'trials_mask': trials_mask
    }

    return neural_dict, behave_dict, meta_data, trials_data
    


