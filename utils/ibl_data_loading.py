import numpy as np
import pandas as pd
from pathlib import Path
from iblutil.numerical import ismember
import brainbox.behavior.dlc as dlc
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.regions import BrainRegions


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


def load_discrete_behaviors(trials, mask=None):
    
    if mask is not None:
        trials = trials[mask]

    choice = trials['choice']
    block = trials['probabilityLeft']
    reward = (trials['rewardVolume'] > 1).astype(int)
    contrast = np.c_[trials['contrastLeft'], trials['contrastRight']]
    contrast = (-1 * np.nan_to_num(contrast, 0)).sum(1)

    behaviors = {'choice': choice, 'block': block, 'reward': reward, 'contrast': contrast}
    return behaviors


def load_continuous_behaviors(one, eid):
    
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_wheel(fs=1000)

    # To load wheel and motion energy, we just use the SessionLoader, e.g.
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_wheel(fs=1000)
    # wheel is a dataframe that contains wheel times and position interpolated to a uniform sampling rate, velocity and
    # acceleration computed using Gaussian smoothing
    wheel = sess_loader.wheel
    wheel_pos = np.c_[wheel.times, wheel.position]
    wheel_vel = np.c_[wheel.times, wheel.velocity]

    # motion_energy is a dictionary of dataframes, each containing the times and the motion energy for each view
    # for the side views, they contain columns ['times', 'whiskerMotionEnergy'] for the body view it contains
    # ['times', 'bodyMotionEnergy']
    sess_loader.load_motion_energy(views=['left', 'right', 'body'])
    left_whisker = sess_loader.motion_energy['leftCamera'].to_numpy()
    right_whisker = sess_loader.motion_energy['rightCamera'].to_numpy()
    body_whisker = sess_loader.motion_energy['bodyCamera'].to_numpy()

    # To load pose (DLC) data, e.g.
    # TO DO: Add pupil traces from lightning pose when they become available in the IBL database, e.g.,
    #        sessions = one.search(dataset='lightningPose', details=False)
    #        pupil_data = one.load_object(eid, f'leftCamera', attribute=['lightningPose', 'times'])
    # TO DO: Sometimes some traces are unavailable. Right now we still load them as 'nan' but need to handle it later.
    # TO DO: Different cameras have very different traces for the same behavior. Treat them as independent? 
    dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
    dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
    dlc_body = one.load_object(eid, "bodyCamera", attribute=["dlc", "features", "times"], collection="alf")
    left_pupil = np.c_[dlc_left.times, dlc_left.features.pupilDiameter_smooth]
    right_pupil = np.c_[dlc_right.times, dlc_right.features.pupilDiameter_smooth]
    # 'left_paw_l' means left paw speed from the left camera
    left_paw_l = np.c_[dlc_left.times, dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_l")]
    left_paw_r = np.c_[dlc_left.times, dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_r")]
    right_paw_l = np.c_[dlc_right.times, dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_l")]
    right_paw_r = np.c_[dlc_right.times, dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_r")]
    left_nose = np.c_[dlc_left.times, dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="nose_tip")]
    right_nose = np.c_[dlc_right.times, dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="nose_tip")]
    
    behaviors = {
        'wheel_pos':wheel_pos, 'wheel_vel': wheel_vel, 
        'left_whisker': left_whisker, 'right_whisker': right_whisker, 'body_whisker': body_whisker,
        'left_pupil': left_pupil, 'right_pupil': right_pupil, 
        'left_paw_l': left_paw_l, 'left_paw_r': left_paw_r, 'right_paw_l': right_paw_l, 'right_paw_r': right_paw_r,
        'left_nose': left_nose, 'right_nose': right_nose 
    }
    return behaviors





