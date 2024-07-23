import sys
from os import symlink
import pickle
import tempfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from one.api import ONE
import scipy
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import neuropixel
from brainbox.io.one import SpikeSortingLoader
from ibldsp.utils import rms, fcn_cosine
from ibldsp.waveforms import compute_spike_features
import spikeglx
import spikeinterface.preprocessing as si
from spikeinterface.extractors.iblextractors import IblRecordingExtractor
from spikeinterface.preprocessing import phase_shift
from utils.ibl_data_utils import load_trials_and_mask

band = 'lp'
BANDS = {
    'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12], 'beta': [15, 30], 'gamma': [30, 90], 'lfp': [0, 90]
}

# ----------------
# helper functions
# ----------------

def _get_power_in_band(fscale, period, band):
    band = np.array(band)
    fweights = fcn_cosine([-np.diff(band), 0])(-abs(fscale - np.mean(band)))
    p = 10 * np.log10(np.sum(period * fweights / np.sum(fweights), axis=-1))  # dB relative to v/sqrt(Hz)
    return p

def lf(data, fs, bands=None):
    """Computes the LF features from a numpy array.
    
    :param data: numpy array with the data (channels, samples)
    :param fs: sampling interval (Hz)
    :param bands: dictionary with the bands to compute (default: BANDS constant)
    :return: pandas dataframe with the columns ['channel', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
       'psd_gamma', 'psd_lfp']
    """
    bands = BANDS if bands is None else bands
    nc = data.shape[0]  # number of channels
    fscale, period = scipy.signal.periodogram(data, fs)
    df_chunk = pd.DataFrame()
    df_chunk['channel'] = np.arange(nc)
    df_chunk['rms_lf'] = rms(data, axis=-1)
    for b in BANDS:
        df_chunk[f"psd_{b}"] = _get_power_in_band(fscale, period, bands[b])
    return df_chunk['psd_lfp']

# ------------------
# load & preprocess
# ------------------

def prepare_lfp(
    one, eid, mask=None, fs=2500.0, **kwargs
):
    """Load, preprocess, and merge LFP from both probes.
    
    Args:
        mask: used to filter out trials; need to be consistent with mask for AP and behavior. 
    """
    trial_window = kwargs["time_window"]
    align_time = kwargs["align_time"]
    
    pids, probes = one.eid2pid(eid)
    ssl = SpikeSortingLoader(pid=pids[0], one=one)
    stimOn_times = one.load_object(eid, 'trials', collection='alf')[align_time]
    if mask:
        stimOn_times = stimOn_times[mask]

    detect_kwargs = {
        "lf": {"fs": fs, "psd_hf_threshold": 1.4, 'similarity_threshold': (-0.25, 1)},
    }

    lfp_per_probe = []
    for idx in range(len(pids)):
        pid, probe = pids[0], probes[0]
        print(probe)
        
        rec_si_stream = IblRecordingExtractor(pid=pid, stream_name=f"{probe}.lf", one=one, stream=True)    
        rec_phs = phase_shift(rec_si_stream) # channel rephasing

        rec_bp = si.bandpass_filter(rec_phs, freq_min=0.5, freq_max=250)
        bad_chans, labels = si.detect_bad_channels(rec_bp, psd_hf_threshold=1.4, num_random_chunks=100, seed=0)
    
        lfp_per_trial = []
        for trial_idx in tqdm(range(len(stimOn_times)), total=len(stimOn_times)):
            t_event = stimOn_times[trial_idx]
            s_event = int(ssl.samples2times(t_event, direction='reverse'))
        
            # for NP probes always 12 because AP is sampled at 12x the frequency of LF
            sample_lf = s_event // 12
            first, last = (
                int(trial_window[0] * fs) + sample_lf, 
                int(trial_window[1] * fs + sample_lf)
            )
            chunk = rec_bp.frame_slice(start_frame=first, end_frame=last) 
            rec_prec = si.interpolate_bad_channels(chunk, bad_chans)
            rec_prec = si.common_reference(rec_prec, reference='global', operator='median')
            lfp_per_trial.append(rec_prec.get_traces().T)
        
        lfp_per_trial = np.array(lfp_per_trial)
        lfp_per_trial = lfp_per_trial.transpose(0,2,1)
        
    lfp_per_probe.append(lfp_per_trial)
    lfp_per_probe = np.stack(lfp_per_probe, axis=-1).squeeze()
    print("Preprocessed LFP data shape: ", lfp_per_probe.shape)
    
    return lfp_per_probe

# ------------------
# feature extraction
# ------------------

def featurize_lfp(lfp_data, n_split=None, bin_size=0.4, samp_freq=2500):
    """Extract Power spectral density for LFP.
    
    Args:
        lfp_data: preprocessed LFP data from prepare_lfp().
        bin_size: millisecond (ms).
        samp_freq: sampling frequency. 
    """
    n_trials, n_time_samples, _ = lfp_data.shape
    interval_len = n_time_samples // samp_freq
    if not n_split:
        n_split = int(interval_len // bin_size)
    print(f"Split {interval_len}-second LFP data into {n_split} time bin of size {bin_size}.")
    lfp_data = np.array(np.split(lfp_data, n_split, axis=1))
    lfp_data = np.transpose(lfp_data, (1, 3, 0, 2))
    
    power_bands = []
    for trial_idx in tqdm(range(n_trials), total=n_trials):
        tmp = []
        for tbin_idx in range(n_split):
            tmp.append(
                np.array(
                    lf(lfp_data[trial_idx,:,tbin_idx,:], fs=samp_freq)
                )
            )
        power_bands.append(tmp)
        
    power_bands = np.transpose(power_bands, (0,2,1))
    print("Featurized power bands shape: ", power_bands.shape)
    return power_bands



