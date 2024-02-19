import numpy as np
import pandas as pd
from pathlib import Path

from iblutil.numerical import ismember
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from ibllib.atlas.regions import BrainRegions


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
    if not qc:
        selected_clusters = clusters_labeled
    else:
        iok = clusters_labeled['label'] >= qc
        selected_clusters = clusters_labeled[iok]

    spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
    selected_clusters.reset_index(drop=True, inplace=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)

    return selected_spikes, selected_clusters



