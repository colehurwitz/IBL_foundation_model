import numpy as np
from scipy.sparse import csr_array
from datasets import Dataset, DatasetInfo
from datasets import load_dataset

def get_sparse_from_binned_spikes(binned_spikes):
    sparse_binned_spikes = [csr_array(binned_spikes[i], dtype=np.ubyte) for i in range(binned_spikes.shape[0])]

    spikes_sparse_data_list = [csr_matrix.data.tolist() for csr_matrix in sparse_binned_spikes] 
    spikes_sparse_indices_list = [csr_matrix.indices.tolist() for csr_matrix in sparse_binned_spikes]
    spikes_sparse_indptr_list = [csr_matrix.indptr.tolist() for csr_matrix in sparse_binned_spikes]
    spikes_sparse_shape_list = [csr_matrix.shape for csr_matrix in sparse_binned_spikes]

    return sparse_binned_spikes, spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list

def get_binned_spikes_from_sparse(spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list):
    sparse_binned_spikes = [csr_array((spikes_sparse_data_list[i], spikes_sparse_indices_list[i], spikes_sparse_indptr_list[i]), shape=spikes_sparse_shape_list[i]) for i in range(len(spikes_sparse_data_list))]

    binned_spikes = np.array([csr_matrix.toarray() for csr_matrix in sparse_binned_spikes])

    return binned_spikes

def create_dataset(
    binned_spikes, 
    bwm_df, 
    eid, 
    params, 
    meta_data=None, 
    binned_behaviors=None,
    intervals=None
):

    # Scipy sparse matrices can't be directly loaded into HuggingFace Datasets so they are converted to lists
    sparse_binned_spikes, spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list = get_sparse_from_binned_spikes(binned_spikes)

    data_dict = {
        'spikes_sparse_data': spikes_sparse_data_list,
        'spikes_sparse_indices': spikes_sparse_indices_list,
        'spikes_sparse_indptr': spikes_sparse_indptr_list,
        'spikes_sparse_shape': spikes_sparse_shape_list,
    }
    
    if binned_behaviors is not None:
        # Store choice behaviors more efficiently (save this option for later)
        # binned_behaviors["choice"] = np.where(binned_behaviors["choice"] > 0, 0, 1).astype(bool)
        data_dict = data_dict | binned_behaviors
        
    if meta_data is not None:
        meta_dict = {
            'binsize': [params['binsize']] * len(sparse_binned_spikes),
            'interval_len': [params['interval_len']] * len(sparse_binned_spikes),
            'eid': [meta_data['eid']] * len(sparse_binned_spikes),
            'probe_name': [meta_data['probe_name']] * len(sparse_binned_spikes),
            'subject': [meta_data['subject']] * len(sparse_binned_spikes),
            'lab': [meta_data['lab']] * len(sparse_binned_spikes),
            'sampling_freq': [meta_data['sampling_freq']] * len(sparse_binned_spikes),
            'cluster_regions': [meta_data['cluster_regions']] * len(sparse_binned_spikes),
            'cluster_channels': [meta_data['cluster_channels']] * len(sparse_binned_spikes),
            'cluster_depths': [meta_data['cluster_depths']] * len(sparse_binned_spikes),
            'good_clusters': [meta_data['good_clusters']] * len(sparse_binned_spikes),
            'cluster_uuids': [meta_data['uuids']] * len(sparse_binned_spikes),
        }
        data_dict = data_dict | meta_dict

    if intervals is not None:
        time_dict = {'intervals': intervals}
        data_dict = data_dict | time_dict

    return Dataset.from_dict(data_dict)

def upload_dataset(dataset, org, eid, is_private=True):
    dataset.push_to_hub(f"{org}/{eid}", private=is_private)

def download_dataset(org, eid, split="train", cache_dir=None):
    if cache_dir is None:
        return load_dataset(f"{org}/{eid}", split=split)
    else:
        return load_dataset(f"{org}/{eid}", split=split, cache_dir=cache_dir)

