import numpy as np
from scipy.sparse import csr_array
from datasets import Dataset
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

def create_dataset(binned_spikes, bwm_df, chosen_idx, eid, probe, region_cluster_ids, beryl_reg, bin_size, binned_behaviors=None, metadata=None):
    session_number = bwm_df.session_number[chosen_idx]
    date = bwm_df.date[chosen_idx]
    subject = bwm_df.subject[chosen_idx]
    lab = bwm_df.lab[chosen_idx]

    # Scipy sparse matrices can't be directly loaded into HuggingFace Datasets so they are converted to lists
    sparse_binned_spikes, spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list = get_sparse_from_binned_spikes(binned_spikes)

    data_dict = {
        'spikes_sparse_data': spikes_sparse_data_list,
        'spikes_sparse_indices': spikes_sparse_indices_list,
        'spikes_sparse_indptr': spikes_sparse_indptr_list,
        'spikes_sparse_shape': spikes_sparse_shape_list,

        'region_cluster_ids': [region_cluster_ids] * len(sparse_binned_spikes),
        'brain_regions': [beryl_reg] * len(sparse_binned_spikes),
        'bin_size': [bin_size] * len(sparse_binned_spikes),
        'eid': [eid] * len(sparse_binned_spikes),
        'probe_name': [probe] * len(sparse_binned_spikes),
        'session_number': [session_number] * len(sparse_binned_spikes),
        'date': [date] * len(sparse_binned_spikes),
        'subject': [subject] * len(sparse_binned_spikes),
        'lab': [lab] * len(sparse_binned_spikes),
    }
    if binned_behaviors is not None:
        # Store choice behaviors more efficiently
        binned_behaviors["choice"] = np.where(binned_behaviors["choice"] > 0, 0, 1).astype(bool)
        
        data_dict = data_dict | binned_behaviors
    if metadata is not None:
        data_dict = data_dict | metadata.to_dict(orient='list')

    return Dataset.from_dict(data_dict)

def upload_dataset(dataset):
    dataset.push_to_hub("berkott/ibl_ssl_data")

def download_dataset():
    return load_dataset("berkott/ibl_ssl_data")
