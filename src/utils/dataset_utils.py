import numpy as np
from scipy.sparse import csr_array
from datasets import Dataset, DatasetInfo
from datasets import load_dataset
import h5py
import torch

class DATASET_MODES:
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"

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

def get_data_from_h5(mode, filepath, config):
    r"""
        returns:
            spikes
            rates (None if not available)
            held out spikes (for cosmoothing, None if not available)
        * Note, rates and held out spikes codepaths conflict
    """

    has_rates = False
    NLB_KEY = 'spikes' # curiously, old code thought NLB data keys came as "train_data_heldin" and not "train_spikes_heldin"
    NLB_KEY_ALT = 'data'

    with h5py.File(filepath, 'r') as h5file:
        h5dict = {key: h5file[key][()] for key in h5file.keys()}
        if f'eval_{NLB_KEY}_heldin' not in h5dict: # double check
            if f'eval_{NLB_KEY_ALT}_heldin' in h5dict:
                NLB_KEY = NLB_KEY_ALT
        if f'eval_{NLB_KEY}_heldin' in h5dict: # NLB data, presumes both heldout neurons and time are available
            get_key = lambda key: h5dict[key].astype(np.float32)
            train_data = get_key(f'train_{NLB_KEY}_heldin')
            train_data_fp = get_key(f'train_{NLB_KEY}_heldin_forward')
            train_data_heldout_fp = get_key(f'train_{NLB_KEY}_heldout_forward')
            train_data_all_fp = np.concatenate([train_data_fp, train_data_heldout_fp], -1)
            valid_data = get_key(f'eval_{NLB_KEY}_heldin')
            train_data_heldout = get_key(f'train_{NLB_KEY}_heldout')
            if f'eval_{NLB_KEY}_heldout' in h5dict:
                valid_data_heldout = get_key(f'eval_{NLB_KEY}_heldout')
            else:
                self.logger.warn('Substituting zero array for heldout neurons. Only done for evaluating models locally, i.e. will disrupt training due to early stopping.')
                valid_data_heldout = np.zeros((valid_data.shape[0], valid_data.shape[1], train_data_heldout.shape[2]), dtype=np.float32)
            if f'eval_{NLB_KEY}_heldin_forward' in h5dict:
                valid_data_fp = get_key(f'eval_{NLB_KEY}_heldin_forward')
                valid_data_heldout_fp = get_key(f'eval_{NLB_KEY}_heldout_forward')
                valid_data_all_fp = np.concatenate([valid_data_fp, valid_data_heldout_fp], -1)
            else:
                self.logger.warn('Substituting zero array for heldout forward neurons. Only done for evaluating models locally, i.e. will disrupt training due to early stopping.')
                valid_data_all_fp = np.zeros(
                    (valid_data.shape[0], train_data_fp.shape[1], valid_data.shape[2] + valid_data_heldout.shape[2]), dtype=np.float32
                )

            # NLB data does not have ground truth rates
            if mode == DATASET_MODES.train:
                return train_data, None, train_data_heldout, train_data_all_fp
            elif mode == DATASET_MODES.val:
                return valid_data, None, valid_data_heldout, valid_data_all_fp
        train_data = h5dict['train_data'].astype(np.float32).squeeze()
        valid_data = h5dict['valid_data'].astype(np.float32).squeeze()
        train_rates = None
        valid_rates = None
        if "train_truth" and "valid_truth" in h5dict: # original LFADS-type datasets
            has_rates = True
            train_rates = h5dict['train_truth'].astype(np.float32)
            valid_rates = h5dict['valid_truth'].astype(np.float32)
            train_rates = train_rates / h5dict['conversion_factor']
            valid_rates = valid_rates / h5dict['conversion_factor']
            if config.data.use_lograte:
                train_rates = torch.log(torch.tensor(train_rates) + config.data.LOG_EPSILON)
                valid_rates = torch.log(torch.tensor(valid_rates) + config.data.LOG_EPSILON)

    if mode == DATASET_MODES.train:
        return train_data, train_rates, None, None
    elif mode == DATASET_MODES.val:
        return valid_data, valid_rates, None, None
    # elif mode == DATASET_MODES.trainval:
    #     # merge training and validation data
    #     if 'train_inds' in h5dict and 'valid_inds' in h5dict:
    #         # if there are index labels, use them to reassemble full data
    #         train_inds = h5dict['train_inds'].squeeze()
    #         valid_inds = h5dict['valid_inds'].squeeze()
    #         file_data = merge_train_valid(
    #             train_data, valid_data, train_inds, valid_inds)
    #         if has_rates:
    #             merged_rates = merge_train_valid(
    #                 train_rates, valid_rates, train_inds, valid_inds
    #             )
    #     else:
    #         if self.logger is not None:
    #             self.logger.info("No indices found for merge. "
    #             "Concatenating training and validation samples.")
    #         file_data = np.concatenate([train_data, valid_data], axis=0)
    #         if self.has_rates:
    #             merged_rates = np.concatenate([train_rates, valid_rates], axis=0)
    #     return file_data, merged_rates if self.has_rates else None, None, None
    else: # test unsupported
        return None, None, None, None
