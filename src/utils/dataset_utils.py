import numpy as np
from scipy.sparse import csr_array
from datasets import Dataset, DatasetInfo, list_datasets, load_dataset, concatenate_datasets
import h5py
import os
import torch
from tqdm import tqdm

class DATASET_MODES:
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"

DATA_COLUMNS = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape','cluster_depths']

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

def create_dataset(binned_spikes, bwm_df, eid, params, meta_data=None, binned_behaviors=None):

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
            'good_clusters': [meta_data['good_clusters']] * len(sparse_binned_spikes)
        }
        data_dict = data_dict | meta_dict

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
    else: # test unsupported
        return None, None, None, None

# This function will fetch all dataset repositories for a given user or organization
def get_user_datasets(user_or_org_name):
    all_datasets = list_datasets()
    user_datasets = [d for d in all_datasets if d.startswith(f"{user_or_org_name}/")]
    return user_datasets

def load_ibl_dataset(cache_dir,
                     user_or_org_name='neurofm123',
                     eid=None, # specify 1 session for training, random_split will be used
                     num_sessions=5, # total number of sessions for training and testing
                     split_method="session_based",
                     test_session_eid=[], # specify session eids for testing, session_based will be used
                     split_size = 0.1,
                     mode = "train",
                     seed=42):
    user_datasets = get_user_datasets(user_or_org_name)
    print("Total session-wise datasets found: ", len(user_datasets))
    cache_dir = os.path.join(cache_dir, "ibl", user_or_org_name)
    if eid is not None:
        eid_dir = os.path.join(user_or_org_name, eid)
        if eid_dir not in user_datasets:
            raise ValueError(f"Dataset with eid: {eid} not found in the user's datasets")
        else:
            user_datasets = [eid_dir]

    test_session_eid_dir = []
    train_session_eid_dir = []
    if len(test_session_eid) > 0:
        test_session_eid_dir = [os.path.join(user_or_org_name, eid) for eid in test_session_eid]
        print("Test session-wise datasets found: ", len(test_session_eid_dir))
        train_session_eid_dir = [eid for eid in user_datasets if eid not in test_session_eid_dir]
        print("Train session-wise datasets found: ", len(train_session_eid_dir))
        train_session_eid_dir = train_session_eid_dir[:num_sessions - len(test_session_eid)]
        print("Number of training datasets to be used: ", len(train_session_eid_dir))
    else:
        train_session_eid_dir = user_datasets
    assert len(train_session_eid_dir) > 0, "No training datasets found"
    assert not (len(test_session_eid) > 0 and split_method == "random_split"), "When you have a test session, the split method should be 'session_based'"

    all_sessions_datasets = []
    if mode == "eval":
        print("eval mode: only loading test datasets...")
        for dataset_eid in tqdm(test_session_eid_dir):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        all_sessions_datasets = concatenate_datasets(all_sessions_datasets)
        test_dataset = all_sessions_datasets.select_columns(DATA_COLUMNS)
        return None, test_dataset
    
    if split_method == 'random_split':
        print("Loading datasets...")
        for dataset_eid in tqdm(train_session_eid_dir[:num_sessions]):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        all_sessions_datasets = concatenate_datasets(all_sessions_datasets)
        # split the dataset to train and test
        dataset = all_sessions_datasets.train_test_split(test_size=split_size, seed=seed)
        train_dataset = dataset["train"].select_columns(DATA_COLUMNS)
        test_dataset = dataset["test"].select_columns(DATA_COLUMNS)
    elif split_method == 'session_based':
        print("Loading train dataset sessions...")
        for dataset_eid in tqdm(train_session_eid_dir):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        train_dataset = concatenate_datasets(all_sessions_datasets)

        print("Loading test dataset session...")
        all_sessions_datasets = []
        for dataset_eid in tqdm(test_session_eid_dir):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        test_dataset = concatenate_datasets(all_sessions_datasets)
        
        train_dataset = train_dataset.select_columns(DATA_COLUMNS)
        test_dataset = test_dataset.select_columns(DATA_COLUMNS)
    else:
        raise ValueError("Invalid split method. Please choose either 'random_split' or 'session_based'")
    
    return train_dataset, test_dataset
            