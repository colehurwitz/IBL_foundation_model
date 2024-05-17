import numpy as np
from scipy.sparse import csr_array
from datasets import Dataset, DatasetInfo, list_datasets, load_dataset, concatenate_datasets,DatasetDict, load_from_disk
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
TARGET_EIDS="data/target_eids.txt"
TEST_RE_EIDS="data/test_re_eids.txt"

def get_target_eids():
    with open(TARGET_EIDS) as file:
        include_eids = [line.rstrip() for line in file]
    return include_eids
def get_test_re_eids():
    with open(TEST_RE_EIDS) as file:
        include_eids = [line.rstrip() for line in file]
    return include_eids

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
        data_dict.update(binned_behaviors)
        
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
            'cluster_qc': [meta_data['cluster_qc']] * len(sparse_binned_spikes),
        }
        data_dict.update(meta_dict)

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
                     aligned_data_dir=None,
                     train_aligned=True,
                     eid=None, # specify 1 session for training, random_split will be used
                     num_sessions=5, # total number of sessions for training and testing
                     split_method="session_based",
                     train_session_eid=[],
                     test_session_eid=[], # specify session eids for testing, session_based will be used
                     split_size = 0.1,
                     mode = "train",
                     batch_size=16,
                     use_re=False,
                     seed=42):
    if aligned_data_dir:
        dataset = load_from_disk(aligned_data_dir)
        # if dataset does not have a 'train' key, it is a single session dataset
        if "train" not in dataset:
            _dataset = dataset.train_test_split(test_size=0.2, seed=seed)
            _dataset_train, _dataset_test = _dataset["train"], _dataset["test"]
            dataset = _dataset_train.train_test_split(test_size=0.1, seed=seed)
            return dataset["train"], dataset["test"], _dataset_test
        return dataset["train"], dataset["val"], dataset["test"]
    
    user_datasets = get_user_datasets(user_or_org_name)
    print("Total session-wise datasets found: ", len(user_datasets))
    cache_dir = os.path.join(cache_dir, "ibl", user_or_org_name)
    test_session_eid_dir = []
    train_session_eid_dir = []
    if eid is not None:
        eid_dir = os.path.join(user_or_org_name, eid+"_aligned")
        if eid_dir not in user_datasets:
            raise ValueError(f"Dataset with eid: {eid} not found in the user's datasets")
        else:
            train_session_eid_dir = [eid_dir]
            user_datasets = [eid_dir]

    if len(test_session_eid) > 0:
        test_session_eid_dir = [os.path.join(user_or_org_name, eid) for eid in test_session_eid]
        print("Test session-wise datasets found: ", len(test_session_eid_dir))
        train_session_eid_dir = [eid for eid in user_datasets if eid not in test_session_eid_dir]
        print("Train session-wise datasets found: ", len(train_session_eid_dir))
        if train_aligned:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" in eid]
        else:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" not in eid]
        train_session_eid_dir = train_session_eid_dir[:num_sessions - len(test_session_eid)]
        print("Number of training sesssion datasets to be used: ", len(train_session_eid_dir))
    else:
        if len(train_session_eid) > 0:
            train_session_eid_dir = [os.path.join(user_or_org_name, eid+'_aligned') for eid in train_session_eid]
        else:
            train_session_eid_dir = user_datasets
        if train_aligned:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" in eid]
        else:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" not in eid]
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
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif split_method == 'predefined':
        print("Loading train dataset sessions for predefined train/val/test split...")
        session_train_datasets = []
        session_val_datasets = []
        session_test_datasets = []

        num_neuron_set = set()
        eids_set = set()
        target_eids = get_target_eids()
        test_re_eids = get_test_re_eids()
        if use_re:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if eid.split('_')[0].split('/')[1] in target_eids]
            # remove the test_re_eids from the train_session_eid_dir
            train_session_eid_dir = [eid for eid in train_session_eid_dir if eid.split('_')[0].split('/')[1] not in test_re_eids]
        for dataset_eid in tqdm(train_session_eid_dir[:num_sessions]):
            try:
                # print("Loading dataset: ", dataset_eid)
                session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)
                train_trials = len(session_dataset["train"]["spikes_sparse_data"])
                train_trials = train_trials - train_trials % batch_size
                session_train_datasets.append(session_dataset["train"].select(list(range(train_trials))))

                val_trials = len(session_dataset["val"]["spikes_sparse_data"])
                val_trials = val_trials - val_trials % batch_size
                session_val_datasets.append(session_dataset["val"].select(list(range(val_trials))))

                test_trials = len(session_dataset["test"]["spikes_sparse_data"])
                test_trials = test_trials - test_trials % batch_size
                session_test_datasets.append(session_dataset["test"].select(list(range(test_trials))))
                binned_spikes_data = get_binned_spikes_from_sparse([session_dataset["train"]["spikes_sparse_data"][0]], 
                                                                    [session_dataset["train"]["spikes_sparse_indices"][0]],
                                                                    [session_dataset["train"]["spikes_sparse_indptr"][0]],
                                                                    [session_dataset["train"]["spikes_sparse_shape"][0]])

                num_neuron_set.add(binned_spikes_data.shape[2])
                eid_prefix = dataset_eid.split('_')[0] if train_aligned else dataset_eid
                eid_prefix = eid_prefix.split('/')[1]
                eids_set.add(eid_prefix)
            except Exception as e:
                print("Error loading dataset: ", dataset_eid)
                print(e)
                continue
        print("session eid used: ", eids_set)
        print("Total number of session: ", len(eids_set))
        train_dataset = concatenate_datasets(session_train_datasets)
        val_dataset = concatenate_datasets(session_val_datasets)
        test_dataset = concatenate_datasets(session_test_datasets)
        print("Train dataset size: ", len(train_dataset))
        print("Val dataset size: ", len(val_dataset))
        print("Test dataset size: ", len(test_dataset))
        num_neuron_set = list(num_neuron_set)
        meta_data = {
            "num_neurons": num_neuron_set,
            "num_sessions": len(eids_set),
            "eids": eids_set
        }
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
        
        train_dataset = train_dataset
        test_dataset = test_dataset
    else:
        raise ValueError("Invalid split method. Please choose either 'random_split' or 'session_based'")
    
    return train_dataset, val_dataset, test_dataset, meta_data

def _time_extract(data):
    data['time'] = data['intervals'][0]
    return data

# split the aligned and unaligned dataset together.
def split_both_dataset(
        aligned_dataset,
        unaligned_dataset,
        train_size=0.9,
        test_size=0.1,
        shuffle=True,
        seed=42
):
    assert train_size + test_size == 1, "The sum of train/test is not equal to 1."

    aligned_dataset = aligned_dataset.map(_time_extract)
    unaligned_dataset = unaligned_dataset.map(_time_extract)

    # split the aligned data first
    _tmp1 = aligned_dataset.train_test_split(train_size=train_size, test_size=test_size, shuffle=shuffle, seed=seed)
    test_alg = _tmp1['test']
    train_alg = _tmp1['train']


    new_aligned_dataset = DatasetDict({
        'train': train_alg,
        'test': test_alg
    })

    # split the unaligned data according to the aligned data
    times_test = test_alg['time']

    train_idxs = []
    test_idxs = []

    for i, data_ual in enumerate(unaligned_dataset):
        time_ual = data_ual['time']

        if any(abs(time_ual - time_test) <= 2 for time_test in times_test):
            test_idxs.append(i)

        else:
            train_idxs.append(i)

    train_ual = unaligned_dataset.select(train_idxs)
    test_ual = unaligned_dataset.select(test_idxs)

    new_unaligned_dataset = DatasetDict({
        'train': train_ual,
        'test': test_ual
    })

    return new_aligned_dataset, new_unaligned_dataset
            
