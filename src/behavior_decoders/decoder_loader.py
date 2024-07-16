import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch import LightningDataModule
import datasets
from utils.dataset_utils import get_binned_spikes_from_sparse

seed = 42

def to_tensor(x, device):
    return torch.tensor(x).to(device)

def standardize_spike_data(spike_data, means=None, stds=None):
    K, T, N = spike_data.shape

    if (means is None) and (stds == None):
        means, stds = np.empty((T, N)), np.empty((T, N))

    std_spike_data = spike_data.reshape((K, -1))
    std_spike_data[np.isnan(std_spike_data)] = 0
    for t in range(T):
        mean = np.mean(std_spike_data[:, t*N:(t+1)*N])
        std = np.std(std_spike_data[:, t*N:(t+1)*N])
        std_spike_data[:, t*N:(t+1)*N] -= mean
        if std != 0:
            std_spike_data[:, t*N:(t+1)*N] /= std
        means[t], stds[t] = mean, std
    std_spike_data = std_spike_data.reshape(K, T, N)
    return std_spike_data, means, stds


def get_binned_spikes(dataset):
    spikes_sparse_data_list = dataset['spikes_sparse_data']
    spikes_sparse_indices_list = dataset['spikes_sparse_indices']
    spikes_sparse_indptr_list = dataset['spikes_sparse_indptr']
    spikes_sparse_shape_list = dataset['spikes_sparse_shape']
    
    binned_spikes  = get_binned_spikes_from_sparse(
        spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list
    )
    return binned_spikes.astype(float)


class SingleSessionDataset(Dataset):
    def __init__(self, data_dir, eid, beh_name, target, device, split='train'):
        # dataset = datasets.load_from_disk(Path(data_dir)/eid)
        dataset = datasets.load_dataset(f'ibl-foundation-model/{eid}_aligned', cache_dir=data_dir)
        self.train_spike = get_binned_spikes(dataset['train'])
        self.train_behavior = np.array(dataset['train'][beh_name])
        if split == 'val':
            try:
                self.spike_data = get_binned_spikes(dataset[split])
            except:
                tmp_dataset = dataset['train'].train_test_split(test_size=0.1, seed=seed)
                self.train_spike = get_binned_spikes(tmp_dataset['train'])
                self.spike_data = get_binned_spikes(tmp_dataset['test'])
        else:
            self.spike_data = get_binned_spikes(dataset[split])
        self.behavior = np.array(dataset[split][beh_name])

        self.n_t_steps = self.spike_data.shape[1]
        self.n_units = self.spike_data.shape[2]

        # fit scaler on train to avoid data leakage
        self.train_spike, self.means, self.stds = standardize_spike_data(self.train_spike)
        self.spike_data, _, _ = standardize_spike_data(self.spike_data, self.means, self.stds)

        if target == 'clf':
            enc = OneHotEncoder(handle_unknown='ignore')
            self.behavior = enc.fit_transform(self.behavior.reshape(-1, 1)).toarray()
        elif self.behavior.shape[1] == self.n_t_steps:
            self.scaler = preprocessing.StandardScaler().fit(self.train_behavior)
            self.behavior = self.scaler.transform(self.behavior) 

        if np.isnan(self.behavior).sum() != 0:
            self.behavior[np.isnan(self.behavior)] = np.nanmean(self.behavior)
            print(f'Session {eid} contains NaNs in {beh_name}.')

        # map to device
        self.spike_data = to_tensor(self.spike_data, device).double()
        self.behavior = to_tensor(self.behavior, device).double()
  
    def __len__(self):
        return len(self.spike_data)

    def __getitem__(self, trial_idx):
        return self.spike_data[trial_idx], self.behavior[trial_idx]

    
class SingleSessionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['dirs']['data_dir']
        self.eid = config['eid']
        self.beh_name = config['target']
        self.target = config['model']['target']
        self.device = config['training']['device']
        self.batch_size = config['training']['batch_size']
        self.n_workers = config['data']['num_workers']

    def setup(self, stage=None):
        self.train = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, self.device, split='train'
        )
        self.val = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, self.device, split='val'
        )
        self.test = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, self.device, split='test'
        )
        self.config.update({'n_units': self.train.n_units, 'n_t_steps': self.train.n_t_steps})
        

    def train_dataloader(self):
        data_loader = DataLoader(
          self.train, batch_size=self.batch_size, shuffle=True, 
          # Setting num_workers > 0 triggers errors so leave it as it is for now
          # num_workers=self.n_workers, pin_memory=True
        )
        return data_loader

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=True)


class MultiSessionDataModule(LightningDataModule):
    def __init__(self, eids, configs):
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['training']['batch_size']

    def setup(self, stage=None):
        self.train, self.val, self.test = [], [], []
        for idx, eid in enumerate(self.eids):
            dm = SingleSessionDataModule(self.configs[idx])
            dm.setup()
            self.train.append(
                DataLoader(dm.train, batch_size = self.batch_size, shuffle=True)
            )
            self.val.append(
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=True)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=True)
            )

    def train_dataloader(self):
        data_loader = CombinedLoader(self.train, mode = "max_size_cycle")
        return data_loader

    def val_dataloader(self):
        data_loader = CombinedLoader(self.val)
        return data_loader

    def test_dataloader(self):
        data_loader = CombinedLoader(self.test)
        return data_loader
    