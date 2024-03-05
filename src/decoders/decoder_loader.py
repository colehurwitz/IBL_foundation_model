import numpy as np
from pathlib import Path
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import datasets
from utils.dataset import get_binned_spikes_from_sparse

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
    return binned_spikes


class SingleSessionDataset(Dataset):
    def __init__(self, data_dir, eid, beh_name, device, split='train'):
        
        dataset = datasets.load_from_disk(Path(data_dir)/eid)
        self.train_spike = get_binned_spikes(dataset['train'])
        self.train_behavior = dataset['train'][beh_name]
        self.spike_data = get_binned_spikes(dataset[split])
        self.behavior = dataset[split][beh_name]
        
        self.n_t_steps = self.spike_data.shape[1]
        self.n_units = self.behavior.shape[2]

        # fit scaler on train to avoid data leakage
        self.train_spike, self.means, self.stds = standardize_spike_data(self.train_spike)
        self.spike_data, _, _ = standardize_spike_data(self.spike_data, self.means, self.stds)

        self.scaler = preprocessing.StandardScaler().fit(self.train_behavior)
        self.behavior = self.scaler.transform(self.behavior)

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
        self.data_dir = config.dirs.data_dir
        self.eid = config.eid
        self.beh_name = config.target
        self.device = config.training.device
        self.batch_size = config.training.batch_size
        self.n_workers = config.data.num_workers

    def setup(self):
        self.train = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.device, split='train'
        )
        self.val = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.device, split='val'
        )
        self.test = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.device, split='test'
        )
        self.config.update({'n_units': session_dataset.n_units, 'n_t_steps': session_dataset.n_t_steps})
        

    def train_dataloader(self):
        data_loader = DataLoader(
          self.train, batch_size=self.batch_size, shuffle=True, 
          # setting num_workers > 0 triggers errors so leave it as it is for now
          # num_workers=self.n_workers, pin_memory=True
        )
        return data_loader

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=True)

