import torch
from loader.base_copy import BaseDataset, NDT2Dataset

def make_loader(dataset, 
                 batch_size, 
                 pad_to_right = True,
                 pad_value = 0.,
                 max_length = 5000,
                 bin_size = 0.05,
                 shuffle = True):
    dataset = BaseDataset(dataset=dataset, 
                          pad_value=pad_value,
                          max_length=max_length,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                            )
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle)
    return dataloader


def make_ndt2_loader(dataset, 
                 batch_size, 
                 pad_to_right = True,
                 pad_value = 0.,
                 max_time_length=100, 
                 max_space_length=15, 
                 n_neurons_per_patch=64,
                 bin_size = 0.05,
                 shuffle = True):
    dataset = NDT2Dataset(
          dataset=dataset, 
          pad_value=pad_value,
          max_time_length=max_time_length, 
          max_space_length=max_space_length, 
          n_neurons_per_patch=n_neurons_per_patch,
          bin_size=bin_size,
          pad_to_right=pad_to_right,
)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

