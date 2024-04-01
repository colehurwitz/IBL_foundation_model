import torch
from loader.base import BaseDataset

def make_loader(dataset, 
                 batch_size, 
                 target = None,
                 pad_to_right = True,
                 patching = False,
                 sort_by_depth = False,
                 pad_value = 0.,
                 max_time_length = 5000,
                 max_space_length = 100,
                 n_neurons_per_patch = 64,
                 bin_size = 0.05,
                 dataset_name = "ibl",
                 shuffle = True):
    dataset = BaseDataset(dataset=dataset, 
                          target=target,
                          pad_value=pad_value,
                          max_time_length=max_time_length,
                          max_space_length=max_space_length,
                          n_neurons_per_patch=n_neurons_per_patch,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                          dataset_name=dataset_name,
                          patching = patching,
                          sort_by_depth = sort_by_depth,
                            )
    print(f"len(dataset): {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle)
    return dataloader
