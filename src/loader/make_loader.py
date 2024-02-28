import torch
from loader.base import BaseDataset

def make_loader(dataset, 
                 batch_size, 
                 pad_to_right = True,
                 pad_value = 0.,
                 max_length = 5000,
                 shuffle = True):
    dataset = BaseDataset(dataset=dataset, 
                          pad_value=pad_value,
                          max_length=max_length,
                          pad_to_right=pad_to_right,
                            )
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle)
    return dataloader