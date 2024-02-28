import torch
from loader.base import BaseDataset

def make_batcher(dataset, 
                 batch_size, 
                 pad_to_ritght = True,
                 max_pad = 5000,
                 shuffle = True):
    dataset = BaseDataset(dataset=dataset, 
                          max_pad=max_pad,
                          pad_to_ritght=pad_to_ritght,
                            )
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle)
    return dataloader