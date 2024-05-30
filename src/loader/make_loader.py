import torch
from loader.base import BaseDataset, LengthStitchGroupedSampler, LengthGroupedSampler

def make_loader(dataset, 
                 batch_size, 
                 target = None,
                 pad_to_right = True,
                 sort_by_depth = False,
                 sort_by_region = False,
                 pad_value = 0.,
                 max_time_length = 5000,
                 max_space_length = 100,
                 bin_size = 0.05,
                 brain_region = 'all',
                 load_meta=False,
                 use_nemo=False,
                 dataset_name = "ibl",
                 stitching = False,
                 shuffle = True):
    
    dataset = BaseDataset(dataset=dataset, 
                          target=target,
                          pad_value=pad_value,
                          max_time_length=max_time_length,
                          max_space_length=max_space_length,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                          dataset_name=dataset_name,
                          sort_by_depth = sort_by_depth,
                          sort_by_region = sort_by_region,
                          brain_region = brain_region,
                          load_meta=load_meta,
                          use_nemo=use_nemo,
                          stitching=stitching
            )
    
    print(f"len(dataset): {len(dataset)}")

    if stitching:
        train_sampler = LengthStitchGroupedSampler(
            dataset=dataset, batch_size=batch_size, lengths=[sum(x["space_attn_mask"]) for x in dataset]
        )
        # batch by neuron lengths makes multi-session training easier and NDT2 training faster
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=train_sampler, 
            batch_size=batch_size,
        )
    else:
        # the original data loader - each batch can contain different neuron lengths
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
