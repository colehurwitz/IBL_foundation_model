from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig

""" Module for masking spikes. Operates in various modes
>``random``: neuron and timestep are both randomly selected
>``temporal``: all channels of randomly selected timesteps are masked. supports consecutive bin masking
>``neuron``: all timebins of randomly selected neurons are masked
>``co-smooth``: a fixed set of channels are masked
>``intra-area``: all neurons except a specific brain region are masked. Some neurons are masked in the target region. The targets are within the unmasked region
>``inter-area``: neurons in specific brain regions are masked. The targets are the masked regions.

CONFIG:
    mode: masking mode
    ratio: fraction of bins to predict in ``random``,``temporal``, ``neuron`` and ``intra-area`` modes
    zero_ratio: of the selected bins, fraction of zeroed out
    random_ratio: of the not zeroed out, fraction of randomly set. the rest are left unchanged
    expand_prob: probability of expanding the mask for consecutive bin masking in ``temporal`` mode
    max_timespan: max length of the expanded mask in ``temporal`` mode
    channels: list of ``int`` containing the indx of channels to mask in ``co-smooth`` mode
    mask_regions: list of ``str`` containing the names of regions to mask in ``inter-region`` mode
    target_regions: list of ``str`` containing the names of the targe region in ``intra-region`` mode
"""
class Masker(nn.Module):

    
    def __init__(self, config: DictConfig):
        super().__init__()

        self.mode = config.mode          
        self.ratio = config.ratio
        self.zero_ratio = config.zero_ratio
        self.random_ratio = config.random_ratio
        self.expand_prob = config.expand_prob
        self.max_timespan = config.max_timespan
        self.channels = config.channels
        self.mask_regions = config.mask_regions
        self.target_regions = config.target_regions

    def forward(
        self, 
        spikes: torch.FloatTensor,                      # (bs, seq_len, n_channels)
        regions: Optional[List[List]] = None,           # (bs, n_channels)     
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (bs, seq_len, n_channels), (bs, seq_len, n_channels)

        if regions is not None:
            regions = np.asarray(regions)

        mask_ratio = self.ratio
        if self.mode == "temporal":
            # Expand mask
            if torch.bernoulli(torch.tensor(self.expand_prob).float()):
                timespan = torch.randint(1, self.max_timespan+1, (1, )).item() 
            else:
                timespan = 1
            mask_ratio = mask_ratio/timespan
            mask_probs = torch.full(spikes[:, :, 0].shape, mask_ratio) # (bs, seq_len)
        elif self.mode == "neuron":
            mask_probs = torch.full(spikes[:, 0].shape, mask_ratio)    # (bs, n_channels)
        elif self.mode == "random":
            mask_probs = torch.full(spikes.shape, mask_ratio)     # (bs, seq_len, n_channels)
        elif self.mode == "co-smooth":
            assert self.channels is not None, "No channels to mask"
            mask_probs = torch.zeros(spikes.shape[2])
            for c in self.channels:
                mask_probs[c] = 1
        elif self.mode == "inter-region":
            assert regions is not None, "Can't mask region without brain region information"
            assert self.mask_regions is not None, "No regions to mask"

            mask_probs = torch.zeros(spikes.shape[0],spikes.shape[2])
            for region in self.mask_regions:
                region_indx = torch.tensor(regions == region, device=spikes.device)
                mask_probs[region_indx] = 1      
        elif self.mode == "intra-region":
            assert regions is not None, "Can't mask region without brain region information"
            assert self.target_regions is not None, "No target regions"

            mask_probs = torch.ones(spikes.shape[0],spikes.shape[2])
            targets_mask = torch.zeros(spikes.shape[0],spikes.shape[2])
            for region in self.target_regions:
                region_indx = torch.tensor(regions == region, device=spikes.device)
                mask_probs[region_indx] = mask_ratio
                targets_mask[region_indx] = 1
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")
        
        # Create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # Expand mask
        if self.mode == "temporal":
            mask = self.expand_timesteps(mask, timespan)
            mask = mask.unsqueeze(2).expand_as(spikes).bool()    
        elif self.mode in ["neuron","region","intra-region","inter-region"]:
            mask = mask.unsqueeze(1).expand_as(spikes).bool()    
        elif self.mode in ["co-smooth"]:
            mask = mask.unsqueeze(0).unsqueeze(1).expand_as(spikes).bool()
        else: # random
            mask = mask.bool()          # (bs, seq_len, n_channels)
            
        # Mask data
        zero_idx = torch.bernoulli(torch.full(spikes.shape, self.zero_ratio)).to(spikes.device).bool() & mask
        spikes[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(spikes.shape, self.random_ratio)).to(spikes.device).bool() & mask & ~zero_idx
        random_spikes = (spikes.max() * torch.rand(spikes.shape, device=spikes.device)).to(spikes.dtype)
        spikes[random_idx] = random_spikes[random_idx]

        targets_mask = mask if self.mode != "intra-region" else mask & targets_mask.unsqueeze(1).expand_as(mask).bool()
        return spikes, targets_mask.to(torch.int64) 

    @staticmethod
    def expand_timesteps(mask, width=1):
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding="same")
        return (expanded_mask.squeeze(1) >= 1)
        