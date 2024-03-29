from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig


""" Module for masking spikes. Operates in various modes
>``temporal``: all channels of randomly selected timesteps are masked. supports consecutive bin masking
>``neuron``: all timebins of randomly selected neurons are masked
>``random``: neuron and timestep are both randomly selected
>``region``: all neurons in a given brain region are masked
>``co-smooth``: a fixed set of channels are masked

CONFIG:
    mode: masking mode
    ratio: fraction of bins to mask in ``random``,``temporal`` and ``neuron`` modes
    zero_ratio: of the masked bins, fraction of zeroed out
    random_ratio: of the not zeroed out, fraction of randomly set. the rest are left unchanged
    expand_prob: probability of expanding the mask for consecutive bin masking in ``temporal`` mode
    max_timespan: max length of the expanded mask in ``temporal`` mode
    regions: list of ``str`` containing the names of regions to mask in ``region`` mode
    channels: list of ``int`` containing the indx of channels to mask in ``co-smooth`` mode
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
        self.regions = config.regions
        self.channels = config.channels

    def forward(
        self, 
        spikes: torch.FloatTensor,                      # (bs, seq_len, n_channels)
        brain_regions: Optional[np.ndarray] = None,     # (bs, n_channels)     
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (bs, seq_len, n_channels), (bs, seq_len, n_channels)

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
        elif self.mode == "region":
            assert brain_regions is not None, "Can't mask region without brain region information"
            assert self.regions is not None, "No regions to mask"
            region_indx = []
            mask_probs = torch.zeros(spikes.shape[0],spikes.shape[2])
            for region in self.regions:
                region_indx = torch.tensor(brain_regions == region, device=spikes.device)
                mask_probs[region_indx] = 1
        elif self.mode == "co-smooth":
            assert self.channels is not None, "No channels to mask"
            mask_probs = torch.zeros(spikes.shape[2])
            for c in self.channels:
                mask_probs[c] = 1
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")
        
        # Create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # Expand mask
        if self.mode == "temporal":
            mask = self.expand_timesteps(mask, timespan)
            mask = mask.unsqueeze(2).expand_as(spikes).bool()    # (bs, seq_len, n_channels)
        elif self.mode in ["neuron","region"]:
            mask = mask.unsqueeze(1).expand_as(spikes).bool()    # (bs, seq_len, n_channels)
        elif self.mode == "co-smooth":
            mask = mask.unsqueeze(0).unsqueeze(1).expand_as(spikes).bool()
        else: # random
            mask = mask.bool()
            
        # Mask data
        zero_idx = torch.bernoulli(torch.full(spikes.shape, self.zero_ratio)).to(spikes.device).bool() & mask
        spikes[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(spikes.shape, self.random_ratio)).to(spikes.device).bool() & mask & ~zero_idx
        random_spikes = (spikes.max() * torch.rand(spikes.shape, device=spikes.device)).to(spikes.dtype)
        spikes[random_idx] = random_spikes[random_idx]

        return spikes, mask.to(torch.int64)

    @staticmethod
    def expand_timesteps(mask, width=1):
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding="same")
        return (expanded_mask.squeeze(1) >= 1)


