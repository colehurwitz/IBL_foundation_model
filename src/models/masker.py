from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import DictConfig


""" Module for masking spikes. Operates in various modes
>``temporal``: all channels of randomly selected timesteps are masked. supports consecutive bin masking
>``neuron``: all timebins of randomly selected neurons are masked
>``random``: neuron and timestep are both randomly selected
>``region``: all neurons in a given brain region are masked

CONFIG:
    mode: masking mode
    ratio: fraction of bins to mask in ``random``,``temporal`` and ``neuron`` modes
    zero_ratio: of the masked bins, fraction of zeroed out
    random_ratio: of the not zeroed out, fraction of randomly set. the rest are left unchanged
    expand_prob: probability of expanding the mask for consecutive bin masking in ``temporal`` mode
    max_timespan: max length of the expanded mask in ``temporal`` mode
    mask_regions: list of ``str`` containing the names of regions to mask in ``region`` mode
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
        self.mask_regions = config.mask_indx

    def forward(
        self, 
        spikes: torch.FloatTensor,                      # (bs, seq_len, n_channels)
        regions: List[List[str]],                       # (bs, n_channels)     
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (bs, seq_len, n_channels), (bs, seq_len, n_channels)

        mask_ratio = self.ratio
        # Expand mask
        if self.mode == "timestep":
            if torch.bernoulli(torch.tensor(self.expand_prob).float()):
                timespan = torch.randint(1, self.max_timespan+1, (1, )).item() 
            else:
                timespan = 1
            mask_ratio = mask_ratio/timespan

        # Get masking probabilities
        if self.mode == "full":
            mask_probs = torch.full(spikes.shape, mask_ratio)     # (bs, seq_len, n_channels)
        elif self.mode == "timestep":
            mask_probs = torch.full(spikes[:, :, 0].shape, mask_ratio) # (bs, seq_len)
        elif self.mode == "neuron":
            mask_probs = torch.full(spikes[:, 0].shape, mask_ratio)    # (bs, n_channels)
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")
        
        # Create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # Expand mask
        if self.mode == "timestep":
            mask = self.expand_timesteps(mask, timespan)
            mask = mask.unsqueeze(2).expand_as(spikes).bool()    # (bs, seq_len, n_channels)
        elif self.mode == "neuron":
            mask = mask.unsqueeze(1).expand_as(spikes).bool()    # (bs, seq_len, n_channels)
        else: # full
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