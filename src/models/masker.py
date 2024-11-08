from typing import Tuple, Optional, List

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig

from torch import no_grad

""" Module for masking spikes. Operates in various modes
>``random``: neuron and timestep are both randomly selected
>``temporal``: all channels of randomly selected timesteps are masked. supports consecutive bin masking
>``neuron``: all timebins of randomly selected neurons are masked
>``co-smooth``: a fixed set of channels are masked
>``forward-pred``: a fixed set of time steps are masked
>``intra-region``: all neurons except a specific brain region are masked. Some neurons are masked in the target region. The targets are within the unmasked region
>``inter-region``: neurons in specific brain regions are masked. The targets are the masked regions.

CONFIG:
    mode: masking mode
    ratio: fraction of bins to predict in ``random``,``temporal``, ``neuron`` and ``intra-area`` modes
    zero_ratio: of the selected bins, fraction of zeroed out
    random_ratio: of the not zeroed out, fraction of randomly set. the rest are left unchanged
    expand_prob: probability of expanding the mask for consecutive bin masking in ``temporal`` mode
    max_timespan: max length of the expanded mask in ``temporal`` mode
    channels: list of ``int`` containing the indx of channels to mask in ``co-smooth`` mode
    timesteps: list of ``int`` containing the indx of time steps to mask in ``forward-pred`` mode
    mask_regions: list of ``str`` containing the names of regions to mask in ``inter-region`` mode
    target_regions: list of ``str`` containing the names of the target region in ``intra-region`` mode
    n_mask_regions: number of regions to select from mask_regions or target_regions
"""
class New_Masker(nn.Module):

    
    def __init__(self, config: DictConfig):
        super().__init__()

        self.force_active = config.force_active if "force_active" in config else False
        self.mode = config.mode          
        self.ratio = config.ratio
        self.zero_ratio = config.zero_ratio
        self.random_ratio = config.random_ratio
        self.expand_prob = config.expand_prob
        self.max_timespan = config.max_timespan
        self.channels = config.channels
        self.timesteps = config.timesteps
        self.mask_regions = config.mask_regions
        self.target_regions = config.target_regions
        self.n_mask_regions = config.n_mask_regions

    @no_grad
    def forward(
        self, 
        patched_spikes: torch.FloatTensor,                     
        max_time_F: torch.FloatTensor,
        embedding_dim: torch.FloatTensor,
        neuron_regions: np.ndarray = None,              # (bs, n_channels)     
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (bs, seq_len, n_channels), (bs, seq_len, n_channels)

        spikes_shape = (patched_spikes.shape[0], patched_spikes.shape[1], patched_spikes.shape[2])

        if not self.training and not self.force_active:
            return torch.zeros(spikes_shape)
        elif self.target_regions is None:
            return torch.zeros(spikes_shape)
        elif self.mask_regions is None:
            return torch.zeros(spikes_shape)
        elif self.ratio == 0:
            return torch.zeros(spikes_shape)


        if 'all' in self.mask_regions:
            self.mask_regions = list(np.unique(neuron_regions))

        if 'all' in self.target_regions:
            self.target_regions = list(np.unique(neuron_regions))

        mask_ratio = self.ratio
        if self.mode in ["temporal", "random_token", "causal"]:
            # Expand mask
            if torch.bernoulli(torch.tensor(self.expand_prob).float()):
                timespan = torch.randint(1, self.max_timespan+1, (1, )).item() 
            else:
                timespan = 1
            mask_ratio = mask_ratio/timespan
            mask_probs = torch.full(spikes_shape[:2], mask_ratio) # (bs, seq_len)
        elif self.mode == "neuron":
            mask_probs = torch.full((spikes_shape[0], spikes_shape[2]), mask_ratio)    # (bs, n_channels)
        elif self.mode == "random":
            mask_probs = torch.full(spikes_shape, mask_ratio)     # (bs, seq_len, n_channels)
        elif self.mode == "co-smooth":
            assert self.channels is not None, "No channels to mask"
            mask_probs = torch.zeros(spikes_shape[2])
            for c in self.channels:
                mask_probs[c] = 1
        elif self.mode == "forward-pred":
            assert self.timesteps is not None, "No time steps to mask"
            mask_probs = torch.zeros(spikes_shape[1])
            for t in self.timesteps:
                mask_probs[t] = 1
        elif self.mode == "inter-region":
            assert neuron_regions is not None, "Can't mask region without brain region information"
            #assert self.mask_regions is not None, "No regions to mask"
            mask_regions = random.sample(self.mask_regions, self.n_mask_regions)
            mask_probs = torch.zeros(spikes_shape[0],spikes_shape[2])
            for region in mask_regions:
                region_indx = torch.tensor(neuron_regions == region, device=patched_spikes.device)
                mask_probs[region_indx] = 1      
        elif self.mode == "intra-region":
            assert neuron_regions is not None, "Can't mask region without brain region information"
            #assert self.target_regions is not None, "No target regions"

            target_regions = random.sample(self.target_regions, self.n_mask_regions)
            mask_probs = torch.ones(spikes_shape[0],spikes_shape[2])
            targets_mask = torch.zeros(spikes_shape[0],spikes_shape[2])
            for region in target_regions:
                region_indx = torch.tensor(neuron_regions == region, device=patched_spikes.device)
                mask_probs[region_indx] = mask_ratio
                targets_mask[region_indx] = 1
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")
        
        # Create mask
        mask = torch.bernoulli(mask_probs).to(patched_spikes.device)

        # Expand mask
        if self.mode in ["temporal", "random_token", "causal"]:
            if timespan > 1:
                mask = self.expand_timesteps(mask, timespan)
            mask = mask.unsqueeze(2).expand(*spikes_shape).bool()    
        elif self.mode in ["neuron","region","intra-region","inter-region"]:
            mask = mask.unsqueeze(1).expand(*spikes_shape).bool()    
        elif self.mode in ["co-smooth"]:
            mask = mask.unsqueeze(0).unsqueeze(1).expand(*spikes_shape).bool()
        elif self.mode in ["forward-pred"]:
            mask = mask.unsqueeze(0).unsqueeze(2).expand(*spikes_shape).bool()
        else: # random
            mask = mask.bool()          # (bs, seq_len, n_channels)
            
        # Mask data
        zero_idx = torch.bernoulli(torch.full(patched_spikes.shape[:-1], self.zero_ratio)).to(patched_spikes.device).bool() & mask
        patched_spikes[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(patched_spikes.shape[:-1], self.random_ratio)).to(patched_spikes.device).bool() & mask & ~zero_idx
        random_spikes = (patched_spikes.max() * torch.rand(patched_spikes.shape, device=patched_spikes.device)).to(patched_spikes.dtype)
        patched_spikes[random_idx] = random_spikes[random_idx]

        targets_mask = mask if self.mode != "intra-region" else mask & targets_mask.unsqueeze(1).expand_as(mask).bool().to(patched_spikes.device)
        #TODO: double check that this method is right and mask == 1 is what should be replaced
        token_masks = targets_mask.clone().flatten(1,2).unsqueeze(-1).expand(-1,-1,embedding_dim).bool()    #(B, tokens, embed_dim)
        targets_mask = targets_mask.repeat_interleave(max_time_F, dim=1).bool()        #(B, time, neurons)
        return patched_spikes, token_masks, targets_mask    

    @staticmethod
    def expand_timesteps(mask, width=1):
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding="same")
        return (expanded_mask.squeeze(1) >= 1)







class Masker(nn.Module):

    
    def __init__(self, config: DictConfig):
        super().__init__()

        self.force_active = config.force_active if "force_active" in config else False
        self.mode = config.mode          
        self.ratio = config.ratio
        self.zero_ratio = config.zero_ratio
        self.random_ratio = config.random_ratio
        self.expand_prob = config.expand_prob
        self.max_timespan = config.max_timespan
        self.channels = config.channels
        self.timesteps = config.timesteps
        self.mask_regions = config.mask_regions
        self.target_regions = config.target_regions
        self.n_mask_regions = config.n_mask_regions

    @no_grad
    def forward(
        self, 
        spikes: torch.FloatTensor,                      # (bs, seq_len, n_channels)
        neuron_regions: np.ndarray = None,              # (bs, n_channels)     
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (bs, seq_len, n_channels), (bs, seq_len, n_channels)

        if not self.training and not self.force_active:
            return spikes, torch.zeros_like(spikes)
        elif self.target_regions is None:
            return spikes, torch.zeros_like(spikes)
        elif self.mask_regions is None:
            return spikes, torch.zeros_like(spikes)
        elif self.ratio == 0:
            return spikes, torch.zeros_like(spikes)


        if 'all' in self.mask_regions:
            self.mask_regions = list(np.unique(neuron_regions))

        if 'all' in self.target_regions:
            self.target_regions = list(np.unique(neuron_regions))

        mask_ratio = self.ratio
        if self.mode in ["temporal", "random_token", "causal"]:
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
        elif self.mode == "forward-pred":
            assert self.timesteps is not None, "No time steps to mask"
            mask_probs = torch.zeros(spikes.shape[1])
            for t in self.timesteps:
                mask_probs[t] = 1
        elif self.mode == "inter-region":
            assert neuron_regions is not None, "Can't mask region without brain region information"
            #assert self.mask_regions is not None, "No regions to mask"
            mask_regions = random.sample(self.mask_regions, self.n_mask_regions)
            mask_probs = torch.zeros(spikes.shape[0],spikes.shape[2])
            for region in mask_regions:
                region_indx = torch.tensor(neuron_regions == region, device=spikes.device)
                mask_probs[region_indx] = 1      
        elif self.mode == "intra-region":
            assert neuron_regions is not None, "Can't mask region without brain region information"
            #assert self.target_regions is not None, "No target regions"

            target_regions = random.sample(self.target_regions, self.n_mask_regions)
            mask_probs = torch.ones(spikes.shape[0],spikes.shape[2])
            targets_mask = torch.zeros(spikes.shape[0],spikes.shape[2])
            for region in target_regions:
                region_indx = torch.tensor(neuron_regions == region, device=spikes.device)
                mask_probs[region_indx] = mask_ratio
                targets_mask[region_indx] = 1
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")
        
        # Create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # Expand mask
        if self.mode in ["temporal", "random_token", "causal"]:
            if timespan > 1:
                mask = self.expand_timesteps(mask, timespan)
            mask = mask.unsqueeze(2).expand_as(spikes).bool()    
        elif self.mode in ["neuron","region","intra-region","inter-region"]:
            mask = mask.unsqueeze(1).expand_as(spikes).bool()    
        elif self.mode in ["co-smooth"]:
            mask = mask.unsqueeze(0).unsqueeze(1).expand_as(spikes).bool()
        elif self.mode in ["forward-pred"]:
            mask = mask.unsqueeze(0).unsqueeze(2).expand_as(spikes).bool()
        else: # random
            mask = mask.bool()          # (bs, seq_len, n_channels)
            
        # Mask data
        zero_idx = torch.bernoulli(torch.full(spikes.shape, self.zero_ratio)).to(spikes.device).bool() & mask
        spikes[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(spikes.shape, self.random_ratio)).to(spikes.device).bool() & mask & ~zero_idx
        random_spikes = (spikes.max() * torch.rand(spikes.shape, device=spikes.device)).to(spikes.dtype)
        spikes[random_idx] = random_spikes[random_idx]

        targets_mask = mask if self.mode != "intra-region" else mask & targets_mask.unsqueeze(1).expand_as(mask).bool().to(spikes.device)
        return spikes, targets_mask.to(torch.int64) 

    @staticmethod
    def expand_timesteps(mask, width=1):
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding="same")
        return (expanded_mask.squeeze(1) >= 1)


