from typing import Tuple, Optional, List
from math import ceil, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig

""" Module for patching spikes. 

CONFIG:
    max_space_F: space patch size (# neurons per patch)
    max_time_F:  time patch size (# timesteps per patch)
    time_stride: 
"""

# TO DO: ADD TIME STRIDE

class Patcher(nn.Module):
    def __init__(self, max_space_F, max_time_F, embed_region, config: DictConfig):
        super().__init__()
        self.max_space_F = max_space_F
        self.max_time_F = max_time_F
        self.pad_value = -1.
        self.time_stride = config.time_stride
        self.embed_region = embed_region

    def forward(
        self, 
        spikes:          torch.FloatTensor,     # (bs, seq_len, n_channels)
        pad_space_len:   int,      
        pad_time_len:    int,
        time_attn_mask:  torch.LongTensor,      # (bs, seq_len,)
        space_attn_mask: torch.LongTensor,       # (bs, seq_len,)
        region_indx:     Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:  
         # (bs, seq_len, n_channels), (bs, seq_len), (bs, seq_len), (bs, seq_len), (bs, seq_len, n_channels)

        B, T, N = spikes.size()

        self.n_time_patches = ceil(T/self.max_time_F)
        self.n_space_patches = ceil(N/self.max_space_F)
        self.n_channels = self.max_time_F * self.max_space_F

        self.pad_space_len = pad_space_len % self.max_space_F
        self.pad_space_idx = floor(pad_space_len/self.max_space_F)
        
        # Group spikes into patches
        patches = torch.ones((B, self.n_time_patches, self.n_space_patches, self.n_channels)) * self.pad_value 
        for t in range(self.n_time_patches):
            for s in range(self.n_space_patches):
                patches[:,t,s] = spikes[:,t*self.max_time_F:(t+1)*self.max_time_F, \
                s*self.max_space_F:(s+1)*self.max_space_F].flatten(1)
                if self.pad_space_len != 0:
                    if s == self.pad_space_idx:
                        patches[:,t,s] = torch.cat(
                            (patches[:,t,s][:,:self.pad_space_len], 
                             patches[:,t,s-1][:,-(self.max_space_F - self.pad_space_len):]
                            ), dim=1
                        )                       
        patches = patches.flatten(1,2).to(spikes.device)

        # Prepare space and time stamps after accounting for [cls] tokens
        # Majority vote the brain region for each patch
        if self.embed_region:
            regionstamps = torch.zeros(self.n_space_patches)
            for s in range(self.n_space_patches):
                if len(region_indx[0, s*self.max_space_F:(s+1)*self.max_space_F]) != 0:
                    regionstamps[s] = torch.mode(
                        region_indx[0, s*self.max_space_F:(s+1)*self.max_space_F]
                    ).values
            regionstamps = regionstamps.to(torch.int64)[None,None,:]
            regionstamps = regionstamps.expand(B, self.n_time_patches,-1).to(spikes.device).flatten(1)
        else:
            regionstamps = None
        
        spacestamps = torch.arange(self.n_space_patches).to(torch.int64)[None,None,:]
        spacestamps = spacestamps.expand(B, self.n_time_patches,-1).to(spikes.device).flatten(1)
        
        timestamps = torch.arange(self.n_time_patches).to(torch.int64)[None,:,None]
        timestamps = timestamps.expand(B, -1, self.n_space_patches).to(spikes.device).flatten(1)
        
        # Prepare space and time masks after accounting for [cls] tokens
        _time_attn_mask = time_attn_mask[:,:,None].expand(-1,-1,self.n_space_patches) 
        time_attn_mask = torch.zeros((B, self.n_time_patches, self.n_space_patches))
        for t in range(self.n_time_patches):
            if (_time_attn_mask[:,t*self.max_time_F:(t+1)*self.max_time_F]==1).sum() > 0:
                time_attn_mask[:,t,:] = 1

        _space_attn_mask = space_attn_mask[:,None,:].expand(-1,self.n_time_patches,-1) 
        space_attn_mask = torch.zeros((B, self.n_time_patches, self.n_space_patches))   
        for s in range(self.n_space_patches):
            if (_space_attn_mask[:,:,s*self.max_space_F:(s+1)*self.max_space_F]==1).sum() > 0:
                space_attn_mask[:,:,s] = 1

        time_attn_mask = time_attn_mask.to(torch.int64).to(spikes.device).flatten(1)
        space_attn_mask = space_attn_mask.to(torch.int64).to(spikes.device).flatten(1)

        return patches, space_attn_mask, time_attn_mask, spacestamps, timestamps, regionstamps

