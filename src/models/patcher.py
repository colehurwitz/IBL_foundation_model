from math import floor
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig

""" Module for patching spikes. Operates in various modes
>``space``: only neurons are patched, e.g., NDT2
>``full``: both neurons and time bins are patched, e.g., ST-Patch

CONFIG:
    mode: patching mode
    space_patch_size:
    time_patch_size:
    time_stride:
"""
# TO DO: Add time strides?

class Patcher(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.mode = config.mode          
        self.space_patch_size = config.space_patch_size
        self.time_patch_size = config.time_patch_size
        self.time_stride = config.time_stride

    def forward(
        self, 
        spikes: torch.FloatTensor,                       # (bs, seq_len, n_channels)
        _target_masks: torch.LongTensor,                  # (bs, seq_len, n_channels)
        timestamps: Optional[torch.LongTensor] = None,   # (bs, seq_len)
        spacestamps: Optional[torch.LongTensor] = None,  # (bs, seq_len)
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:  
         # (bs, seq_len, n_channels), (bs, seq_len, n_channels), (bs, seq_len, n_channels), (bs, seq_len), (bs, seq_len)

        B, T, N = spikes.size()

        pad_time_len = ((spikes[0,:,0] == 0).nonzero()).min().item()
        pad_space_len = ((spikes[0,0,:] == 0).nonzero()).min().item()
        _time_attn_mask = self._attention_mask(T, pad_time_len)[None,:,None,None]
        _space_attn_mask = self._attention_mask(N, pad_space_len)[None,None,:,None]
        
        if self.mode == 'space':
            # patch spikes
            self.n_patches = floor(N/self.space_patch_size)
            spikes_patches = torch.zeros((B, T, self.n_patches, self.space_patch_size))  
            for idx in range(self.n_patches):
                spikes_patches[:,:,idx] = spikes[:,:,idx*self.space_patch_size:(idx+1)*self.space_patch_size]

            # patch target masks
            target_masks = torch.zeros((B, T, self.n_patches, self.space_patch_size))
            for idx in range(self.n_patches):
                target_masks[:,:,idx] = _target_masks[:,:,idx*self.space_patch_size:(idx+1)*self.space_patch_size]

            # create space and time attention masks
            time_attn_mask = _time_attn_mask.expand(B,-1,self.n_patches,self.space_patch_size)
            _space_attn_mask = _space_attn_mask.expand(B,T,-1,self.space_patch_size)
            space_attn_mask = torch.ones((B, T, self.n_patches, self.space_patch_size))       
            for idx in range(self.n_patches):
                if _space_attn_mask[:,:,idx*self.space_patch_size:(idx+1)*self.space_patch_size].sum() == 0:
                    space_attn_mask[:,:,idx] = 0

            # create space and time positions
            if timestamps is None:
                timestamps = torch.arange(T)[None,:,None]
            timestamps = timestamps.expand(B,-1,self.n_patches)
            
            if spacestamps is None:
                spacestamps = torch.arange(self.n_patches)[None,None,:]
            spacestamps = spacestamps.expand(B,T,-1)
            
        elif self.mode == 'full':
            # patch spikes
            self.n_time_patches = floor(T/self.time_patch_size)
            self.n_space_patches = floor(N/self.space_patch_size)
            self.n_patches = self.n_time_patches * self.n_space_patches
            self.patch_size = self.time_patch_size * self.space_patch_size
            spikes_patches = torch.zeros((B, self.n_time_patches, self.n_space_patches, self.patch_size))
            for t_idx in range(self.n_time_patches):
                for s_idx in range(self.n_space_patches):
                    spikes_patches[:,t_idx,s_idx] = spikes[:,t_idx*self.time_patch_size:(t_idx+1)*self.time_patch_size,s_idx*self.space_patch_size:(s_idx+1)*self.space_patch_size]

            # patch target masks
            target_masks = torch.zeros((B, self.n_time_patches, self.n_space_patches, self.patch_size))
            for t_idx in range(self.n_time_patches):
                for s_idx in range(self.n_space_patches):
                    target_masks[:,t_idx,s_idx] = _target_masks[:,t_idx*self.time_patch_size:(t_idx+1)*self.time_patch_size,s_idx*self.space_patch_size:(s_idx+1)*self.space_patch_size]

            # create space and time attention masks
            _time_attn_mask = _time_attn_mask.expand(B,-1,self.n_space_patches,self.patch_size)
            time_attn_mask = torch.ones((B, self.n_time_patches, self.n_space_patches, self.patch_size))
            for t_idx in range(self.n_time_patches):
                if _time_attn_mask[:,t_idx*self.time_patch_size:(t_idx+1)*self.time_patch_size].sum() == 0:
                    time_attn_mask[:,t_idx] = 0
            
            _space_attn_mask = _space_attn_mask.expand(B,self.n_time_patches,-1,self.patch_size) 
            space_attn_mask = torch.ones((B, self.n_time_patches, self.n_space_patches, self.patch_size))       
            for s_idx in range(self.n_space_patches):
                if _space_attn_mask[:,:,s_idx*self.space_patch_size:(s_idx+1)*self.space_patch_size].sum() == 0:
                    space_attn_mask[:,:,s_idx] = 0

            # create space and time positions
            if timestamps is None:
                timestamps = torch.arange(self.n_time_patches)[None,:,None]
            timestamps = timestamps.expand(B,-1,self.n_space_patches)
            
            if spacestamps is None:
                spacestamps = torch.arange(self.n_space_patches)[None,None,:]
            spacestamps = spacestamps.expand(B,self.n_time_patches,-1)

        else:
            raise NotImplementedError

        spikes_patches = spikes_patches.to(spikes.device).flatten(1,-2)
        time_attn_mask = time_attn_mask.bool().to(spikes.device).flatten(1,-2)
        space_attn_mask = space_attn_mask.bool().to(spikes.device).flatten(1,-2)
        timestamps = timestamps.bool().to(spikes.device).flatten(1)
        spacestamps = spacestamps.bool().to(spikes.device).flatten(1)
        target_masks = target_masks.bool().to(spikes.device).flatten(1,-2)

        return spikes_patches, time_attn_mask, space_attn_mask, timestamps, spacestamps, target_masks

    def _attention_mask(seq_length: int, pad_length: int,) -> torch.tensor:
        mask = torch.ones(seq_length)
        if pad_length:
            mask[-pad_length:] = 0
        else:
            mask[:pad_length] = 0
        return mask
        