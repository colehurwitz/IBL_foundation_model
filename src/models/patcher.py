from math import floor, ceil
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig

""" Module for patching spikes. 

CONFIG:
    max_space_F:
    max_time_F:
    time_stride:
    n_cls_tokens:
"""
# TO DO: Add time strides?

class Patcher(nn.Module):
    def __init__(self, max_space_F, max_time_F, n_cls_tokens, config: DictConfig):
        super().__init__()
        self.max_space_F = max_space_F
        self.max_time_F = max_time_F
        self.n_cls_tokens = n_cls_tokens
        self.time_stride = config.time_stride
        self.pad_value = -1.

    def forward(
        self, 
        spikes: torch.FloatTensor,                       # (bs, seq_len, n_channels)
        spikes_masks: torch.LongTensor,                  # (bs, seq_len, n_channels)
        pad_space_len,
        pad_time_len
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:  
         # (bs, seq_len, n_channels), (bs, seq_len), (bs, seq_len), (bs, seq_len), (bs, seq_len)

        B, T, N = spikes.size()

        # self.n_time_patches = T//self.max_time_F
        # self.n_space_patches = N//self.max_space_F
        self.n_time_patches = floor(T/self.max_time_F)
        self.n_space_patches = floor(N/self.max_space_F)
        self.n_channels = self.max_time_F * self.max_space_F

        # group neurons into patches
        patches = torch.ones(
            (B, self.n_time_patches, self.n_space_patches, self.n_channels)
        ) * self.pad_value    
        for t in range(self.n_time_patches):
            for s in range(self.n_space_patches):
                patches[:,t,s] = spikes[:,t*self.max_time_F:(t+1)*self.max_time_F, s*self.max_space_F:(s+1)*self.max_space_F].flatten(1)
        patches = patches.flatten(1,2).to(spikes.device)

        target_masks = torch.ones(
            (B, self.n_time_patches, self.n_space_patches, self.n_channels)
        )     
        for t in range(self.n_time_patches):
            for s in range(self.n_space_patches):
                target_masks[:,t,s] = spikes_masks[:,t*self.max_time_F:(t+1)*self.max_time_F, s*self.max_space_F:(s+1)*self.max_space_F].flatten(1)
        target_masks = target_masks.flatten(1,2).to(spikes.device)

        # Prepend space and time stamps for [cls] tokens
        spacestamps = torch.arange(self.n_space_patches+self.n_cls_tokens).to(torch.int64)[None,None,:]
        spacestamps = spacestamps.expand(B, self.n_time_patches,-1).to(spikes.device).flatten(1)
        timestamps = torch.arange(self.n_time_patches).to(torch.int64)[None,:,None]
        timestamps = timestamps.expand(B, -1, self.n_space_patches+self.n_cls_tokens).to(spikes.device).flatten(1)
        
        # Prepend space and time masks for [cls] tokens
        B, T, N = spikes.size()
        _time_attn_mask = self._attention_mask(T, pad_time_len)[None,:,None]
        _space_attn_mask = self._attention_mask(N, pad_space_len)[None,None,:]

        _time_attn_mask = _time_attn_mask.expand(B,-1,self.n_space_patches+ self.n_cls_tokens)
        time_attn_mask = torch.ones((B, self.n_time_patches, self.n_space_patches+self.n_cls_tokens))
        for t_idx in range(self.n_time_patches):
            if _time_attn_mask[:,t_idx*self.max_time_F:(t_idx+1)*self.max_time_F].sum() != self.n_channels:
                time_attn_mask[:,t_idx,self.n_time_patches*self.n_cls_tokens:] = 0
        
        _space_attn_mask = _space_attn_mask.expand(B,self.n_time_patches,-1) 
        space_attn_mask = torch.ones((B, self.n_time_patches, self.n_space_patches+ self.n_cls_tokens))       
        for s_idx in range(self.n_space_patches):
            if _space_attn_mask[:,:,s_idx*self.max_space_F:(s_idx+1)*self.max_space_F].sum() != self.n_channels:
                space_attn_mask[:,:,self.n_cls_tokens+s_idx] = 0

        time_attn_mask = time_attn_mask.to(torch.int64).to(spikes.device).flatten(1)
        space_attn_mask = space_attn_mask.to(torch.int64).to(spikes.device).flatten(1)

        return patches, space_attn_mask, time_attn_mask, spacestamps, timestamps, target_masks

    def _attention_mask(self, seq_length: int, pad_length: int) -> torch.tensor:
        mask = torch.ones(seq_length)
        if pad_length:
            mask[-pad_length:] = 0
        else:
            mask[:pad_length] = 0
        return mask


