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
    def __init__(self, max_space_F, max_time_F, n_cls_tokens, config: DictConfig):
        super().__init__()
        self.max_space_F = max_space_F
        self.max_time_F = max_time_F
        self.n_cls_tokens = n_cls_tokens
        self.pad_value = -1.
        self.time_stride = config.time_stride

    def forward(
        self, 
        spikes: torch.FloatTensor,                       # (bs, seq_len, n_channels)
        pad_space_len,
        pad_time_len,
        time_attn_mask,       
        space_attn_mask,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:  
         # (bs, seq_len, n_channels), (bs, seq_len), (bs, seq_len), (bs, seq_len), (bs, seq_len, n_channels)

        B, T, N = spikes.size()

        self.n_time_patches = ceil(T/self.max_time_F)
        self.n_space_patches = ceil(N/self.max_space_F)
        self.n_channels = self.max_time_F * self.max_space_F

        self.pad_space_len = torch.tensor(
            [pad_space_len[b]%self.max_space_F for b in range(B)]
        ) 
        self.pad_space_idx = torch.tensor(
            [floor(pad_space_len[b]/self.max_space_F) for b in range(B)]
        ) 

        # Group spikes into patches
        patches = torch.ones((B, self.n_time_patches, self.n_space_patches, self.n_channels)) * self.pad_value    
        for b in range(B):
            for t in range(self.n_time_patches):
                for s in range(self.n_space_patches):
                    patches[b,t,s] = spikes[b,t*self.max_time_F:(t+1)*self.max_time_F, \
                    s*self.max_space_F:(s+1)*self.max_space_F].flatten()
                    if self.pad_space_len[b] != 0:
                        if s == self.pad_space_idx[b]:
                            patches[b,t,s] = torch.cat(
                                (patches[b,t,s][:self.pad_space_len[b]], 
                                 patches[b,t,s-1][-(self.max_space_F - self.pad_space_len[b]):]
                                ), dim=0
                            )                       
        patches = patches.flatten(1,2).to(spikes.device)

        # Prepare space and time stamps after accounting for [cls] tokens
        spacestamps = torch.arange(self.n_space_patches+self.n_cls_tokens).to(torch.int64)[None,None,:]
        spacestamps = spacestamps.expand(B, self.n_time_patches,-1).to(spikes.device).flatten(1)
        timestamps = torch.arange(self.n_time_patches).to(torch.int64)[None,:,None]
        timestamps = timestamps.expand(B, -1, self.n_space_patches+self.n_cls_tokens).to(spikes.device).flatten(1)
        
        # Prepare space and time masks after accounting for [cls] tokens
        # _time_attn_mask = self._attention_mask(T, pad_time_len)[None,:,None]
        # _time_attn_mask = _time_attn_mask.expand(B,-1,self.n_space_patches+self.n_cls_tokens)
        _time_attn_mask = time_attn_mask[:,:,None].expand(-1,-1,self.n_space_patches+self.n_cls_tokens) 
        time_attn_mask = torch.zeros((B, self.n_time_patches, self.n_space_patches+self.n_cls_tokens))
        time_attn_mask[:,:,:self.n_cls_tokens] = 1
        for b in range(B):
            for t in range(self.n_time_patches):
                if (_time_attn_mask[b,t*self.max_time_F:(t+1)*self.max_time_F]==1).sum() > 0:
                    time_attn_mask[b,t,self.n_cls_tokens:] = 1

        # _space_attn_mask = self._attention_mask(N, pad_space_len)[None,None,:]
        # _space_attn_mask = _space_attn_mask.expand(B,self.n_time_patches,-1) 
        _space_attn_mask = space_attn_mask[:,None,:].expand(-1,self.n_time_patches,-1) 
        space_attn_mask = torch.zeros((B, self.n_time_patches, self.n_space_patches+self.n_cls_tokens))   
        space_attn_mask[:,:,:self.n_cls_tokens] = 1
        for b in range(B):
            for s in range(self.n_space_patches):
                if (_space_attn_mask[b,:,s*self.max_space_F:(s+1)*self.max_space_F]==1).sum() > 0:
                    space_attn_mask[b,:,self.n_cls_tokens+s] = 1

        time_attn_mask = time_attn_mask.to(torch.int64).to(spikes.device).flatten(1)
        space_attn_mask = space_attn_mask.to(torch.int64).to(spikes.device).flatten(1)

        return patches, space_attn_mask, time_attn_mask, spacestamps, timestamps

    def _attention_mask(self, seq_length: int, pad_length: int) -> torch.tensor:
        mask = torch.ones(seq_length)
        if pad_length:
            mask[-pad_length:] = 0
        else:
            mask[:pad_length] = 0
        return mask
