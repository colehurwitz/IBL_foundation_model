
from typing import Tuple, Optional, List
from math import ceil, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config_utils import DictConfig

from torch import no_grad

""" Module for patching spikes. 

CONFIG:
    max_space_F: space patch size (# neurons per patch)
    max_time_F:  time patch size (# timesteps per patch)
    time_stride: 
"""

import torch
import torch.nn.functional as F
from math import ceil

def Patcher(
    spikes: torch.FloatTensor,      # (B, T, N)
    pad_time_len: int,
    time_attn_mask: torch.LongTensor,  # (B, T)
    space_attn_mask: torch.LongTensor,  # (B, N)
    max_time_F: int,
    pad_value: float = -1.0,
):
    """
    Efficiently patches neural data by chunking in time, optimized for cases where 
    n_space_patches equals the number of neurons (max_space_F = 1).

    Args:
        spikes (torch.FloatTensor): Input spikes tensor of shape (B, T, N).
        pad_time_len (int): Padding length for the time dimension.
        time_attn_mask (torch.LongTensor): Time attention mask of shape (B, T).
        space_attn_mask (torch.LongTensor): Space attention mask of shape (B, N).
        max_time_F (int): Maximum number of time frames per patch.
        pad_value (float, optional): Value to use for padding. Defaults to -1.0.

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
            - patches (torch.FloatTensor): Patched spikes of shape (B, new_seq_len, n_channels).
            - patch_mask (torch.LongTensor): Combined attention mask for patches.
            - spacestamps (torch.LongTensor): Spatial indices for patches.
            - timestamps (torch.LongTensor): Temporal indices for patches.
    """
    B, T, N = spikes.size()
    device = spikes.device

    # Calculate the number of time patches
    n_time_patches = ceil(T / max_time_F)
    total_padded_length = n_time_patches * max_time_F
    pad_time_len = total_padded_length - T  # Total padding required in time dimension

    # Pad spikes along the time dimension
    spikes = F.pad(spikes, (0, 0, 0, pad_time_len), value=pad_value)  # Shape: (B, total_padded_length, N)
    # Pad time attention mask
    time_attn_mask = F.pad(time_attn_mask, (0, pad_time_len), value=0)  # Shape: (B, total_padded_length)

    # Reshape and permute spikes to shape (B, n_time_patches, N, max_time_F)
    spikes = spikes.view(B, n_time_patches, max_time_F, N).permute(0, 1, 3, 2)

    # Flatten the time dimension to create patches
    n_channels = max_time_F
    patches = spikes.reshape(B, n_time_patches, N, n_channels)

    # Flatten n_time_patches and N dimensions to create the sequence dimension
    patches = patches.reshape(B, n_time_patches * N, n_channels)  # Shape: (B, n_time_patches * N, n_channels)

    # Prepare the patch mask
    # # Reshape time attention mask to (B, n_time_patches, max_time_F)
    # time_attn_mask = time_attn_mask.view(B, n_time_patches, max_time_F)
    # # Compute time patch mask: (B, n_time_patches)
    # time_patch_mask = (time_attn_mask.sum(dim=2) > 0).long()
    # # Expand to (B, n_time_patches, N)
    # time_patch_mask = time_patch_mask[:, :, None].expand(-1, -1, N)
    # # Expand space attention mask to (B, n_time_patches, N)
    # space_attn_mask = space_attn_mask[:, None, :].expand(-1, n_time_patches, -1)
    # # Compute combined patch mask
    # patch_mask = time_patch_mask * space_attn_mask  # Shape: (B, n_time_patches, N)
    # # Flatten to (B, n_time_patches * N)
    # patch_mask = patch_mask.view(B, n_time_patches * N)

    # Prepare timestamps and spacestamps
    timestamps = torch.arange(n_time_patches, device=device).unsqueeze(1).expand(n_time_patches, N).flatten()
    spacestamps = torch.arange(N, device=device).unsqueeze(0).expand(n_time_patches, N).flatten()
    # Expand to match batch size
    timestamps = timestamps.unsqueeze(0).expand(B, -1)  # Shape: (B, n_time_patches * N)
    spacestamps = spacestamps.unsqueeze(0).expand(B, -1)  # Shape: (B, n_time_patches * N)

    return patches, spacestamps, timestamps




# class Patcher(nn.module):
#     def __init__(self, max_space_F, max_time_F, embed_region, config: DictConfig):
#         super().__init__()
#         self.max_space_F = max_space_F
#         self.max_time_F = max_time_F
#         self.pad_value = -1.
#         self.time_stride = config.time_stride
#         self.embed_region = embed_region

#     @no_grad
#     def forward(
#         self, 
#         spikes:          torch.FloatTensor,     # (bs, seq_len, n_channels)
#         pad_space_len:   int,      
#         pad_time_len:    int,
#         time_attn_mask:  torch.LongTensor,      # (bs, seq_len,)
#         space_attn_mask: torch.LongTensor,       # (bs, seq_len,)
#         region_indx:     Optional[torch.LongTensor] = None,
#     ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:  
#          # (bs, seq_len, n_channels), (bs, seq_len), (bs, seq_len), (bs, seq_len), (bs, seq_len, n_channels)

#         B, T, N = spikes.size()

#         self.n_time_patches = ceil(T/self.max_time_F)
#         self.n_space_patches = ceil(N/self.max_space_F)
#         self.n_channels = self.max_time_F * self.max_space_F

#         self.pad_space_len = pad_space_len % self.max_space_F
#         self.pad_space_idx = floor(pad_space_len/self.max_space_F)
        
#         # Group spikes into patches
#         patches = torch.ones((B, self.n_time_patches, self.n_space_patches, self.n_channels)) * self.pad_value 
#         for t in range(self.n_time_patches):
#             for s in range(self.n_space_patches):
#                 patches[:,t,s] = spikes[:,t*self.max_time_F:(t+1)*self.max_time_F, \
#                 s*self.max_space_F:(s+1)*self.max_space_F].flatten(1)
#                 if self.pad_space_len != 0:
#                     if s == self.pad_space_idx:
#                         patches[:,t,s] = torch.cat(
#                             (patches[:,t,s][:,:self.pad_space_len], 
#                              patches[:,t,s-1][:,-(self.max_space_F - self.pad_space_len):]
#                             ), dim=1
#                         )                       
#         patches = patches.flatten(1,2).to(spikes.device)

#         # Prepare space and time stamps after accounting for [cls] tokens
#         # Majority vote the brain region for each patch
#         if self.embed_region: 
#             regionstamps = torch.zeros(self.n_space_patches)
#             for s in range(self.n_space_patches):
#                 if len(region_indx[0, s*self.max_space_F:(s+1)*self.max_space_F]) != 0:
#                     regionstamps[s] = torch.mode(
#                         region_indx[0, s*self.max_space_F:(s+1)*self.max_space_F]
#                     ).values
#             regionstamps = regionstamps.to(torch.int64)[None,None,:]
#             regionstamps = regionstamps.expand(B, self.n_time_patches,-1).to(spikes.device).flatten(1)
#         else:
#             regionstamps = None
        
#         spacestamps = torch.arange(self.n_space_patches).to(torch.int64)[None,None,:]
#         spacestamps = spacestamps.expand(B, self.n_time_patches,-1).to(spikes.device).flatten(1)
        
#         timestamps = torch.arange(self.n_time_patches).to(torch.int64)[None,:,None]
#         timestamps = timestamps.expand(B, -1, self.n_space_patches).to(spikes.device).flatten(1)
        
#         # Prepare space and time masks after accounting for [cls] tokens
#         _time_attn_mask = time_attn_mask[:,:,None].expand(-1,-1,self.n_space_patches) 
#         time_attn_mask = torch.zeros((B, self.n_time_patches, self.n_space_patches))
#         for t in range(self.n_time_patches):
#             # if (_time_attn_mask[:,t*self.max_time_F:(t+1)*self.max_time_F]==1).sum() > 0:
#             if (_time_attn_mask[:,t*self.max_time_F:(t+1)*self.max_time_F]==0).sum() == 0:
#                 time_attn_mask[:,t,:] = 1

#         _space_attn_mask = space_attn_mask[:,None,:].expand(-1,self.n_time_patches,-1) 
#         space_attn_mask = torch.zeros((B, self.n_time_patches, self.n_space_patches))   
#         for s in range(self.n_space_patches):
#             if (_space_attn_mask[:,:,s*self.max_space_F:(s+1)*self.max_space_F]==1).sum() > 0:
#                 space_attn_mask[:,:,s] = 1

#         time_attn_mask = time_attn_mask.to(torch.int64).to(spikes.device).flatten(1)
#         space_attn_mask = space_attn_mask.to(torch.int64).to(spikes.device).flatten(1)

#         return patches, space_attn_mask, time_attn_mask, spacestamps, timestamps, regionstamps









