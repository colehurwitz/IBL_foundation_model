import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config

def create_context_mask(context_forward, context_backward, max_F) -> torch.LongTensor: 
    
    if context_forward == -1 and context_backward == -1:
        return torch.ones(max_F, max_F).to(torch.int64)

    context_forward = context_forward if context_forward >= 0 else max_F
    context_backward = context_backward if context_backward >= 0 else max_F
    mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
    if context_backward > 0:
        back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.int64))
        mask = mask & back_mask
    return mask


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
        

class MLP(nn.Module):
    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()
        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))


class FactorsProjection(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.out_size = config.size if config.active else hidden_size        
        self.dropout = nn.Dropout(config.dropout)
        if config.active:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
            if config.fixup_init:
                self.proj[0].weight.data.uniform_(-config.init_range, config.init_range)
                if config.bias:
                    self.proj[0].bias.data.zero_()
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.dropout(x))


class Attention(nn.Module):
    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout, max_F=1024):
        super().__init__()
        
        self.idx = idx

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, "Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        # torch.backends.cuda.enable_flash_sdp(True)
        self.attn_dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

    def forward(
        self,       
        x:          torch.FloatTensor,                      
        attn_mask:  torch.LongTensor,                       
    ) -> torch.FloatTensor:                                

        B, T, _  = x.size()    

        attn_mask = attn_mask.unsqueeze(1).expand(B,self.n_heads,T,T).bool()           
        
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)        
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)     

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.attn_dropout if self.training else 0.0), is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size) 

        return self.out_proj(self.dropout(out)) 


class CrossAttention(nn.Module):
    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout, max_F=1024):
        super().__init__()

        self.idx = idx
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, "Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads
        
        self.scale = self.head_size ** -0.5

        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.kv = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=use_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, context, attn_mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "b n m -> b 1 n m") # unsqueeze / reshape for multi-head
            attn = attn.masked_fill(attn_mask, -torch.finfo(attn.dtype).max)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




    
        