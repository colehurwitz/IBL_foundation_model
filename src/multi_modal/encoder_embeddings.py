import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from multi_modal.mm_utils import ScaleNorm, MLP, Attention

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"


class EncoderEmbeddingLayer(nn.Module):

    def __init__(self, hidden_size, n_channels, config: DictConfig):
        super().__init__()

        self.bias = config.bias
        self.n_channels = n_channels
        self.input_dim = self.n_channels*config.mult

        self.token_embed = nn.Linear(self.n_channels, self.input_dim, bias=self.bias)

        self.projection = nn.Linear(self.input_dim, hidden_size)

        self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()

        self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

        self.mod_emb = nn.Embedding(config.n_modality, hidden_size)

        self.pos = config.pos
        if self.pos:
            self.pos_embed = nn.Embedding(config.max_F, hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, d : Dict[str, torch.Tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  

        inputs, inputs_timestamp, inputs_modality  = d['inputs'], d['inputs_timestamp'], d['inputs_modality']
        
        B, N, _ = inputs.size()

        x = self.token_embed(inputs)

        x = self.act(x) * self.scale

        x = self.projection(x)

        x_embed = self.mod_emb(inputs_modality)[None,None,:].expand(B,N,-1).clone()

        if self.pos:
            x_embed += self.pos_embed(inputs_timestamp)

        return self.dropout(x), x_embed


class EncoderEmbedding(nn.Module):

    def __init__(
        self, 
        n_channel,
        config: DictConfig,
        **kwargs
    ):
        super().__init__() 

        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers
        self.max_F = config.embedder.max_F
        self.n_channel = n_channel

        self.embedder = EncoderEmbeddingLayer(self.hidden_size, self.n_channel, config.embedder)
    
    def forward(self, d : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
                        
        x, x_emb = self.embedder(d)

        d['x'] = x
        d['emb'] = x_emb

        return d


class EncoderLayer(nn.Module):
    
    def __init__(self, idx, config: DictConfig):
        super().__init__()

        self.idx = idx
    
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.attn = Attention(idx, config.hidden_size, config.n_heads, config.attention_bias, config.dropout)
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.mlp = MLP(config.hidden_size, config.inter_size, config.act, config.mlp_bias, config.dropout)

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:     torch.FloatTensor,                  
        mask:  torch.LongTensor,                  
    ) -> torch.FloatTensor :                           
        
        x = x + self.attn(self.ln1(x), mask)

        x = x + self.mlp(self.ln2(x))

        return x

    def fixup_initialization(self, n_layers):
        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * param
            elif name.endswith("value.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * (param * (2**0.5))
                
        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)   



