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

DEFAULT_CONFIG = "src/configs/multi_modal/encoder.yaml"


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

    def forward(
            self, 
            inputs:           torch.FloatTensor,      
            inputs_mask:      torch.LongTensor,    
            inputs_timestamp: torch.LongTensor,  
            inputs_modality:  int,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  
        
        B, _, _ = inputs.size()

        x = self.token_embed(inputs)

        x = self.act(x) * self.scale

        x = self.projection(x)

        x_embed = self.mod_emb(inputs_modality)

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
        
        inputs, inputs_mask, inputs_timestamp, inputs_modality  = d['inputs'], d['inputs_mask'], d['inputs_timestamp'], d['inputs_modality']
        
        B, T, N = inputs.size() 
        
        x, x_emb = self.embedder(inputs, inputs_mask, inputs_timestamp, inputs_modality)

        d['x'] = x
        d['emb'] = x_emb

        return d

# TO DO
class Encoder(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__() 

    def forward(self):
        pass

    
