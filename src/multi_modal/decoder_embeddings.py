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

DEFAULT_CONFIG = "src/configs/multi_modal/decoder.yaml"


class DecoderEmbeddingLayer(nn.Module):

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
            targets:           torch.FloatTensor,      
            targets_mask:      torch.LongTensor,    
            targets_timestamp: torch.LongTensor,  
            targets_modality:  int,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  
        
        B, _, _ = targets.size()

        x = self.token_embed(targets)

        x = self.act(x) * self.scale

        x = self.projection(x)

        x_embed = self.mod_emb(targets_modality)

        if self.pos:
            x_embed += self.pos_embed(targets_timestamp)

        return self.dropout(x), x_embed


class DecoderEmbedding(nn.Module):

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
        
        targets, targets_mask, targets_timestamp, targets_modality  = d['targets'], d['targets_mask'], d['targets_timestamp'], d['targets_modality']
        
        B, T, N = inputs.size() 
        
        x, x_emb = self.embedder(inputs, inputs_mask, inputs_timestamp, inputs_modality)

        d['x'] = x
        d['emb'] = x_emb
        d['gt'] = d['targets']

        return d

# TO DO
class Decoder(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__() 

    def forward(self):
        pass

