import os
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Union

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput
from multi_modal.encoder_embeddings import EncoderLayer
from multi_modal.decoder_embeddings import DecoderLayer
from models.masker import Masker
from multi_modal.mm_utils import create_context_mask

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

@dataclass
class MultiModalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mod_loss: Optional[torch.FloatTensor] = None
    mod_n_examples: Optional[torch.LongTensor] = None
    mod_preds: Optional[torch.FloatTensor] = None
    mod_targets: Optional[torch.FloatTensor] = None


class MultiModal(nn.Module):
    def __init__(
        self, 
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        avail_mod:          List,
        config: DictConfig,
        share_modality_embeddings: bool = True,
        **kwargs
    ):
        super().__init__()

        self.avail_mod = avail_mod
        self.mod_to_indx = {r: i for i,r in enumerate(self.avail_mod)}
        self.decoder_sep_mask = config.decoder.decoder_sep_mask

        self.n_enc_layers = config.encoder.transformer.n_layers
        self.n_dec_layers = config.decoder.transformer.n_layers
        self.hidden_size = config.encoder.transformer.hidden_size
        self.max_F = config.encoder.embedder.max_F
        self.context_forward = config.context.forward
        self.context_backward = config.context.backward

        self.encoder_modalities = set(encoder_embeddings.keys())
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        self.decoder_modalities = set(decoder_embeddings.keys())
        self.decoder_embeddings = nn.ModuleDict(decoder_embeddings)

        if share_modality_embeddings:
            self.share_modality_embeddings()

        self.mask = config.masker.force_active
        if self.mask:
            assert config.masker.mode in ['temporal'], "Only token-wise masking is allowed for multi-modal model for now."
            self.masker = Masker(config.masker)

        self.encoder = nn.ModuleList([EncoderLayer(idx, config.encoder.transformer) for idx in range(self.n_enc_layers)])
        self.encoder_norm = nn.LayerNorm(self.hidden_size) 

        self.decoder_proj_context = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.decoder = nn.ModuleList([DecoderLayer(idx, config.decoder.transformer) for idx in range(self.n_dec_layers)])
        self.decoder_norm = nn.LayerNorm(self.hidden_size) 

        self.loss_mod = {
            'ap': nn.PoissonNLLLoss(reduction="none", log_input=True),
            'behavior': nn.MSELoss(reduction="none"),
        }
        
    def share_modality_embeddings(self):
        shared_modalities = self.encoder_modalities & self.decoder_modalities
        for mod in shared_modalities:
            self.decoder_embeddings[mod].embedder.mod_emb = self.encoder_embeddings[mod].embedder.mod_emb

    
    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        encoder_tokens = []
        encoder_emb = []
        encoder_mask = []
        attention_mask = []
        mod_mask = []

        for mod, d in mod_dict.items():
            encoder_tokens.append(d['x'])
            encoder_emb.append(d['emb'])
            encoder_mask.append(d['inputs_mask'])
            attention_mask.append(d['encoder_attn_mask'])
            mod_mask.append(torch.full_like(d['inputs_mask'], self.mod_to_indx[mod], dtype=torch.int16))
    
        encoder_tokens = torch.cat(encoder_tokens, dim=1)
        encoder_emb = torch.cat(encoder_emb, dim=1)
        encoder_mask = torch.cat(encoder_mask, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)
        mod_mask = torch.cat(mod_mask, dim=1)

        return encoder_tokens, encoder_emb, encoder_mask, attention_mask, mod_mask

    
    def cat_decoder_tensors(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        decoder_tokens = []
        target_gts = {}
        decoder_emb = []
        decoder_mask = []
        attention_mask = []
        mod_mask = []

        # shuffle order in which modalities are provided (useful for modality causal mask)
        # mod_dict = {mod: d for mod, d in random.sample(mod_dict.items(), len(mod_dict))}

        for mod, d in mod_dict.items():
            decoder_tokens.append(d['x'])
            target_gts[mod] = d['gt']
            decoder_emb.append(d['emb'])
            decoder_mask.append(d['targets_mask'])
            attention_mask.append(d['decoder_attn_mask'])
            mod_mask.append(torch.full_like(d['targets_mask'], self.mod_to_indx[mod], dtype=torch.int16))
        
        decoder_tokens = torch.cat(decoder_tokens, dim=1)
        decoder_emb = torch.cat(decoder_emb, dim=1)
        decoder_mask = torch.cat(decoder_mask, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)
        mod_mask = torch.cat(mod_mask, dim=1)

        return decoder_tokens, target_gts, decoder_emb, decoder_mask, attention_mask, mod_mask

    
    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, mod_mask = self.cat_encoder_tensors(mod_dict)

        B, N, _ = encoder_tokens.size()
        
        encoder_tokens[encoder_mask] = 0.
        encoder_emb[encoder_mask] = 0.

        encoder_attn_mask = encoder_attn_mask.unsqueeze(1).expand(B,N,N)
        self_mask = torch.eye(N).to(encoder_attn_mask.device, torch.int64).expand(B,N,N)
        # TO DO: Change context_mask
        context_mask = torch.ones_like(encoder_attn_mask).to(encoder_attn_mask.device, torch.int64)
        encoder_attn_mask = self_mask | (context_mask & encoder_attn_mask)
        
        return encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, mod_mask

    
    def forward_mask_decoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        decoder_tokens, target_gts, decoder_emb, decoder_mask, decoder_attn_mask, mod_mask = self.cat_decoder_tensors(mod_dict)

        B, N, _ = decoder_tokens.size()

        decoder_tokens[decoder_mask] = 0.
        decoder_emb[decoder_mask] = 0.
        decoder_attn_mask = self.adapt_decoder_attention_mask(decoder_attn_mask, mod_mask)
        
        return decoder_tokens, target_gts, decoder_emb, decoder_mask, decoder_attn_mask, mod_mask
        
    # TO DO
    def adapt_decoder_attention_mask(self, decoder_attn_mask: torch.Tensor, mod_mask=Optional[torch.Tensor]) -> torch.Tensor:

        B, N = decoder_attn_mask.shape
        
        if self.decoder_sep_mask:
            # separate attention between tokens based on their modality using mod_mask.
            sep_mask = repeat(mod_mask, "b n2 -> b n1 n2", n1=N) != repeat(mod_mask, "b n1 -> b n1 n2", n2=N)
            adapted_attn_mask = decoder_attn_mask | sep_mask
        else:
            adapted_attn_mask = decoder_attn_mask

        adapted_attn_mask = adapted_attn_mask.unsqueeze(1).expand(B,N,N)

        return adapted_attn_mask

    
    def forward_encoder(self, x: torch.Tensor, encoder_attn_mask: torch.Tensor) -> torch.Tensor:
        
        for layer in self.encoder:
            x = layer(x, mask=encoder_attn_mask)

        x = self.encoder_norm(x)

        return x

    
    def forward_decoder(self, y: torch.Tensor, context: torch.Tensor, encoder_attn_mask: torch.Tensor, decoder_attn_mask: torch.Tensor) -> torch.Tensor:

        for layer in self.decoder:
            y = layer(y, context, sa_mask=decoder_attn_mask, xa_mask=encoder_attn_mask)

        y = self.decoder_norm(y)

        return y
    

    def forward_loss(self, 
        decoder_mod_dict: Dict[str, Any], target_gts: torch.Tensor, 
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        mod_loss, mod_n_examples, mod_preds, mod_targets = {}, {}, {}, {}
        for mod, d in decoder_mod_dict.items():
            targets = target_gts[mod]
            B, T, N = targets.size()
            preds = decoder_mod_dict[mod]['preds']
            targets_mask = 1 - decoder_mod_dict[mod]['targets_mask'].unsqueeze(-1).expand(B,T,N)
            loss = (self.loss_mod[mod](preds, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()
            mod_loss[mod] = loss
            mod_n_examples[mod] = n_examples
            mod_preds[mod] = preds
            mod_targets[mod] = targets

        loss = sum(mod_loss.values()) / sum(mod_n_examples.values())

        return loss, mod_loss, mod_n_examples, mod_preds, mod_targets

    
    def forward(
            self, mod_dict: Dict[str, Dict[str, torch.Tensor]]
        ) -> MultiModalOutput:

        for mod, d in mod_dict.items():

            # TO DO
            if mod == 'behavior':
                mod_dict[mod]['inputs'] = mod_dict[mod]['inputs'].unsqueeze(-1)
                mod_dict[mod]['targets'] = mod_dict[mod]['targets'].unsqueeze(-1)
                
            B, N, D = mod_dict[mod]['inputs'].size()
            if self.mask:
                inputs_regions = mod_dict[mod]['inputs_regions'] if mod == 'ap' else None
                _, mask = self.masker(mod_dict[mod]['inputs'].clone(), inputs_regions)
                mask = mask[:,:,0] & mod_dict[mod]['inputs_attn_mask'] # use token-wise mask now and expand later
            else:
                mask = mod_dict[mod]['inputs_attn_mask']
                
            mod_dict[mod]['inputs_mask'] = mask
            mod_dict[mod]['targets_mask'] = mask
            mod_dict[mod]['encoder_attn_mask'] = mod_dict[mod]['inputs_attn_mask']
            mod_dict[mod]['decoder_attn_mask'] = mod_dict[mod]['inputs_attn_mask']

        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}

        encoder_tokens, encoder_emb, encoder_mask, encoder_attn_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict)

        decoder_mod_dict = {mod: self.decoder_embeddings[mod].forward_embed(d)
                            for mod, d in mod_dict.items()
                            if mod in self.decoder_embeddings}

        decoder_tokens, target_gts, decoder_emb, decoder_mask, decoder_attn_mask, decoder_mod_mask = self.forward_mask_decoder(decoder_mod_dict)

        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_attn_mask=encoder_attn_mask)

        context = self.decoder_proj_context(x) + encoder_emb
        y = decoder_tokens + decoder_emb
        y = self.forward_decoder(y, context, encoder_attn_mask=encoder_attn_mask, decoder_attn_mask=decoder_attn_mask)

        decoder_mod_dict = {mod: self.decoder_embeddings[mod].out_proj(self.mod_to_indx[mod], d, y, decoder_mod_mask, len(self.avail_mod))
                            for mod, d in decoder_mod_dict.items()
                            if mod in self.decoder_embeddings}

        loss, mod_loss, mod_n_examples, mod_preds, mod_targets = self.forward_loss(decoder_mod_dict, target_gts)

        return MultiModalOutput(
            loss=loss,
            mod_loss=mod_loss,
            mod_n_examples=mod_n_examples,
            mod_preds=mod_preds,
            mod_targets=mod_targets
        )

    
