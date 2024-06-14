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

with open('data/target_eids.txt') as file:
    include_eids = [line.rstrip() for line in file]

@dataclass
class MultiModalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None


class MultiModal(nn.Module):
    def __init__(
        self, 
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        avail_mod:          List,
        config: DictConfig,
        share_modality_embeddings: bool = True,
        use_session: bool = False,
        **kwargs
    ):
        super().__init__()

        self.avail_mod = avail_mod
        self.mod_to_indx = {r: i for i,r in enumerate(self.avail_mod)}
        self.use_session = use_session
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
            assert config.masker.mode in ['temporal', 'causal'], "Only token-wise masking is allowed for multi-modal model for now."
            self.masker = Masker(config.masker)

        self.encoder = nn.ModuleList([EncoderLayer(idx, self.max_F, config.encoder.transformer) for idx in range(self.n_enc_layers)])
        self.encoder_norm = nn.LayerNorm(self.hidden_size) 

        self.decoder_proj_context = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.decoder = nn.ModuleList([DecoderLayer(idx, self.max_F, config.decoder.transformer) for idx in range(self.n_dec_layers)])
        self.decoder_norm = nn.LayerNorm(self.hidden_size) 
            
        if self.use_session:
            self.eid_lookup = include_eids
            self.eid_to_indx = {r: i for i,r in enumerate(self.eid_lookup)}
            self.session_embed = nn.Embedding(len(self.eid_lookup), self.hidden_size) 

        self.loss_mod = {
            'ap': nn.PoissonNLLLoss(reduction="none", log_input=True),
            'behavior': nn.MSELoss(reduction="none"),
        }
        
    def share_modality_embeddings(self):
        shared_modalities = self.encoder_modalities & self.decoder_modalities
        for mod in shared_modalities:
            self.decoder_embeddings[mod].embedder.mod_emb = self.encoder_embeddings[mod].embedder.mod_emb

    
    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        encoder_tokens_all = []
        emb_all = []
        encoder_attn_mask_all = []
        mod_mask_all = []
        inputs_mask_all = []

        for mod, d in mod_dict.items():
            encoder_tokens_all.append(d['x'])
            emb_all.append(d['emb'])
            encoder_attn_mask_all.append(d['encoder_attn_mask'])
            mod_mask_all.append(torch.full_like(d['encoder_attn_mask'], self.mod_to_indx[mod], dtype=torch.int16))
            inputs_mask_all.append(d['inputs_mask'])

        encoder_tokens_all = torch.cat(encoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        encoder_attn_mask_all = torch.cat(encoder_attn_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)
        inputs_mask_all = torch.cat(inputs_mask_all, dim=1)

        print('encoder_attn_mask_concat size:', encoder_attn_mask_all.size())

        return encoder_tokens_all, emb_all, encoder_attn_mask_all, mod_mask_all, inputs_mask_all

    
    def cat_decoder_tensors(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        decoder_tokens_all = []
        emb_all = []
        decoder_attn_mask_all = []
        mod_mask_all = []
        targets_mask_all = []
        target_gts_all = []

        # shuffle order in which modalities are provided (useful for modality causal mask)
        # mod_dict = {mod: d for mod, d in random.sample(mod_dict.items(), len(mod_dict))}

        for mod, d in mod_dict.items():
            decoder_tokens_all.append(d['x'])
            emb_all.append(d['emb'])
            decoder_attn_mask_all.append(d['decoder_attn_mask'])
            mod_mask_all.append(torch.full_like(d['id'], self.mod_to_indx[mod], dtype=torch.int16))
            targets_mask_all.append(d['targets_mask'])
            target_gts_all.append(d['gt'])

        decoder_tokens_all = torch.cat(decoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        decoder_attn_mask_all = torch.cat(decoder_attn_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)
        targets_mask_all = torch.cat(targets_mask_all, dim=1)
        target_gts_all = torch.cat(target_gts_all, dim=1)

        return decoder_tokens_all, emb_all, decoder_attn_mask_all, mod_mask_all, targets_mask_all, target_gts_all

    
    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        B = list(mod_dict.values())[0]['inputs'].shape[0]

        encoder_tokens, encoder_emb, encoder_attn_mask, mod_mask, inputs_mask = self.cat_encoder_tensors(mod_dict)
        
        if self.use_session:
            sess_idx = torch.tensor(self.eid_to_indx[mod_dict[mod]['eid']], dtype=torch.int64, device=encoder_tokens.device)
            sess_token = self.session_embed(sess_idx)[None,None,:].expand(B,-1,-1)
            encoder_tokens = torch.cat([sess_token, encoder_tokens], dim=1)
            encoder_emb = torch.cat([torch.zeros_like(sess_token), encoder_emb], dim=1)
            encoder_attn_mask = torch.cat([torch.zeros((B, sess_token.shape[1]), dtype=torch.bool, device=encoder_attn_mask.device), encoder_attn_mask], dim=1)
            mod_mask = torch.cat([torch.full((B, sess_token.shape[1]), -1, dtype=torch.int16, device=mod_mask.device), mod_mask], dim=1)
            inputs_mask = torch.cat([torch.zeros((B, sess_token.shape[1]), dtype=torch.bool, device=inputs_mask.device), inputs_mask], dim=1)

        encoder_tokens[inputs_mask] = 0.
        encoder_emb[inputs_mask] = 0.
        mod_mask[inputs_mask] = -1
        # mask could be of shape 'b n1 n2' but not needed for masked_fill
        # this means this mask can then be re-used for decoder cross-attention
        # encoder_attn_mask = rearrange(encoder_attn_mask, 'b n2 -> b 1 n2')
        
        return encoder_tokens, encoder_emb, encoder_attn_mask, mod_mask, inputs_mask

    
    def forward_mask_decoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        decoder_tokens, decoder_emb, decoder_attn_mask, mod_mask, targets_mask, target_gts_all = self.cat_decoder_tensors(mod_dict)

        decoder_tokens[targets_mask] = 0.
        decoder_emb[targets_mask] = 0.
        decoder_attn_mask = self.adapt_decoder_attention_mask(decoder_attn_mask, mod_mask)
        mod_mask[targets_mask] = -1
        target_gts_all[targets_mask] = 0.

        # this means this mask can then be re-used for decoder cross-attention
        # decoder_attn_mask = rearrange(decoder_attn_mask, 'b n2 -> b 1 n2')
        
        return decoder_tokens, decoder_emb, decoder_attn_mask, mod_mask, targets_mask
        
    
    def adapt_decoder_attention_mask(self, decoder_attn_mask: torch.Tensor, mod_mask=Optional[torch.Tensor]) -> torch.Tensor:

        B, N = decoder_attn_mask.shape
        
        if self.decoder_sep_mask:
            # separate attention between tokens based on their modality using mod_mask.
            sep_mask = repeat(mod_mask, "b n2 -> b n1 n2", n1=N) != repeat(mod_mask, "b n1 -> b n1 n2", n2=N)
            adapted_attn_mask = decoder_attn_mask | sep_mask
        else:
            adapted_attn_mask = decoder_attn_mask

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

    # TO DO: Change to logits loss for detokenizer 
    def forward_loss(self, 
        y: torch.Tensor, target_gts: torch.Tensor, decoder_mod_dict: Dict[str, Any], decoder_mod_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        mod_loss, mod_n_examples, mod_preds, mod_targets = {}, {}, {}, {}
        for mod, d in decoder_mod_dict.items():
            preds = y[decoder_mod_mask[mod]]
            targets = target_gts[decoder_mod_mask[mod]]
            loss = (self.loss_mod[mod](preds, targets) * decoder_mod_dict[mod]['targets_mask']).sum()
            n_examples = decoder_mod_mask[mod].sum()
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
            
            if mod == 'behavior':
                mod_dict[mod]['inputs'] = mod_dict[mod]['inputs'].unsqueeze(-1)
                
            B, N, D = mod_dict[mod]['inputs'].size()
            if self.mask:
                if mod == 'ap':
                    _, mask = self.masker(mod_dict[mod]['inputs'], mod_dict[mod]['inputs_regions'])
                elif mod == 'behavior':
                    _, mask = self.masker(mod_dict[mod]['inputs'], None)
                # NOTE: use token-wise mask for now and extend later
                mask = mask[:,:,0] & mod_dict[mod]['inputs_mask']
            else:
                mask = torch.zeros_like(mod_dict[mod]['inputs_mask']).to(torch.int64).to(mod_dict[mod]['inputs_mask'].device)
            mod_dict[mod]['inputs_mask'] = 1 - mask
            mod_dict[mod]['targets_mask'] = mask

            context_mask = create_context_mask(self.context_forward, self.context_backward, self.max_F)
            context_mask = context_mask.to(mod_dict[mod]['inputs'].device, torch.int64)
            self_mask = torch.eye(N).to(mod_dict[mod]['inputs'].device, torch.int64).expand(B,N,N) 
            
            inputs_mask = mod_dict[mod]['inputs_mask'].unsqueeze(1).expand(B,N,N)
            mod_dict[mod]['encoder_attn_mask'] = self_mask | (context_mask & inputs_mask)

            targets_mask = mod_dict[mod]['targets_mask'].unsqueeze(1).expand(B,N,N)
            mod_dict[mod]['decoder_attn_mask'] = self_mask | (context_mask & targets_mask)

        
        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}

        encoder_tokens, encoder_emb, encoder_attn_mask, encoder_mod_mask, inputs_mask = self.forward_mask_encoder(encoder_mod_dict)

        decoder_mod_dict = {mod: self.decoder_embeddings[mod].forward_embed(d)
                            for mod, d in mod_dict.items()
                            if mod in self.decoder_embeddings}

        decoder_tokens, decoder_emb, decoder_attn_mask, decoder_mod_mask, targets_mask, target_gts = self.forward_mask_decoder(decoder_mod_dict)

        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_attn_mask=encoder_attn_mask)

        context = self.decoder_proj_context(x) + encoder_emb
        y = decoder_tokens + decoder_emb
        y = self.forward_decoder(y, context, encoder_attn_mask=encoder_attn_mask, decoder_attention_mask=decoder_attn_mask)

        loss, mod_loss, mod_n_examples, mod_preds, mod_targets = self.forward_loss(y, target_gts, decoder_mod_dict, decoder_mod_mask, targets_mask)

        return MultiModalOutput(
            loss=loss,
            mod_loss=mod_loss,
            mod_n_examples=mod_n_examples,
            mod_preds=mod_preds,
            mod_targets=mod_targets
        )

    
