import os
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput
from models.multi_modal.encoder import EncoderEmbedding, Encoder
from models.multi_modal.decoder import DecoderEmbedding, Decoder

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

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
        modality_info:      Dict[str, Any],
        share_modality_embeddings: bool = True,
        use_session: bool = False,
        config: DictConfig,
        **kwargs
    ):
        super().__init__()
        
        self.modality_info = modality_info
        self.share_modality_embeddings = share_modality_embeddings
        self.use_session = use_session
        self.use_prompt = config.use_prompt

        self.encoder_modalities = set(encoder_embeddings.keys())
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        self.decoder_modalities = set(decoder_embeddings.keys())
        self.decoder_embeddings = nn.ModuleDict(decoder_embeddings)

        if share_modality_embeddings:
            self.share_modality_embeddings()

        # TO DO
        self.encoder = Encoder(config.encoder, **kwargs)
        self.decoder = Decoder(config.decoder, **kwargs)

        # TO DO
        if self.use_prompt:
            pass

        # TO DO
        if self.use_session:
            pass
        
    def share_modality_embeddings(self):
        shared_modalities = self.encoder_modalities & self.decoder_modalities
        for mod in shared_modalities:
            self.decoder_embeddings[mod].mod_emb = self.encoder_embeddings[mod].mod_emb

    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        encoder_tokens_all = []
        emb_all = []
        encoder_mask_all = []
        mod_mask_all = []

        for mod, d in mod_dict.items():
            encoder_tokens_all.append(d['x'])
            emb_all.append(d['emb'])
            encoder_mask_all.append(d['inputs_mask'])
            mod_mask_all.append(torch.full_like(d['inputs_mask'], self.modality_info[mod]['id'], dtype=torch.int16))

        encoder_tokens_all = torch.cat(encoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        encoder_mask_all = torch.cat(encoder_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)

        return encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all

    
    def cat_decoder_tensors(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        decoder_tokens_all = []
        target_gts_all = []
        emb_all = []
        decoder_mask_all = []
        attention_mask_all = []
        mod_mask_all = []

        # Shuffle order in which modalities are provided (useful for modality causal mask)
        mod_dict = {mod: d for mod, d in random.sample(mod_dict.items(), len(mod_dict))}

        for mod, d in mod_dict.items():
            decoder_tokens_all.append(d['x'])
            target_gts_all.append(d['gt']) 
            emb_all.append(d['emb'])
            decoder_mask_all.append(d['targets_mask'])
            attention_mask_all.append(d['decoder_attention_mask'])
            mod_mask_all.append(torch.full_like(d['gt'], self.modality_info[mod]['id'], dtype=torch.int16))

        decoder_tokens_all = torch.cat(decoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        decoder_mask_all = torch.cat(decoder_mask_all, dim=1)
        target_gts_all = torch.cat(target_ids_all, dim=1)
        attention_mask_all = torch.cat(attention_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)

        return decoder_tokens_all, emb_all, decoder_mask_all, target_gts_all, attention_mask_all, mod_mask_all

    
    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_encoder_tokens: int) -> Tuple[torch.Tensor]:
        encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all = self.cat_encoder_tensors(mod_dict)

        # TO DO: Prepend tokens
        
        return encoder_tokens, encoder_emb, encoder_mask, mod_mask

    def forward_mask_decoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_decoder_tokens: int) -> Tuple[torch.Tensor]:
        decoder_tokens_all, emb_all, decoder_mask_all, target_gts_all, decoder_attention_mask_all, mod_mask_all = self.cat_decoder_tensors(mod_dict)

        # TO DO: Prepend tokens
        
        return decoder_tokens, decoder_emb, decoder_mask, target_gts, decoder_attention_mask, mod_mask
        

    def adapt_decoder_attention_mask(self, decoder_attention_mask: torch.Tensor, mod_mask=Optional[torch.Tensor]) -> torch.Tensor:
        pass

    def forward_encoder(self, x: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        
        for blk in self.encoder:
            x = blk(x, mask=encoder_mask)

        return x

    def forward_decoder(self, y: torch.Tensor, context: torch.Tensor, encoder_mask: torch.Tensor, decoder_attention_mask: torch.Tensor) -> torch.Tensor:

        for blk in self.decoder:
            y = blk(y, context, sa_mask=decoder_attention_mask, xa_mask=encoder_mask)

        return y

    def forward_mod_loss(self, 
         y: torch.Tensor, 
         target_ids: torch.Tensor, 
         decoder_mod_dict: Dict[str, Any], 
         decoder_mod_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def forward_token_loss(self, 
       y: torch.Tensor, 
       target_ids: torch.Tensor, 
       decoder_mod_dict: Dict[str, Any], 
       decoder_mod_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def forward_loss(self, 
                     y: torch.Tensor, 
                     target_gts: torch.Tensor, 
                     decoder_mod_dict: Dict[str, Any], 
                     decoder_mod_mask: torch.Tensor, loss_type: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_type in ['mod', 'modality']:
            loss, mod_loss = self.forward_mod_loss(y, target_gts, decoder_mod_dict, decoder_mod_mask)
        elif loss_type == 'token':
            loss, mod_loss = self.forward_token_loss(y, target_gts, decoder_mod_dict, decoder_mod_mask)
        else:
            raise ValueError("Invalid loss type")

        return loss, mod_loss

    def forward(self, 
        mod_dict: Dict[str, Dict[str, torch.Tensor]], 
        num_encoder_tokens: int, 
        num_decoder_tokens: int, 
        loss_type: str = 'mod', 
        return_logits: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict, num_encoder_tokens)

        decoder_mod_dict = {mod: self.decoder_embeddings[mod].forward_embed(d)
                            for mod, d in mod_dict.items()
                            if mod in self.decoder_embeddings}
        decoder_tokens, decoder_emb, decoder_mask, target_ids, decoder_attention_mask, decoder_mod_mask = self.forward_mask_decoder(decoder_mod_dict, num_decoder_tokens)

        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_mask=encoder_mask)

        # TO DO: Do we need to project the tokens again? 
        context = self.decoder_proj_context(x) + encoder_emb
        y = decoder_tokens + decoder_emb
        y = self.forward_decoder(y, context, encoder_mask=encoder_mask, decoder_attention_mask=decoder_attention_mask)

        loss, mod_loss = self.forward_loss(y, target_gts, decoder_mod_dict, decoder_mod_mask, loss_type)

        return loss, mod_loss


    def freeze_encoder(self, freeze_embeddings=True):
        pass

    def save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, load_dir):
        pass




    
