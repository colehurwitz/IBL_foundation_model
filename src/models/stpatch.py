import os
import numpy as np
from math import ceil, floor
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Optional, Tuple, Dict
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from models.masker import Masker
from models.patcher import Patcher
from models.region_lookup import RegionLookup 

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput

DEFAULT_CONFIG = "src/configs/stpatch.yaml"

with open('data/target_eids.txt') as file:
    include_eids = [line.rstrip() for line in file]

@dataclass
class STPatchOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None
    num_neuron: Optional[int] = None
  

# Create buffer of biggest possible context mask 
def create_context_mask(
    context_forward, context_backward, max_space_F, max_time_F
) -> torch.LongTensor: # (max_n_token, max_n_token)

    # bidirectional
    if context_forward == -1 and context_backward == -1:
        return torch.ones(max_time_F*max_space_F, max_time_F*max_space_F).to(torch.int64)

    context_forward = context_forward if context_forward >= 0 else max_time_F
    context_backward = context_backward if context_backward >= 0 else max_time_F
    mask = (torch.triu(torch.ones(max_time_F, max_time_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
    if context_backward > 0:
        back_mask = (torch.triu(torch.ones(max_time_F, max_time_F), diagonal=-context_backward).to(torch.int64))
        mask = mask & back_mask

    # (max_seq_len, max_seq_len) -> (max_n_token, max_n_token) 
    mask = mask.repeat(max_space_F, max_space_F)
    return mask


class NeuralEmbeddingLayer(nn.Module):

    def __init__(
        self, hidden_size, embed_region, max_region_indx, config: DictConfig, **kwargs
    ):
        super().__init__()

        self.bias = config.bias

        self.n_neurons = kwargs['max_space_length']
        self.n_timesteps = config.n_timesteps
        self.max_space_F = config.max_space_F
        self.max_time_F = config.max_time_F

        self.n_space_patches = ceil(self.n_neurons/self.max_space_F)
        self.n_time_patches = ceil(self.n_timesteps/self.max_time_F)
        self.n_channels = self.max_time_F * self.max_space_F

        self.input_dim = self.n_channels*config.mult
        self.mode = config.mode
       
        if config.mode == "linear":
            self.embed_spikes = nn.Linear(self.n_channels, self.input_dim, bias=config.bias)
        elif config.mode == "identity":
            self.embed_spikes = nn.Identity()
        else:
            raise Exception(f"Invalid embed mode {config.mode}.")

        self.projection = nn.Linear(self.input_dim, hidden_size)
        
        # activation after embedding 
        self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()

        # embedding scale
        self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

        # embed patch postion
        self.embed_region = embed_region
        if self.embed_region:
            self.region_embeddings = nn.Sequential(
                nn.Embedding(
                    max_region_indx, hidden_size),
                nn.LayerNorm(hidden_size),
            )

        self.embed_space_pos = nn.Sequential(
            nn.Embedding(
                self.n_space_patches, hidden_size
            ),
            nn.LayerNorm(hidden_size),
        )
            
        self.embed_time_pos = nn.Sequential(
            nn.Embedding(
                self.n_time_patches, hidden_size
            ),
            nn.LayerNorm(hidden_size),
        )

        # Embed prompt token
        self.use_prompt = config.use_prompt
        if self.use_prompt:
            self.mask_types = ['neuron', 'causal', 'inter-region', 'intra-region']
            self.mask_to_indx = {r: i for i,r in enumerate(self.mask_types)}
            self.embed_prompt = nn.Embedding(len(self.mask_types), hidden_size) 

        self.use_session = config.use_session
        if self.use_session:
            self.eid_lookup = include_eids
            self.eid_to_indx = {r: i for i,r in enumerate(self.eid_lookup)}
            self.embed_session = nn.Embedding(len(self.eid_lookup), hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        spikes:               torch.FloatTensor,                   # (n_batch, n_token, n_channels)
        space_attn_mask:      torch.LongTensor,
        time_attn_mask:       torch.LongTensor,
        spacestamps:          torch.LongTensor,
        timestamps:           torch.LongTensor,
        regionstamps:         Optional[torch.LongTensor] = None,
        masking_mode:         Optional[str] = None,
        eid:                  Optional[str] = None,
    ) -> Tuple[torch.FloatTensor,torch.LongTensor,torch.LongTensor]:  
        #   (n_batch, new_n_token, hidden_size) ->
        #   (n_batch, new_n_token), (n_batch, new_n_token), (n_batch, new_n_token), (n_batch, new_n_token)

        B, T, N = spikes.size()
        
        # Embed spikes
        x = self.embed_spikes(spikes)
        
        # Rescaling
        x = self.act(x) * self.scale

        x = self.projection(x)
       
        # Embed patch position
        x += self.embed_space_pos(spacestamps)
        x += self.embed_time_pos(timestamps)

        if self.embed_region:
            region_embeds = self.region_embeddings(regionstamps)  
            x += region_embeds

        # Prepend prompt token 
        if self.use_prompt:
            mask_idx = torch.tensor(self.mask_to_indx[masking_mode], dtype=torch.int64, device=spikes.device)
            x = torch.cat((self.embed_prompt(mask_idx)[None,None,:].expand(B,-1,-1), x), dim=1) 
            space_attn_mask = F.pad(space_attn_mask, (1, 0), value=1)
            time_attn_mask = F.pad(time_attn_mask, (1, 0), value=1)
            spacestamps = torch.cat(
                (torch.zeros((spacestamps.size(0), 1), dtype=spacestamps.dtype, device=spacestamps.device), 
                 spacestamps+1), dim=1
            )
            timestamps = torch.cat(
                (torch.zeros((timestamps.size(0), 1), dtype=timestamps.dtype, device=timestamps.device), 
                 timestamps+1), dim=1
            )

        if self.use_session:
            session_idx = torch.tensor(self.eid_to_indx[eid], dtype=torch.int64, device=spikes.device)
            x = torch.cat((self.embed_session(session_idx)[None,None,:].expand(B,-1,-1), x), dim=1)
            space_attn_mask = F.pad(space_attn_mask, (1, 0), value=1)
            time_attn_mask = F.pad(time_attn_mask, (1, 0), value=1)
            spacestamps = torch.cat(
                (torch.zeros((spacestamps.size(0), 1), dtype=spacestamps.dtype, device=spacestamps.device), 
                 spacestamps+1), dim=1
            )
            timestamps = torch.cat(
                (torch.zeros((timestamps.size(0), 1), dtype=timestamps.dtype, device=timestamps.device), 
                 timestamps+1), dim=1
            )
            
        return self.dropout(x), space_attn_mask, time_attn_mask, spacestamps, timestamps


class NeuralMLP(nn.Module):
    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()

        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))


class NeuralAttention(nn.Module):
    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout):
        super().__init__()

        self.idx = idx

        # Architecture config
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, f"Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads

        # Attention parameters
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        # Flash attention
        # torch.backends.cuda.enable_flash_sdp(True)
        self.attn_dropout = dropout

        # Final projection
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

    def forward(
        self,       
        x:                    torch.FloatTensor,                      # (n_batch, n_token, hidden_size)
        attn_mask:            torch.LongTensor,                       # (n_batch, n_token, n_token)
    ) -> torch.FloatTensor:                                           # (n_batch, n_token, hidden_size)

        B, T, _  = x.size()     

        # create batched bool attention mask 
        # TODO: assert attn_mask?
        # assert attn_mask.max() == 1 and attn_mask.min() == 0, ["assertion", attn_mask.max(), attn_mask.min()]
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, T, T).bool()    # (n_batch, n_heads, n_token, n_token)
        
        # compute query, key, value for attention
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (n_batch, n_heads, n_token, head_size)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)   # (n_batch, n_heads, n_token, head_size)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (n_batch, n_heads, n_token, head_size)

        # compute attention efficiently
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.attn_dropout if self.training else 0.0), is_causal=False) # (n_batch, n_heads, n_token, head_size)
        out = out.transpose(1, 2).contiguous().view(B,T, self.hidden_size)         # (n_batch, n_token, hidden_size)

        return self.out_proj(self.dropout(out))                                    # (n_batch, n_token, hidden_size)


# Encoder layer: bidirectional self-attention + mlp
class NeuralEncoderLayer(nn.Module):
    def __init__(self, idx, max_F, config: DictConfig):
        super().__init__()

        self.idx = idx
    
        # Encoder block
        self.ln1 = nn.LayerNorm(config.hidden_size) 
        self.attn = NeuralAttention(
            idx, config.hidden_size, 
            config.n_heads, 
            config.attention_bias, 
            config.dropout
        )
        self.ln2 = nn.LayerNorm(config.hidden_size) 
        self.mlp = NeuralMLP(
            config.hidden_size, 
            config.inter_size, 
            config.act, 
            config.mlp_bias, 
            config.dropout
        )

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:          torch.FloatTensor,                            # (n_batch, n_token, hidden_size)
        attn_mask:  torch.LongTensor,                             # (n_batch, n_token, n_token)
    ) -> torch.FloatTensor :                                      # (n_batch, n_token, hidden_size)
        
        # LN -> Attention -> Residual connection
        x = x + self.attn(self.ln1(x), attn_mask)

        # LN -> MLP -> Residual connection
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



class SpaceTimeTransformer(nn.Module):
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__() 

        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers

        self.max_time_F = config.embedder.max_time_F
        self.max_space_F = config.embedder.max_space_F

        self.n_space_patches = ceil(kwargs['max_space_length']/self.max_space_F)
        self.n_time_patches = ceil(config.embedder.n_timesteps/self.max_time_F)

        self.n_channels = self.max_time_F * self.max_space_F

        # Mask
        self.mask = config.masker.force_active
        self.mask_token = False
        if config.masker.mode == 'random_token':
            self.mask = False
            self.mask_token = True

        if self.mask | self.mask_token:
            self.masker = Masker(config.masker)

        self.embed_region = config.embed_region
        self.regionlookup = RegionLookup(config)

        # Patcher
        self.patch = config.patcher.active
        if self.patch:
            self.patcher = Patcher(
                self.max_space_F, self.max_time_F, self.embed_region, config.patcher
            )

        # Context span mask
        self.context_forward = config.context.forward
        self.context_backward = config.context.backward
        # context_mask = create_context_mask(
        #     config.context.forward, config.context.backward, 
        #     self.n_time_patches, self.n_time_patches
        # )
        # self.register_buffer("context_mask", context_mask, persistent=False)

        self.use_prompt = config.embedder.use_prompt
        self.use_session = config.embedder.use_session

        # Embedding layer
        self.embedder = NeuralEmbeddingLayer(
            self.hidden_size, self.embed_region, self.regionlookup.max_region_indx, 
            config.embedder, **kwargs
        )

        # Transformer
        self.layers = nn.ModuleList(
            [NeuralEncoderLayer(
                idx, self.n_time_patches*self.n_space_patches, config.transformer
            ) for idx in range(self.n_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size) 

    def forward(
            self, 
            spikes:               torch.FloatTensor,  # (bs, seq_len, n_channels)
            pad_space_len:        int,   # (bs,)
            pad_time_len:         int,
            time_attn_mask:       torch.LongTensor,   # (bs, seq_len)
            space_attn_mask:      torch.LongTensor,   # (bs, seq_len)
            spikes_timestamp:     torch.LongTensor,   # (bs, seq_len)
            spikes_spacestamp:    torch.LongTensor,   # (bs, seq_len)
            neuron_regions:       Optional[np.ndarray] = None,  # (bs, n_channels)
            masking_mode:         Optional[str] = None,
            eval_mask:        Optional[torch.LongTensor] = None,
            num_neuron:       Optional[torch.LongTensor] = None,
            eid:              Optional[str] = None,
    ) -> torch.FloatTensor:                           # (seq_len, seq_len, hidden_size)

        B, T, N = spikes.size() 

        if masking_mode == 'causal':
            self.masker.mode = 'temporal'
            self.context_forward = 0 
            self.context_mask = create_context_mask(
                self.context_forward, self.context_backward, 
                self.n_time_patches, self.n_time_patches
            )
        else:
            self.masker.mode = masking_mode
            self.context_forward = -1
            self.context_mask = create_context_mask(
                self.context_forward, self.context_backward, 
                self.n_time_patches, self.n_time_patches
            )
        
        # Mask spikes
        if self.mask:
            spikes, targets_mask = self.masker(spikes, neuron_regions)
        else:
            targets_mask = None

        if self.mask:
            targets_mask = targets_mask.to(torch.int64) & time_attn_mask.unsqueeze(-1).expand(B,T,N).to(torch.int64)
            targets_mask = targets_mask.to(torch.int64) & space_attn_mask.unsqueeze(1).expand(B,T,N).to(torch.int64)

        if self.embed_region:
            region_indx = self.regionlookup(neuron_regions).to(spikes.device)
        else:
            region_indx = None

        # Patch neural data
        if self.patch:
            spikes, _space_attn_mask, _time_attn_mask, spacestamps, timestamps, regionstamps  =\
            self.patcher(spikes, pad_space_len, pad_time_len, time_attn_mask, space_attn_mask, region_indx)

        B, _T, N = spikes.size()
        
        # Mask tokens
        if self.mask_token:
            spikes, targets_mask = self.masker(spikes)
            targets_mask = targets_mask.reshape(B,T,N).to(torch.int64) & time_attn_mask.unsqueeze(-1).expand(B,T,N)
            targets_mask = targets_mask.reshape(B,T,N).to(torch.int64) & space_attn_mask.unsqueeze(1).expand(B,T,N)

        if eval_mask is not None:
            targets_mask = eval_mask.clone()
            
        # Embed neural data
        x, _space_attn_mask, _time_attn_mask, spacestamps, timestamps = self.embedder(
            spikes, _space_attn_mask, _time_attn_mask, spacestamps, timestamps, regionstamps, 
            masking_mode, eid
        )

        _, T, _ = x.size() 

        if self.use_prompt or self.use_session:
            context_mask = torch.cat((torch.ones((_T,T-_T)), self.context_mask[:_T,:_T]), dim=1)
            context_mask = torch.cat((torch.ones((T-_T,T)), context_mask), dim=0)
            context_mask = context_mask.to(x.device, torch.int64).unsqueeze(0).expand(B,T,T)
        else:
            context_mask = self.context_mask[:T,:T].to(x.device, torch.int64).unsqueeze(0).expand(B,T,T)

        space_attn_mask = _space_attn_mask.unsqueeze(1).expand(B,T,T)
        time_attn_mask = _time_attn_mask.unsqueeze(1).expand(B,T,T)
        
        attn_mask = torch.eye(T).to(x.device, torch.int64).expand(B,T,T) 
        # hack so that even padded spikes attend to themselves and avoid attention issues

        attn_mask = attn_mask | (context_mask & space_attn_mask & time_attn_mask)

        # Forward transformer
        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
        x = self.out_norm(x)

        return x, targets_mask, _T



class STPatch(nn.Module):
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)
        self.method = kwargs["method_name"]

        self.pad_value = -1.
        
        # Build encoder
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = os.path.join(encoder_pt_path, "encoder_config.yaml")
            config["encoder"] = update_config(config.encoder, encoder_config)
        self.encoder = SpaceTimeTransformer(config.encoder, **kwargs)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))

        self.use_prompt = config.encoder.embedder.use_prompt
        self.use_session = config.encoder.embedder.use_session

        # Build decoder
        if self.method == "ssl":
            assert config.encoder.masker.force_active, "Can't pretrain with inactive masking"
            n_outputs = config.encoder.embedder.max_time_F * config.encoder.embedder.max_space_F
        elif self.method == "sl":
            n_outputs = kwargs["output_size"]
        else:
            raise Exception(f"Method {self.method} not implemented yet for NDT2")

        decoder_layers = []
        if self.method == "sl":
            decoder_layers.append(
                nn.Linear(
                    (self.encoder.n_time_patches * self.encoder.n_space_patches) * self.encoder.hidden_size, n_outputs
                )
            )
        else:
            decoder_layers.append(nn.Linear(self.encoder.hidden_size, n_outputs))

        if self.method == "sft" and not kwargs["use_lograte"]:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        if self.method == "sl":
            if kwargs["clf"]:
                pass  # cross-entropy loss uses logits as inputs
            elif kwargs["reg"]:
                pass
            else:
                raise Exception(f"Decoder not implemented yet for sl")
        
        self.decoder = nn.Sequential(*decoder_layers)

        # Load decoder weights
        if config.decoder.from_pt is not None:
            self.decoder.load_state_dict(torch.load(os.path.join(config.decoder.from_pt,"decoder.bin")))

        # Build loss function
        if self.method == "ssl":
            if kwargs["loss"] == "poisson_nll":
                self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=kwargs["use_lograte"])
            elif kwargs["loss"] == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            else:   
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for ssl")
        elif self.method == "sl":
            if kwargs["loss"] == "cross_entropy":
                self.loss_fn = nn.CrossEntropyLoss(reduction="none")
            elif kwargs["loss"] == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            else:
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for sl")
        

    def forward(
        self, 
        spikes:               torch.FloatTensor,                   # (bs, seq_len, n_channels)
        time_attn_mask:       torch.LongTensor,                    # (bs, seq_len)
        space_attn_mask:      torch.LongTensor,                    # (bs, seq_len)
        spikes_timestamps:    torch.LongTensor,                    # (bs, seq_len)
        spikes_spacestamps:   torch.LongTensor,                    # (bs, seq_len)
        targets:              Optional[torch.FloatTensor] = None,  # (bs, target_len) 
        spikes_lengths:       Optional[torch.LongTensor] = None,   # (bs,)
        targets_lengths:      Optional[torch.LongTensor] = None,   # (bs,)
        neuron_regions:       Optional[torch.LongTensor] = None,   # (bs, n_channels)
        masking_mode:     Optional[str] = None,
        spike_augmentation: Optional[bool] = False,
        eval_mask:        Optional[torch.LongTensor] = None,
        num_neuron:       Optional[torch.LongTensor] = None,
        eid:              Optional[str] = None,
    ) -> STPatchOutput:   

        B, T_, N = spikes.size()

         # Augmentation
        if spike_augmentation:
            if self.training:
                # 50% of the time, we reverse the spikes
                if torch.rand(1) > 0.5:
                    # calculate unmask timestamps
                    unmask_temporal = time_attn_mask.sum(dim=1)
                    for i in range(len(unmask_temporal)):
                        # reverse idx from unmask_temporal to 0
                        reverse_idx = torch.arange(unmask_temporal[i]-1, -1, -1)
                        spikes[i, :unmask_temporal[i]] = spikes[i, reverse_idx]

        # if neuron_regions type is list 
        if isinstance(neuron_regions, list):
            neuron_regions = np.asarray(neuron_regions).T
        
        if self.method == "ssl":
            targets = spikes.clone()

        # Track padded values to prevent them from being used
        if (spikes[0,:,0] == self.pad_value).sum() == 0:
            pad_time_len = T_
        else:
            pad_time_len = (spikes[0,:,0] == self.pad_value).nonzero().min().item() 

        if (spikes[0,0,:] == self.pad_value).sum() == 0:
            pad_space_len = N
        else:
            pad_space_len = (spikes[0,0,:] == self.pad_value).nonzero().min().item() 

        # Encode neural data
        x, targets_mask, _T = self.encoder(
            spikes, pad_space_len, pad_time_len, time_attn_mask, space_attn_mask, spikes_timestamps, spikes_spacestamps, neuron_regions, masking_mode, eval_mask, num_neuron, eid
        )

        _, T, _ = x.size()

        if self.use_prompt or self.use_session:
            x = x[:,T-_T:]

        # Transform neural embeddings into rates/logits
        if self.method == "ssl":
            outputs = self.decoder(x).reshape((B,T_,-1))[:,:,:pad_space_len]
            targets = targets.reshape((B,T_,-1))[:,:,:pad_space_len]
            targets_mask = targets_mask.reshape((B,T_,-1))[:,:,:pad_space_len]
            
        elif self.method == "sl":
            x = x.flatten(start_dim=1)  
            outputs = self.decoder(x)

        # Compute the loss over unmasked outputs
        if self.method == "ssl":
            loss = (self.loss_fn(outputs, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()
        elif self.method == "sl":
            loss = self.loss_fn(outputs, targets).sum()
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)

        return STPatchOutput(
            loss=loss,
            n_examples=n_examples,
            preds=outputs,
            targets=targets,
            num_neuron=pad_space_len
        )  

    def save_checkpoint(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,"encoder.bin"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,"decoder.bin"))

    def load_checkpoint(self, load_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir,"encoder.bin")))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir,"decoder.bin")))
        
