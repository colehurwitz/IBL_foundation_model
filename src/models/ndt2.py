import os
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Optional, Tuple, Dict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput

DEFAULT_CONFIG = "src/configs/ndt2.yaml"

@dataclass
class NDT2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None


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


class Masker(nn.Module):
    '''
    Mask spikes: (1) time steps (2) random patches.
    '''
    def __init__(self, embed_mode, config: DictConfig):
        super().__init__()

        self.mode = config.mode
        self.ratio = config.ratio
        self.zero_ratio = config.zero_ratio
        self.random_ratio = config.random_ratio
        self.embed_mode = embed_mode
        self.n_time_steps = config.n_time_steps

    def forward(
        self, 
        spikes: torch.FloatTensor,                      # (n_batch, n_token, n_channels)
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (n_batch, n_token, n_channels), (n_batch, n_token, n_channels)

        # (n_batch, n_token, n_channels) -> (n_batch, n_time_bins, n_patches, n_channels)
        B, T, N = spikes.shape
        spikes = spikes.reshape(B, self.n_time_steps, T//self.n_time_steps, N)
        
        mask_ratio = deepcopy(self.ratio)

        # get masking probabilities
        if self.mode == "full":
            mask_probs = torch.full(spikes[:,:,:,0].shape, mask_ratio) # (n_batch, n_time_bins, n_patches)
        elif self.mode == "timestep":
            mask_probs = torch.full(spikes[:,:,0,0].shape, mask_ratio) # (n_batch, n_time_bins)
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")

        # create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # expand mask
        if self.mode == "timestep":
            # (n_batch, n_time_bins, n_patches, n_channels)
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(spikes).bool()  
        elif self.mode == "full":
            # (n_batch, n_time_bins, n_patches, n_channels)
            mask = mask.unsqueeze(-1).expand_as(spikes).bool()                

        # mask data
        zero_idx = torch.bernoulli(torch.full(spikes.shape, self.zero_ratio)).to(spikes.device).bool() & mask
        spikes[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(spikes.shape, self.random_ratio)).to(spikes.device).bool() & mask & ~zero_idx
        random_spikes = (spikes.max() * torch.rand(spikes.shape, device=spikes.device) )
        if self.embed_mode == "identity":
            random_spikes = random_spikes.round()
        else:
            random_spikes = random_spikes.float()

        spikes[random_idx] = random_spikes[random_idx]

        # (n_batch, n_time_bins, n_patches, n_channels) -> (n_batch, n_token, n_channels) 
        spikes, mask = spikes.flatten(1,-2), mask.flatten(1,-2)

        return spikes, mask.to(torch.int64)


class NeuralEmbeddingLayer(nn.Module):

    def __init__(self, hidden_size, config: DictConfig):
        super().__init__()

        self.bias = config.bias
        # NDT1,: n_channels = n_neurons vs. NDT2: n_channels = n_neurons_per_patch
        self.input_dim = config.n_channels*config.mult

        if config.mode == "linear":
            self.embed_spikes = nn.Linear(config.n_channels, self.input_dim, bias=config.bias)
        elif config.mode == "identity":
            self.embed_spikes = nn.Identity()
        else:
            raise Exception(f"Invalid embed mode {config.mode}.")

        self.projection = nn.Linear(self.input_dim, hidden_size)

        # activation after embedding
        self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()

        # embedding scale
        self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

        # embed space postion
        self.space_pos = config.space_pos
        if self.space_pos:
            self.embed_space_pos = nn.Embedding(config.max_time_F*config.max_space_F, hidden_size)

        # embed time postion
        self.time_pos = config.time_pos
        if self.time_pos:
            self.embed_time_pos = nn.Embedding(config.max_time_F*config.max_space_F, hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        spikes:               torch.FloatTensor,                   # (n_batch, n_token, n_channels)
        spikes_space_mask:    Optional[torch.LongTensor],          # (n_batch, n_token)
        spikes_time_mask:     Optional[torch.LongTensor],          # (n_batch, n_token)
        spikes_spacestamp:    Optional[torch.LongTensor],          # (n_batch, n_token)
        spikes_timestamp:     Optional[torch.LongTensor],          # (n_batch, n_token)
    ) -> Tuple[torch.FloatTensor,torch.LongTensor,torch.LongTensor]:  
        #   (n_batch, new_n_token, hidden_size) ->
        #   (n_batch, new_n_token), (n_batch, new_n_token), (n_batch, new_n_token), (n_batch, new_n_token)

        # Embed spikes
        x = self.embed_spikes(spikes)

        # Rescaling
        x = self.act(x) * self.scale

        x = self.projection(x)

        # Embed space position
        if self.space_pos:
            x += self.embed_space_pos(spikes_spacestamp)

        # Embed time position
        if self.time_pos:
            x += self.embed_time_pos(spikes_timestamp)

        return self.dropout(x), spikes_space_mask, spikes_time_mask, spikes_spacestamp, spikes_timestamp


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

        self.use_space = config.transformer.use_space
        self.use_time = config.transformer.use_time

        # Masker
        self.mask = config.masker.active
        if self.mask:
            self.masker = Masker(config.embedder.mode, config.masker)

        # Context span mask
        context_mask = create_context_mask(
            config.context.forward, config.context.backward, config.embedder.max_space_F, config.embedder.max_time_F
        )
        self.register_buffer("context_mask", context_mask, persistent=False)

        # Embedding layer
        self.embedder = NeuralEmbeddingLayer(self.hidden_size, config.embedder)

        # Transformer
        self.layers = nn.ModuleList(
            [NeuralEncoderLayer(
                idx, config.embedder.max_time_F*config.embedder.max_space_F, config.transformer
            ) for idx in range(self.n_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size) 

    def forward(
            self, 
            spikes:                  torch.FloatTensor,  # (n_batch, n_token, n_channels)
            spikes_space_mask:       torch.LongTensor,   # (n_batch, n_token)
            spikes_time_mask:        torch.LongTensor,   # (n_batch, n_token)
            spikes_spacestamp:       torch.LongTensor,   # (n_batch, n_token)
            spikes_timestamp:        torch.LongTensor,   # (n_batch, n_token)
    ) -> torch.FloatTensor:                              # (n_batch, n_token, hidden_size)

        B, T, N = spikes.size() # n_batch, n_token, n_channels

        # Mask neural data
        if self.mask:
            spikes, targets_mask = self.masker(spikes)
            if self.use_space:
                targets_mask = targets_mask & spikes_space_mask.unsqueeze(-1).expand(B,T,N)
            if self.use_time:
                targets_mask = targets_mask & spikes_time_mask.unsqueeze(-1).expand(B,T,N)
        else:
            targets_mask = None

        # Embed neural data
        x, spikes_space_mask, spikes_time_mask, spikes_spacestamp, spikes_timestamp = \
        self.embedder(spikes, spikes_space_mask, spikes_time_mask, spikes_spacestamp, spikes_timestamp)

        _, T, _ = x.size() 

        context_mask = self.context_mask[:T,:T].to(x.device).unsqueeze(0).expand(B,T,T)
        if self.use_space:
            spikes_space_mask = spikes_space_mask.unsqueeze(-1).expand(B,T,T)
        if self.use_time:
            spikes_time_mask = spikes_time_mask.unsqueeze(-1).expand(B,T,T)
        attn_mask = torch.eye(T).to(x.device, torch.int64).expand(B,T,T) 
        # hack so that even padded spikes attend to themselves and avoid attention issues

        if self.use_space & self.use_time:
            attn_mask = attn_mask | (context_mask & spikes_space_mask & spikes_time_mask)
        else:
            if self.use_space:
                attn_mask = attn_mask | (context_mask & spikes_space_mask)
            elif self.use_time:
                attn_mask = attn_mask | (context_mask & spikes_time_mask)

        # Forward transformer
        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
        x = self.out_norm(x)

        return x, targets_mask
        


class NDT2(nn.Module):
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)
        self.method = kwargs["method_name"]
        
        # Build encoder
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = os.path.join(encoder_pt_path, "encoder_config.yaml")
            config["encoder"] = update_config(config.encoder, encoder_config)
        self.encoder = SpaceTimeTransformer(config.encoder)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))

        # Build decoder
        if self.method == "ssl":
            assert config.encoder.masker.active, "Can't pretrain with inactive masking"
            n_outputs = config.encoder.embedder.n_channels
        else:
            raise Exception(f"Method {self.method} not implemented yet for NDT2")

        decoder_layers = []
        decoder_layers.append(nn.Linear(self.encoder.hidden_size, n_outputs))

        if self.method == "sft" and not kwargs["use_lograte"]:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        self.decoder = nn.Sequential(*decoder_layers)

        # Load decoder weights
        if config.decoder.from_pt is not None:
            self.decoder.load_state_dict(torch.load(os.path.join(config.decoder.from_pt,"decoder.bin")))

        # Build loss function
        if kwargs["loss"] == "poisson_nll":
            self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=kwargs["use_lograte"])
        elif kwargs["loss"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:   
            raise Exception(f"Loss {kwargs['loss']} not implemented yet for ssl")

    def forward(
        self, 
        spikes:            torch.FloatTensor,                   # (n_batch, n_token, n_channels)
        spikes_space_mask: torch.LongTensor,                    # (n_batch, n_token)
        spikes_time_mask:  torch.LongTensor,                    # (n_batch, n_token)
        spikes_spacestamp: torch.LongTensor,                    # (n_batch, n_token)
        spikes_timestamp:  torch.LongTensor,                    # (n_batch, n_token)
        spikes_lengths:    Optional[torch.LongTensor] = None,   # (n_batch)
        targets:           Optional[torch.FloatTensor] = None,  # (n_batch, target_len) 
        targets_lengths:   Optional[torch.LongTensor] = None,   # (n_batch)
    ) -> NDT2Output:   

        if self.method == "ssl":
            targets = spikes.clone()

        # Encode neural data
        x, targets_mask = self.encoder(
            spikes, spikes_space_mask, spikes_time_mask, spikes_spacestamp, spikes_timestamp, 
        )

        # Transform neural embeddings into rates/logits
        outputs = self.decoder(x)

        # Compute the loss over unmasked outputs
        if self.method == "ssl":
            loss = (self.loss_fn(outputs, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()

        return NDT2Output(
            loss=loss,
            n_examples=n_examples,
            preds=outputs,
        )  

    def save_checkpoint(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,"encoder.bin"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,"decoder.bin"))

    def load_checkpoint(self, load_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir,"encoder.bin")))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir,"decoder.bin")))


    
    