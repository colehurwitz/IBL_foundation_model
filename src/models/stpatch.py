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

# DEFAULT_CONFIG = "src/configs/stpatch.yaml"
DEFAULT_CONFIG = "/home/yizi/IBL_foundation_model/src/configs/stpatch.yaml"

@dataclass
class STPatchOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None


# Create buffer of biggest possible context mask 
def create_context_mask():
    pass


class Masker(nn.Module):
    '''
    Mask spikes: random patches.
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

        # (n_batch, n_token, n_channels) -> (n_batch, n_token, n_channels) 
        B, T, N = spikes.shape
        
        mask_ratio = deepcopy(self.ratio)

        # get masking probabilities
        if self.mode == "full":
            mask_probs = torch.full(spikes[:,:,0].shape, mask_ratio) # (n_batch, n_token,)
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")

        # create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # expand mask
        if self.mode == "full":
            # (n_batch, n_token, n_channels)
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

        return spikes, mask.to(torch.int64)


class NeuralEmbeddingLayer(nn.Module):

    def __init__(self, hidden_size, config: DictConfig):
        super().__init__()

        self.bias = config.bias

        self.n_neurons = config.n_neurons
        self.n_timesteps = config.n_timesteps
        self.max_space_F = config.max_space_F
        self.max_time_F = config.max_time_F
        
        self.n_patches = (self.n_timesteps//self.max_time_F) * (self.n_neurons//self.max_space_F)
        self.n_channels = self.max_time_F * self.max_space_F

        self.input_dim = self.n_channels*config.mult
        self.mode = config.mode
       
        if config.mode == "linear":
            self.embed_spikes = nn.Linear(self.n_channels, self.input_dim, bias=config.bias)
        elif config.mode == "identity":
            self.embed_spikes = nn.Identity()
        else:
            raise Exception(f"Invalid embed mode {config.mode}.")

        # initiate [cls] tokens for downstream tasks
        self.n_cls_tokens = config.n_cls_tokens
        if self.n_cls_tokens != 0:
            self.cls_tokens = nn.Parameter(torch.zeros(1, self.n_cls_tokens, self.input_dim))

        self.projection = nn.Linear(self.input_dim, hidden_size)

        # activation after embedding
        self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()

        # embedding scale
        self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

        # embed patch postion
        self.embed_pos = nn.Embedding(self.n_patches+self.n_cls_tokens, hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        spikes:               torch.FloatTensor,                   # (n_batch, n_token, n_channels)
        spikes_mask:      Optional[torch.LongTensor] = None,          # (bs, seq_len)
        spikes_timestamp: Optional[torch.LongTensor] = None,          # (bs, seq_len)
    ) -> Tuple[torch.FloatTensor,torch.LongTensor,torch.LongTensor]:  
        #   (n_batch, new_n_token, hidden_size) ->
        #   (n_batch, new_n_token), (n_batch, new_n_token), (n_batch, new_n_token), (n_batch, new_n_token)

        B, T, N = spikes.size()

        # group neurons into patches
        patches = torch.ones(
            (B, T//self.max_time_F, N//self.max_space_F, self.n_channels)
        ) * -1.    
        for t in range(T//self.max_time_F):
            for s in range(N//self.max_space_F):
                patches[:,t,s] = spikes[:,t*self.max_time_F:(t+1)*self.max_time_F, s*self.max_space_F:(s+1)*self.max_space_F].flatten(1)
        patches = patches.flatten(1,2).to(spikes.device)
        
        # Embed spikes
        x = self.embed_spikes(patches)

        # Prepend [cls] tokens for downstream tasks
        if self.n_cls_tokens != 0:
            cls_tokens = self.cls_tokens.expand(B, -1, -1)  
            x = torch.cat((cls_tokens, x), dim=1)

        # Rescaling
        x = self.act(x) * self.scale

        x = self.projection(x)

        # Prepend space and time stamps for [cls] tokens
        spikes_patchstamp = torch.arange(self.n_patches+self.n_cls_tokens).to(torch.int64)[None,:]
        spikes_patchstamp = spikes_patchstamp.expand(B, -1).to(spikes.device)

        # Prepend space and time masks for [cls] tokens
        # TO DO: There may be better ways to handle the cls space & time masks
        spikes_patchmask = torch.ones((B, self.n_patches+self.n_cls_tokens)).to(torch.int64).to(spikes.device)
       
        # Embed patch position
        x += self.embed_pos(spikes_patchstamp)

        return self.dropout(x), spikes_patchmask, spikes_patchstamp


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
        self.n_patches = (config.embedder.n_neurons // self.max_space_F) * (config.embedder.n_timesteps // self.max_time_F)

        self.n_cls_tokens = config.embedder.n_cls_tokens

        # Masker
        self.mask = config.masker.active
        if self.mask:
            self.masker = Masker(config.embedder.mode, config.masker)

        # Context span mask
        # context_mask = create_context_mask(
        #     config.context.forward, config.context.backward, 
        #     config.embedder.max_space_F+self.n_cls_tokens, config.embedder.max_time_F
        # )
        # self.register_buffer("context_mask", context_mask, persistent=False)

        # Embedding layer
        self.embedder = NeuralEmbeddingLayer(self.hidden_size, config.embedder)

        # Transformer
        self.layers = nn.ModuleList(
            [NeuralEncoderLayer(
                idx, self.n_patches+self.n_cls_tokens, config.transformer
            ) for idx in range(self.n_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size) 

    def forward(
            self, 
            spikes:                  torch.FloatTensor,  # (n_batch, n_token, n_channels)
            spikes_mask:      Optional[torch.LongTensor] = None,          # (bs, seq_len)
            spikes_timestamp: Optional[torch.LongTensor] = None,          # (bs, seq_len)
    ) -> torch.FloatTensor:                              # (n_batch, n_token, hidden_size)

        B, T, N = spikes.size() # n_batch, n_token, n_channels

        # Mask neural data
        if self.mask:
            spikes, targets_mask = self.masker(spikes)
        else:
            targets_mask = None

        # Embed neural data
        x, spikes_patchmask, spikes_patchstamp = self.embedder(spikes)

        _, T, _ = x.size() 

        # context_mask = self.context_mask[:T,:T].to(x.device).unsqueeze(0).expand(B,T,T)
        spikes_patchmask = spikes_patchmask.unsqueeze(1).expand(B,T,T)
        attn_mask = torch.eye(T).to(x.device, torch.int64).expand(B,T,T) 
        # hack so that even padded spikes attend to themselves and avoid attention issues

        attn_mask = attn_mask | spikes_patchmask

        # Forward transformer
        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
        x = self.out_norm(x)

        return x, targets_mask
        


class STPatch(nn.Module):
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
            n_outputs = config.encoder.embedder.max_time_F * config.encoder.embedder.max_space_F
        elif self.method == "sl":
            n_outputs = kwargs["output_size"]
        else:
            raise Exception(f"Method {self.method} not implemented yet for NDT2")

        decoder_layers = []
        if self.method == "sl":
            decoder_layers.append(
                nn.Linear(
                    self.encoder.n_cls_tokens * self.encoder.hidden_size, n_outputs
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
        spikes:            torch.FloatTensor,                   # (n_batch, n_token, n_channels)
        spikes_mask:      Optional[torch.LongTensor] = None,          # (bs, seq_len)
        spikes_timestamp: Optional[torch.LongTensor] = None,          # (bs, seq_len)
        targets:           Optional[torch.FloatTensor] = None,  # (n_batch, target_len) 
        spikes_lengths:    Optional[torch.LongTensor] = None,   # (n_batch)
        targets_lengths:   Optional[torch.LongTensor] = None,   # (n_batch)
    ) -> STPatchOutput:   

        if self.method == "ssl":
            targets = spikes.clone()
            _, T, N = targets.size()

        # Encode neural data
        x, targets_mask = self.encoder(spikes)

        # Transform neural embeddings into rates/logits
        if self.method == "ssl":
            # Grab the spike embeddings
            x = x[:,self.encoder.n_cls_tokens:]
            outputs = self.decoder(x).reshape((-1, T, N))
        elif self.method == "sl":
            # Grab the prepended [cls] embeddings
            x = x[:,:self.encoder.n_cls_tokens]
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
        )  

    def save_checkpoint(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,"encoder.bin"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,"decoder.bin"))

    def load_checkpoint(self, load_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir,"encoder.bin")))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir,"decoder.bin")))


    
    