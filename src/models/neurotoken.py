import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from math import ceil

from pytorch_memlab import profile, MemReporter


from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput
from models.masker import Masker
# from models.patcher import Patcher
from models.region_lookup import RegionLookup 


from transformers.models.llama.modeling_llama import LlamaFlashAttention2 # Import the new attention layer
from transformers import LlamaConfig

DEFAULT_CONFIG = "src/configs/ndt1.yaml"
#TODO: make default config for neurotoken

with open('data/target_eids.txt') as file:
    include_eids = [line.rstrip() for line in file]

@dataclass
class NeurotokenizerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None
    # num_neuron: Optional[int] = None      #TOOD: decide whether to add - included in stpatch


def create_context_mask(          
    context_forward, context_backward, max_space_patches, max_time_patches
) -> torch.LongTensor: # (max_n_token, max_n_token)

    # bidirectional
    if context_forward == -1 and context_backward == -1:
        return torch.ones(max_time_patches*max_space_patches, max_time_patches*max_space_patches).to(torch.int64)

    context_forward = context_forward if context_forward >= 0 else max_time_patches
    context_backward = context_backward if context_backward >= 0 else max_time_patches
    mask = (torch.triu(torch.ones(max_time_patches, max_time_patches), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
    if context_backward > 0:
        back_mask = (torch.triu(torch.ones(max_time_patches, max_time_patches), diagonal=-context_backward).to(torch.int64))
        mask = mask & back_mask

    # (max_seq_len, max_seq_len) -> (max_n_token, max_n_token) 
    mask = mask.repeat(max_space_patches, max_space_patches)
    return mask

# # Create buffer of biggest possible context mask -- FROM NDT1
# def create_context_mask(context_forward, context_backward, max_F) -> torch.LongTensor: # (max_seq_len, max_seq_len)

#         if context_forward == -1 and context_backward == -1:
#             return torch.ones(max_F, max_F).to(torch.int64)

#         context_forward = context_forward if context_forward >= 0 else max_F
#         context_backward = context_backward if context_backward >= 0 else max_F
#         mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
#         if context_backward > 0:
#             back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.int64))
#             mask = mask & back_mask
#         return mask

# Copied from hf Llama
# Precompute cos and sin for RoPE
def get_cos_sin(dim, max_F, base=10000, dtype=torch.get_default_dtype(), device=None):

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        t = torch.arange(max_F, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

# Rotates half the hidden dims of the input.
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)

# Applies RoPE to the query and key tensors.
def apply_rotary_pos_emb(q, k, pos_ids, cos, sin, unsqueeze_dim=1):

    cos = cos[pos_ids].unsqueeze(unsqueeze_dim)
    sin = sin[pos_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    
    return q_embed, k_embed

# MLP
class NeuralMLP(nn.Module):

    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()

        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))



class NeuralAttention(nn.Module):

    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout, use_rope=False, base=10000., max_F=1024, causal = False):
        super().__init__()
        
        # self.idx = idx
        # self.hidden_size = hidden_size
        self.n_heads = n_heads
        # assert self.hidden_size % self.n_heads == 0, f"Hidden dim is not multiple of head size"
        # self.head_size = self.hidden_size // self.n_heads
        # self.use_rope = use_rope

        llama_config = LlamaConfig(
        hidden_size=hidden_size,               # Adjust based on your model’s requirements
        num_attention_heads=n_heads,          # Number of attention heads
        attention_dropout=dropout,          # Dropout rate for attention
        num_key_value_heads=4,          # Set based on model configuration
        head_dim=hidden_size // n_heads,                    # Dimension of each attention head
        max_position_embeddings=1024,   # Maximum position embeddings
        rope_theta=10000,               # Rotary embedding theta
        attention_bias=True,             # Bias usage in projections
        is_causal = causal
            )


        # Replace the existing query, key, value layers with LlamaFlashAttention2
        self.attention = LlamaFlashAttention2(llama_config, layer_idx=0)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
    
    # @profile
    def forward(
        self,       
        x:          torch.FloatTensor,                      # (bs, seq_len, hidden_size)
        attn_mask:  torch.LongTensor,                       # (bs, seq_len, seq_len)
        timestamp:  Optional[torch.LongTensor] = None,      # (bs, seq_len)
    ) -> torch.FloatTensor:                                 # (bs, seq_len, hidden_size)

        B, T, _  = x.size()     # batch size and fea len

        # Create batched bool attention mask 
        # attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, T, T).bool()  # (B, n_heads, T, T)
        # attn_mask = attn_mask[:, 0, :].bool()
        # attn_mask = attn_mask.bool()

        # Use the LlamaFlashAttention2 layer
        out = self.attention(x, attn_mask, position_ids=timestamp)  # Adjust this line as per LlamaFlashAttention2's API
        # Extract the tensor from the tuple
        if isinstance(out, tuple):
            out = out[0]  # Get the tensor output

        return self.out_proj(self.dropout(out))  # (B, T, hidden_size)
        # return out

    
    

# Encoder layer: bidirectional self-attention + mlp
class NeuralEncoderLayer(nn.Module):
    
    def __init__(self, idx, max_F, config: DictConfig, hidden_size=None):
        super().__init__()

        self.idx = idx
        # Architecture config
        self.use_rope = config.use_rope

        if hidden_size is None: 
            hidden_size = config.hidden_size

        # Encoder block
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(hidden_size) 
        self.attn = NeuralAttention(idx, hidden_size, config.n_heads, config.attention_bias, config.dropout, config.use_rope, config.rope_theta, max_F, config.causal)
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(hidden_size) 
        self.mlp = NeuralMLP(hidden_size, config.inter_size, config.act, config.mlp_bias, config.dropout)

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:          torch.FloatTensor,                  # (bs, seq_len, hidden_size)
        attn_mask:  torch.LongTensor,                   # (bs, seq_len, seq_len)
        timestamp:  Optional[torch.LongTensor] = None,  # (bs, seq_len)          
    ) -> torch.FloatTensor :                            # (bs, seq_len, hidden_size)
        
        # LN -> Attention -> Residual connectiob
        x = x + self.attn(self.ln1(x), attn_mask, timestamp if self.use_rope else None)

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





class NeuralFactorsProjection(nn.Module):

    def __init__(self, hidden_size, config):
        
        super().__init__()
        
        self.out_size = config.size if config.active else hidden_size
        # self.out_space = "factors" if config.active else "hidden"
        
        self.dropout = nn.Dropout(config.dropout)

        if config.active:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
            # Renitialize weights
            if config.fixup_init:
                self.proj[0].weight.data.uniform_(-config.init_range, config.init_range)
                if config.bias:
                    self.proj[0].bias.data.zero_()
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.dropout(x))
        

class NeuralEncoder(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__() 

        self.int_spikes = config.embedder.mode == "embed"
        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers
        self.max_F = config.embedder.max_F
        self.max_time_F = config.embedder.max_time_F
        self.max_space_F = config.embedder.max_space_F
        self.n_timesteps = config.embedder.n_timesteps
        self.n_time_patches = ceil(self.n_timesteps/self.max_time_F)
        self.n_space_patches = ceil(config.embedder.n_channels/self.max_space_F)
    


        # Masker
        self.mask = config.masker.force_active
        self.mask_token = True
        # if config.masker.mode == 'random_token':
        #     self.mask = False
        #     self.mask_token = True

        # if self.mask | self.mask_token:
        #     self.masker = Masker(config.masker)

        self.embed_region = config.embed_region
        self.regionlookup = RegionLookup(config)

        # Patcher
        # self.patch = config.patcher.active
        
        # Context span mask
        self.context_forward = config.context.forward
        self.context_backward = config.context.backward
        # context_mask = create_context_mask(self.context_forward, self.context_backward, self.n_space_patches, self.n_time_patches)  #NEW
        # self.register_buffer("context_mask", context_mask, persistent=False)        #NEW

        # Build stitcher
        if config.stitching:
            self.stitcher = NeuralStitcher(kwargs['num_neurons'],
                                           config.embedder.n_channels)

        self.use_prompt = config.embedder.use_prompt
        self.use_session = config.embedder.use_session

        # Transformer
        self.embedding_layer = nn.Linear(self.max_time_F, self.hidden_size)
        self.layers = nn.ModuleList([NeuralEncoderLayer(idx, config.embedder.max_F, config.transformer) for idx in range(self.n_layers)])
        self.out_norm = ScaleNorm(self.hidden_size ** 0.5) if config.transformer.use_scalenorm else nn.LayerNorm(self.hidden_size) 
       
        # Out projection
        self.out_proj = NeuralFactorsProjection(self.hidden_size, config.factors)

        self.pos = config.embedder.pos
        if self.pos:
            if self.embed_region:
                self.region_embeddings = nn.Sequential(
                    nn.Embedding(self.regionlookup.max_region_indx, self.hidden_size),nn.LayerNorm(self.hidden_size),)
            self.embed_unit = nn.Sequential(
                nn.Embedding(self.n_space_patches, self.hidden_size),nn.LayerNorm(self.hidden_size),)

        self.learned_mask = nn.Parameter(torch.randn(1, 1, self.hidden_size))


    # @profile
    def forward(
            self, 
            spikes:           torch.FloatTensor,  # (bs, seq_len, n_channels)
            pad_space_len:    int,   # (bs,)
            pad_time_len:     int,
            timestamps:       Optional[torch.LongTensor],  # (bs, seq_len) or None
            spacestamps:      Optional[torch.LongTensor],  # (bs, seq_len) or None
            token_masks:       Optional[torch.BoolTensor],
            time_attn_mask:   torch.LongTensor,   # (bs, seq_len)
            space_attn_mask:  torch.LongTensor,   # (bs, seq_len)
            block_idx:        Optional[torch.LongTensor] = None,   # (bs)
            date_idx:         Optional[torch.LongTensor] = None,   # (bs)
            neuron_regions:   Optional[np.ndarray] = None,  # (bs, n_channels)
            masking_mode:     Optional[str] = None,
            eval_mask:        Optional[torch.LongTensor] = None,
            num_neuron:       Optional[torch.LongTensor] = None,
            eid:              Optional[str] = None,

    ) -> torch.FloatTensor:                     # (bs, seq_len, hidden_size)
    
        # B, T, N = spikes.size() # batch size, fea len, n_channels
        if self.int_spikes:
            spikes = spikes.to(torch.int64)

        if self.embed_region:
            region_indx = self.regionlookup(neuron_regions).to(spikes.device)
        else:
            region_indx = None

        # Patch neural data
        # if self.patch:    patching is done on CPU in trainer code now
            
        B, _T, N = spikes.size()        

        # stitcher 
        if hasattr(self, 'stitcher'):
            spikes = self.stitcher(spikes, str(num_neuron))

        #Create tokens
        spikes_embed = self.embedding_layer(spikes)

        # Replace with learned mask token  TODO: double check that this method is right
        expanded_mask_token = self.learned_mask.expand(B, _T, -1)
        x = torch.where(token_masks, expanded_mask_token, spikes_embed)

        if False and self.embed_region:
            x = x + self.region_embeddings(regionstamps)
        x = x + self.embed_unit(spacestamps)
        _, T, N_embed = x.shape

        # # Prepare 
        # if self.use_prompt or self.use_session: #TODO: Update this 
        #     context_mask = torch.cat((torch.ones((_T,T-_T)), self.context_mask[:_T,:_T]), dim=1)
        #     context_mask = torch.cat((torch.ones((T-_T,T)), context_mask), dim=0)
        #     context_mask = context_mask.to(x.device, torch.int64)
        # else:
        #     context_mask = torch.ones((B, T), device=x.device).bool()
        
        # Forward transformer
        # for idx, layer in enumerate(self.layers):
        for layer in self.layers:
            x = layer(x, attn_mask=None, timestamp = timestamps) 
        x = self.out_norm(x)

        #NOTE: _T value may be wrong here 
        return self.out_proj(x), _T     


class NeuralStitcher(nn.Module):

    def __init__(self, 
                 num_neurons:list,
                 n_channels:int,):
        super().__init__()

        stitcher_dict = {}
        for num_neuron in num_neurons:
            stitcher_dict[str(num_neuron)] = nn.Linear(num_neuron, n_channels)
        self.stitcher_dict = nn.ModuleDict(stitcher_dict)

    def forward(self, x, block_idx):
        return self.stitcher_dict[block_idx](x)
    
class StitchDecoder(nn.Module):

    def __init__(self,
                 num_neurons:list,
                 n_channels:int):
        super().__init__()

        stitch_decoder_dict = {}
        for num_neuron in num_neurons:
            stitch_decoder_dict[str(num_neuron)] = nn.Linear(n_channels, num_neuron)
        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)

    def forward(self, x, block_idx):
        return self.stitch_decoder_dict[block_idx](x)

# Encoder for time binned neural data
class Neurotokenizer(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)
        self.method = kwargs["method_name"]
        
        self.pad_value = -1.    #added from stpatch

        # Build encoder
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = os.path.join(encoder_pt_path, "encoder_config.yaml")
            config["encoder"] = update_config(config.encoder, encoder_config)
        self.encoder = NeuralEncoder(config.encoder, **kwargs)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))

        self.use_prompt = config.encoder.embedder.use_prompt
        self.use_session = config.encoder.embedder.use_session

        # stitching
        if config.encoder.stitching:
            self.stitching=True
            self.n_channels = config.encoder.embedder.n_channels
            self.hidden_size = config.encoder.transformer.hidden_size
            self.stitch_decoder = StitchDecoder(kwargs['num_neurons'], self.hidden_size)
        else:
            self.n_channels = kwargs['num_neurons'][0]

        # Build decoder
        if self.method == "ssl":
            assert config.encoder.masker.force_active, "Can't pretrain with inactive masking"
            n_outputs = config.encoder.embedder.max_time_F * config.encoder.embedder.max_space_F
        elif self.method == "ctc":
            n_outputs = kwargs["vocab_size"]
        elif self.method == "sl":
            n_outputs = kwargs["output_size"]
        else:
            raise Exception(f"Method {self.method} not implemented yet for Neurotokenizer")

        decoder_layers = []
        if self.method == "sl":     #TODO: figure out how this should be changed
            decoder_layers.append(
                # nn.Linear(config.encoder.embedder.max_F * self.encoder.out_proj.out_size, n_outputs)
                nn.Linear((self.encoder.n_time_patches * self.encoder.n_space_patches) * self.encoder.out_proj.out_size, n_outputs)
            )
        else:
            decoder_layers.append(nn.Linear(self.encoder.hidden_size, n_outputs))
            # decoder_layers.append(nn.Linear(self.encoder.out_proj.out_size, self.n_channels*self.encoder.max_time_F))

        if self.method == "sft" and not kwargs["use_lograte"]:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        if self.method == "ctc":
            decoder_layers.append(nn.LogSoftmax(dim=-1))  # CTC loss asks for log-softmax-normalized logits
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
        elif self.method == "ctc":
             self.loss_fn = nn.CTCLoss(reduction="none", blank=kwargs["blank_id"], zero_infinity=kwargs["zero_infinity"])
        elif self.method == "sl":
            if kwargs["loss"] == "cross_entropy":
                self.loss_fn = nn.CrossEntropyLoss(reduction="none")
            elif kwargs["loss"] == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            else:
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for sl")
        

    def forward(
        self, 
        spikes:             torch.FloatTensor,  # (bs, seq_len, n_channels)
        time_attn_mask:     torch.LongTensor,   # (bs, seq_len)
        space_attn_mask:    torch.LongTensor,   # (bs, seq_len)
        timestamps:         torch.LongTensor,   # (bs, seq_len)
        spacestamps:        torch.LongTensor,   # (bs, seq_len)
        token_masks:        torch.BoolTensor,
        targets_mask:      torch.BoolTensor,
        targets:            Optional[torch.FloatTensor] = None,  # (bs, tar_len)
        spikes_lengths:     Optional[torch.LongTensor] = None,   # (bs) 
        targets_lengths:    Optional[torch.LongTensor] = None,   # (bs)
        block_idx:          Optional[torch.LongTensor] = None,   # (bs)
        date_idx:           Optional[torch.LongTensor] = None,   # (bs)
        neuron_regions:     Optional[torch.LongTensor] = None,   # (bs, n_channels)
        masking_mode:       Optional[str] = None,
        spike_augmentation: Optional[bool] = False,
        eval_mask:          Optional[torch.LongTensor] = None,
        num_neuron:         Optional[torch.LongTensor] = None,
        eid:                Optional[str] = None,
    ) -> NeurotokenizerOutput:  

        #from stpatch, T_ is for orig seq length, _T is for patched seq length, T is for output length (output: x)
        B, T_, N = spikes.size()    

        if eval_mask is not None:
            targets_mask = eval_mask.clone()
        
        # if neuron_regions type is list 
        if isinstance(neuron_regions, list):
            neuron_regions = np.asarray(neuron_regions).T

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

        if self.method == "ssl":
            targets = spikes.clone()
            if self.encoder.int_spikes:
                targets = targets.to(torch.int64)

        #From stpatch, not sure if handling padding will be necessary but leaving here
        # Track padded values to prevent them from being used
        if (spikes[0,:,0] == self.pad_value).sum() == 0:
            pad_time_len = T_
        else:
            pad_time_len = (spikes[0,:,0] == self.pad_value).nonzero().min().item() 

        if (spikes[0,0,:] == self.pad_value).sum() == 0:
            pad_space_len = N
        else:
            pad_space_len = (spikes[0,0,:] == self.pad_value).nonzero().min().item() 

        # Encode neural data    TODO: figure out why the targets_mask | new_mask part is needed isntead of just getting targets_mask from encoder 
        # targets_mask = torch.zeros_like(spikes, dtype=torch.int64)
        x, _T = self.encoder(spikes, pad_space_len, pad_time_len, timestamps, spacestamps, token_masks, time_attn_mask, space_attn_mask, block_idx, date_idx, neuron_regions, masking_mode, eval_mask, num_neuron, eid)
        # targets_mask = targets_mask | new_mask

        if False:   #TODO: embedder class is not being used atm, need to move other required functions out of that class 
                            #or move embedding funcs into that class
            spikes_lengths = self.encoder.embedder.get_stacked_lens(spikes_lengths)

        _, T, _ = x.size()

        if self.use_prompt or self.use_session:
            x = x[:,T-_T:]

        # Transform neural embeddings into rates/logits -- get outputs and targets
        if self.method == "sl":
            x = x.flatten(start_dim=1)
            outputs = self.decoder(x)

        if hasattr(self, "stitching") and self.method == "ssl":
            outputs = self.stitch_decoder(x, str(num_neuron))

        elif self.method == "ssl":
            outputs = self.decoder(x).reshape((B,T_,-1))[:,:,:pad_space_len]
            targets = targets.reshape((B,T_,-1))[:,:,:pad_space_len]
            targets_mask = targets_mask.reshape((B,T_,-1))[:,:,:pad_space_len]


        # Compute the loss over unmasked outputs
        if self.method == "ssl":
            if self.encoder.mask:
                loss = (self.loss_fn(outputs, targets) * targets_mask).sum()
            else:
                loss = self.loss_fn(outputs, targets).sum()
            
            # if hasattr(self, "stitching"):        #originally commented out as first if statement in ndt1, not sure if it's safe to delete or still being debugged
            #         if self.n_channels <= targets.shape[2]:
            #             targets, targets_mask = targets[:,:,:self.n_channels], targets_mask[:,:,:self.n_channels]
            #         else:
            #             outputs = outputs[:,:,:targets.shape[2]]

            n_examples = targets_mask.sum()
        elif self.method == "ctc":      #THIS METHOD ISN'T TESTED WITH MODIFIED CODE
            loss = self.loss_fn(outputs.transpose(0,1), targets, spikes_lengths, targets_lengths)
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)
        elif self.method == "sl":
            loss = self.loss_fn(outputs, targets).sum()
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)

        return NeurotokenizerOutput(
            loss=loss,
            n_examples=n_examples,
            preds=outputs,
            targets=targets
            # num_neuron = pad_space_len     #TOOD: decide whether to add - included in stpatch  
        )


    def save_checkpoint(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,"encoder.bin"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,"decoder.bin"))

    def load_checkpoint(self, load_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir,"encoder.bin")))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir,"decoder.bin")))



class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
    
#Not used for this architecture
class MegaTokenizer(nn.Module):
    def __init__(self, neuron_combined_hidden_size, MEGA_hidden_size):
        super().__init__()
        self.neuron_combined_hidden_size = neuron_combined_hidden_size
        self.middle_layer_size = (neuron_combined_hidden_size + MEGA_hidden_size) // 2
        self.embed_tokens = nn.Sequential(
                                nn.Linear(neuron_combined_hidden_size, self.middle_layer_size, bias=True),
                                nn.ReLU(),
                                # nn.Linear(self.middle_layer_size, self.middle_layer_size, bias=True),
                                # nn.ReLU(),
                                nn.Linear(self.middle_layer_size, MEGA_hidden_size, bias=True))
                                
        #skipping projection Linear layer used in NeuralEmbeddingLayer since we don't need to embed pos and time in a lower dim (input_dim)
    
    def forward(self, x):
        megaX = self.embed_tokens(x)
        return megaX


    # Compute new lens after stacking
    def get_stacked_lens(self, lens):
        return lens if not self.stack else (1 + (lens - self.stack_size) / self.stack_stride).to(lens.dtype)

    # Initialization methods copied from NDT
    def fixup_initialization(self, init_range, spike_log_init, max_spikes, adapt):
        if adapt:
            for i in range(len(self.embed_spikes)):
                if spike_log_init:
                    # Use a log scale, since we expect spike semantics to follow compressive distribution
                    log_scale = torch.arange(1, max_spikes+1).float().log() # 1 to lg
                    log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                    log_scale = log_scale * init_range
                    # Add some noise
                    self.embed_spikes[i][0].weight.data.uniform_(-init_range / 10, init_range / 10)
                    self.embed_spikes[i][0].weight.data += log_scale.unsqueeze(1).expand_as(self.embed_spikes[i][0].weight.data)
                else:
                    self.embed_spikes[i][0].weight.data.uniform_(-init_range, init_range)
        else:
            if spike_log_init:
                # Use a log scale, since we expect spike semantics to follow compressive distribution
                log_scale = torch.arange(1, max_spikes+1).float().log() # 1 to lg
                log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                log_scale = log_scale * init_range
                # Add some noise
                self.embed_spikes[0].weight.data.uniform_(-init_range / 10, init_range / 10)
                self.embed_spikes[0].weight.data += log_scale.unsqueeze(1).expand_as(self.embed_spikes[0].weight.data)
            else:
                self.embed_spikes[0].weight.data.uniform_(-init_range, init_range)


