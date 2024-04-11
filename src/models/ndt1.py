import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput
from models.masker import Masker

DEFAULT_CONFIG = "src/configs/ndt1.yaml"
# DEFAULT_CONFIG = "/home/yizi/IBL_foundation_model/src/configs/ndt1.yaml"


@dataclass
class NDT1Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None


# Create buffer of biggest possible context mask 
def create_context_mask(context_forward, context_backward, max_F) -> torch.LongTensor: # (max_seq_len, max_seq_len)

        if context_forward == -1 and context_backward == -1:
            return torch.ones(max_F, max_F).to(torch.int64)

        context_forward = context_forward if context_forward >= 0 else max_F
        context_backward = context_backward if context_backward >= 0 else max_F
        mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
        if context_backward > 0:
            back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.int64))
            mask = mask & back_mask
        return mask

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


# Normalize and add noise
class NormAndNoise(nn.Module): 

    def __init__(self, input_size, config):
        super().__init__()
        self.active = config.active

        self.normalize = config.norm is not None
        if self.normalize:
            if config.norm == "layernorm":
                self.norm = nn.LayerNorm(input_size)
            elif config.norm == "scalenorm":
                self.norm = ScaleNorm(input_size ** 0.5)
            elif config.norm == "zscore":
                self.norm = None
            else:
                raise Exception(f"Norm layer {config.norm} not implemented")
        self.eps = config.eps
        self.white_noise_sd = config.white_noise_sd
        self.constant_offset_sd = config.constant_offset_sd
        self.smooth = config.smooth_sd is not None
        if self.smooth:
            kernel = torch.from_numpy(signal.gaussian(1 +config.smooth_sd*6, config.smooth_sd))
            kernel = kernel / kernel.sum()
            self.register_buffer("kernel", kernel, persistent=False)
    

    def forward(self, spikes):
        if not self.active:
            return spikes
            
        B, T, N = spikes.size()

        if self.smooth:
            spikes = F.conv1d(spikes.transpose(-1,-2),self.kernel.unsqueeze(0).unsqueeze(0).expand(N,1,self.kernel.size(0)).to(spikes.dtype), padding="same", groups=N).transpose(-1,-2)
        

        if self.normalize:  
            if self.norm is None:
                spikes = (spikes - spikes.mean(-1).unsqueeze(-1)) / (spikes.std(-1).unsqueeze(-1) + self.eps)
            else:
                spikes = self.norm(spikes)

        if self.white_noise_sd is not None:
            spikes += self.white_noise_sd*torch.randn(B,T,N, dtype=spikes.dtype, device=spikes.device)

        if self.constant_offset_sd is not None:
            spikes += self.constant_offset_sd*torch.randn(B,1,N, dtype=spikes.dtype, device=spikes.device)

        
        return spikes


# Embed and stack
class NeuralEmbeddingLayer(nn.Module):

    def __init__(self, hidden_size, config: DictConfig):
        super().__init__()

        self.adapt = config.adapt
        self.bias = config.bias
        self.input_dim = config.n_channels*config.mult

        if self.adapt:
             # One embedding layer for each day
            if config.mode == "linear":
                self.embed_spikes = nn.ModuleList([
                    nn.Linear(config.n_channels, self.input_dim, bias=config.bias) 
                for i in range(config.n_dates)])

            elif config.mode == "embed":
                self.embed_spikes = nn.ModuleList([
                    nn.Sequential(
                        nn.Embedding(config.max_spikes, config.mult),
                        nn.Flatten(start_dim=-2)
                    )
                for i in range(config.n_dates)])
            else:
                raise Exception(f"Embedding mode {config.mode} cannot be adaptative")
        else:
            # One common embedding layer
            if config.mode == "linear":
                self.embed_spikes = nn.Linear(config.n_channels, self.input_dim, bias=config.bias)
            elif config.mode == "embed":
                self.embed_spikes = nn.Sequential(
                    nn.Embedding(config.max_spikes, config.mult),
                    nn.Flatten(start_dim=-2)
                )
            elif config.mode == "identity":
                self.embed_spikes = nn.Identity()
            else:
                raise Exception(f"Invalid embed mode {config.mode}.")

        if config.mode == "embed" and config.fixup_init:
            self.fixup_initialization(config.init_range, config.spike_log_init, config.max_spikes, adapt=self.adapt)

        # Stacking
        self.stack = config.stack.active
        if self.stack:
            self.stack_size = config.stack.size
            self.stack_stride = config.stack.stride
            self.stacking = nn.Unfold(kernel_size=(config.stack.size, self.input_dim),stride=(config.stack.stride,1))
            self.stacking_mask = nn.Unfold(kernel_size=(config.stack.size, 1),stride=(config.stack.stride,1))
            self.stack_projection = nn.Linear(self.input_dim*config.stack.size,hidden_size)
        else:
            self.projection = nn.Linear(self.input_dim, hidden_size)

        # Activation after embedding
        self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()

        # Embedding scale
        self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

        # Embed postion
        self.pos = config.pos
        if self.pos:
            self.embed_pos = nn.Embedding(config.max_F, hidden_size)

        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self, 
            spikes:           torch.FloatTensor,      # (bs, seq_len, n_channels)
            spikes_mask:      Optional[torch.LongTensor],          # (bs, seq_len)
            spikes_timestamp: Optional[torch.LongTensor],          # (bs, seq_len)
            block_idx:          Optional[torch.LongTensor] = None,   # (bs)
            date_idx:           Optional[torch.LongTensor] = None,   # (bs)
        ) -> Tuple[torch.FloatTensor,torch.LongTensor,torch.LongTensor]:   # (bs, new_seq_len, hidden_size),  (bs, new_seq_len), (bs, new_seq_len)

        # Embed spikes
        if self.adapt:
            x = torch.stack([self.embed_spikes[date_idx[i]](f) for i, f in enumerate(spikes)], 0)
        else:
            x = self.embed_spikes(spikes)

        # Rescaling
        x = self.act(x) * self.scale

        # Stacking
        if self.stack:
            x = self.stack_projection(self.stacking(x.unsqueeze(1)).transpose(1,2))
            spikes_timestamp = spikes_timestamp[:,:x.size(1)] # keep the first positions
            spikes_mask = self.stacking_mask(spikes_mask.unsqueeze(-1).unsqueeze(1).float()).transpose(1,2).prod(-1).to(spikes_mask.dtype) # unmask only spikes tha come from unmasked spikes
        else:
            x = self.projection(x)

        # Embed position
        if self.pos:
            x += self.embed_pos(spikes_timestamp)

        return self.dropout(x), spikes_mask, spikes_timestamp


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



# Attention module.
class NeuralAttention(nn.Module):

    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout, use_rope=False, base=10000., max_F=1024):
        super().__init__()
        
        self.idx = idx

        # Architecture config
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, f"Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads
        self.use_rope = use_rope

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


        # RoPE parameters
        if use_rope:
            cos, sin = get_cos_sin(self.head_size, max_F, base=base, dtype=self.query.weight.dtype, device=self.query.weight.device)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,       
        x:          torch.FloatTensor,                      # (bs, seq_len, hidden_size)
        attn_mask:  torch.LongTensor,                       # (bs, seq_len, seq_len)
        timestamp:  Optional[torch.LongTensor] = None,      # (bs, seq_len)
    ) -> torch.FloatTensor:                                 # (bs, seq_len, hidden_size)

        B, T, _  = x.size()     # batch size and fea len

        # Create batched bool attention mask 
        # assert attn_mask.max() == 1 and attn_mask.min() == 0, ["assertion", attn_mask.max(), attn_mask.min()]
        attn_mask = attn_mask.unsqueeze(1).expand(B,self.n_heads,T,T).bool()            # (B,n_heads,T,T)
        
        # Compute query, key, value for attention
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,T,head_size)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)        #(B,n_heads,T,head_size)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,T,head_size)

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)  # (B,n_heads,T,head_size)

        # Compute attention efficiently
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.attn_dropout if self.training else 0.0), is_causal=False) # (B,n_heads,T,head_size)
        out = out.transpose(1, 2).contiguous().view(B,T, self.hidden_size)       # (B, T, hidden_size)

        return self.out_proj(self.dropout(out)) # (B, T, hidden_size)

    
    

# Encoder layer: bidirectional self-attention + mlp
class NeuralEncoderLayer(nn.Module):
    
    def __init__(self, idx, max_F, config: DictConfig):
        super().__init__()

        self.idx = idx
        
        # Architecture config
        self.use_rope = config.use_rope

        # Encoder block
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.attn = NeuralAttention(idx, config.hidden_size, config.n_heads, config.attention_bias, config.dropout, config.use_rope, config.rope_theta, max_F)
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.mlp = NeuralMLP(config.hidden_size, config.inter_size, config.act, config.mlp_bias, config.dropout)

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

        # Masker
        self.mask = config.masker.active
        if self.mask:
            self.masker = Masker(config.masker)
        
        # Context span mask
        context_mask = create_context_mask(config.context.forward, config.context.backward, config.embedder.max_F)
        self.register_buffer("context_mask", context_mask, persistent=False)

        # Normalization and noising layer
        self.norm_and_noise = NormAndNoise(config.embedder.n_channels, config.norm_and_noise)

        # Embedding layer
        self.embedder = NeuralEmbeddingLayer(self.hidden_size, config.embedder)

        # Transformer
        self.layers = nn.ModuleList([NeuralEncoderLayer(idx, config.embedder.max_F, config.transformer) for idx in range(self.n_layers)])
        self.out_norm = ScaleNorm(self.hidden_size ** 0.5) if config.transformer.use_scalenorm else nn.LayerNorm(self.hidden_size) 
       
        # Out projection
        self.out_proj = NeuralFactorsProjection(self.hidden_size, config.factors)


    def forward(
            self, 
            spikes:           torch.FloatTensor,  # (bs, seq_len, n_channels)
            spikes_mask:      torch.LongTensor,   # (bs, seq_len)
            spikes_timestamp: torch.LongTensor,   # (bs, seq_len)
            block_idx:          Optional[torch.LongTensor] = None,   # (bs)
            date_idx:           Optional[torch.LongTensor] = None,   # (bs)
    ) -> torch.FloatTensor:                     # (bs, seq_len, hidden_size)
        
        B, T, N = spikes.size() # batch size, fea len, n_channels
        
        if self.int_spikes:
            spikes = spikes.to(torch.int64)
        
        # Normalize across channels and add noise
        spikes = self.norm_and_noise(spikes)

        # Mask neural data
        if self.mask:
            spikes, targets_mask = self.masker(spikes)
            targets_mask = targets_mask & spikes_mask.unsqueeze(-1).expand(B,T,N)
        else:
            targets_mask = None

        # Embed neural data
        x, spikes_mask, spikes_timestamp = self.embedder(spikes, spikes_mask, spikes_timestamp, block_idx, date_idx)

        _, T, _ = x.size() # feature len may have changed after stacking

        # Prepare 
        context_mask = self.context_mask[:T,:T].to(x.device).unsqueeze(0).expand(B,T,T)
        spikes_mask = spikes_mask.unsqueeze(1).expand(B,T,T)
        self_mask = torch.eye(T).to(x.device, torch.int64).expand(B,T,T) # hack so that even padded spikes attend to themselves and avoid attention issues
        attn_mask = self_mask | (context_mask & spikes_mask)
        
        # Forward transformer
        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask, timestamp=spikes_timestamp)
        x = self.out_norm(x)

        return self.out_proj(x), targets_mask



# Encoder for time binned neural data
class NDT1(nn.Module):

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
        self.encoder = NeuralEncoder(config.encoder)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))


        # Build decoder
        if self.method == "ssl":
            assert config.encoder.masker.active, "Can't pretrain with inactive masking"
            n_outputs = config.encoder.embedder.n_channels
        elif self.method == "ctc":
            n_outputs = kwargs["vocab_size"]
        elif self.method == "sl":
            n_outputs = kwargs["output_size"]
        else:
            raise Exception(f"Method {self.method} not implemented yet for NDT1")

        decoder_layers = []
        if self.method == "sl":
            decoder_layers.append(
                nn.Linear(config.encoder.embedder.max_F * self.encoder.out_proj.out_size, n_outputs)
            )
        else:
            decoder_layers.append(nn.Linear(self.encoder.out_proj.out_size, n_outputs))

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
        spikes:           torch.FloatTensor,  # (bs, seq_len, n_channels)
        spikes_mask:      torch.LongTensor,   # (bs, seq_len)
        spikes_timestamp: torch.LongTensor,   # (bs, seq_len)
        targets:          Optional[torch.FloatTensor] = None,  # (bs, tar_len)
        spikes_lengths:   Optional[torch.LongTensor] = None,   # (bs) 
        targets_lengths:  Optional[torch.LongTensor] = None,   # (bs)
        block_idx:        Optional[torch.LongTensor] = None,   # (bs)
        date_idx:         Optional[torch.LongTensor] = None,   # (bs)
    ) -> NDT1Output:      

        if self.method == "ssl":
            targets = spikes.clone()
            if self.encoder.int_spikes:
                targets = targets.to(torch.int64)

        # Encode neural data
        x, targets_mask = self.encoder(spikes, spikes_mask, spikes_timestamp, block_idx, date_idx)
        spikes_lengths = self.encoder.embedder.get_stacked_lens(spikes_lengths)

        # Transform neural embeddings into rates/logits
        if self.method == "sl":
            x = x.flatten(start_dim=1)
            
        outputs = self.decoder(x)

        # Compute the loss over unmasked outputs
        if self.method == "ssl":
            loss = (self.loss_fn(outputs, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()
        elif self.method == "ctc":
            loss = self.loss_fn(outputs.transpose(0,1), targets, spikes_lengths, targets_len)
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)
        elif self.method == "sl":
            loss = self.loss_fn(outputs, targets).sum()
            n_examples = torch.Tensor([len(targets)]).to(loss.device, torch.long)

        return NDT1Output(
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



class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm