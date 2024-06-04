import os
from typing import List, Union, Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvision.ops import MLP
import numpy as np

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput
from models.masker import Masker
from models.region_lookup import RegionLookup

DEFAULT_CONFIG = "src/configs/itransformer/itransformer.yaml"

with open('data/target_eids.txt') as file:
    include_eids = [line.rstrip() for line in file]

@dataclass
class iTransformerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None


class AverageTokens(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(dim=self.dim)



class CustomTransformerEncoderLayer(nn.Module):

        def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            activation=ACT2FN["gelu"],
            dropout=0.1,
            bias=True,
            batch_first=True,
        ):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = activation

        def forward(
            self,
            src,
            src_mask=None,
        ):
            # Attention block
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, average_attn_weights=False)
            
            src = src + self.dropout1(src2)
            
            src = self.norm1(src)

            # Feedforward block
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn_weights


class CustomTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        activation=ACT2FN["gelu"],
        dropout=0.1,
        bias=True,
        batch_first=True,
        num_layers=5,
        norm=None,        # Norm at the end of the encoder
        attention_mode_list=None,  # list of inter-region, intra-region, all, custom-attn
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
            )
            for i in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        src,
        attn_mask_list,  # a list of attn_mask for each layer
    ):
        x = src
        for i, layer in enumerate(self.layers):
            mask = attn_mask_list[i].to(torch.bool)  # it has to be a bool tensor

            # _ is the attention weights
            x, _ = layer(x, src_mask=mask)

        if self.norm is not None:
            x = self.norm(x)

        return x




class iTransformerEncoder(nn.Module):

    def __init__(
        self,
        config: DictConfig,
        use_prompt: bool,
        use_session: bool,
    ):
        super().__init__()

        self.attn_mode = config.attn_mode
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.attn_mix_sequence = ["all"] * self.n_layers  # default attention mode for each layer

        if self.attn_mode == 'mix_multihead':
            self.attn_mix_n = config.attn_mix_n
        elif self.attn_mode == 'mix_sample':
            self.attn_mix_ratio = config.attn_mix_ratio
        elif self.attn_mode == 'mix_sequence':
            self.attn_mix_sequence = config.attn_mix_seq
        else:
            pass  # TODO: implement other mix methods.

        self.embed = nn.Sequential(
            MLP(
                in_channels=config.embedder.max_n_bins, 
                hidden_channels=[config.hidden_size, config.hidden_size],
                bias=config.embedder.bias,
                dropout=config.embedder.dropout,
            ),
            nn.LayerNorm(config.hidden_size), # MAJOR CHANGE HERE
        )
        
        self.embed_channel = (config.max_n_channels != 0)
        if self.embed_channel:
            self.channel_embeddings = nn.Sequential(
                nn.Embedding(config.max_n_channels, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            )

        self.embed_region = config.embed_region
        if self.embed_region:
            self.regionlookup = RegionLookup(config)
            self.region_embeddings = nn.Sequential(
                nn.Embedding(self.regionlookup.max_region_indx, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            )

        self.embed_nemo = config.embed_nemo
        if self.embed_nemo:
            self.nemo_embeddings = nn.Sequential(
                MLP(
                    in_channels=config.embedder.nemo_dim, 
                    hidden_channels=[config.hidden_size, config.hidden_size],
                    bias=config.embedder.bias,
                    dropout=config.embedder.dropout,
                ),
                nn.LayerNorm(config.hidden_size), 
            )


        # Embed prompt token
        self.use_prompt = use_prompt
        if self.use_prompt:
            self.mask_types = ['neuron', 'temporal', 'causal', 'inter-region', 'intra-region']
            self.mask_to_indx = {r: i for i,r in enumerate(self.mask_types)}
            self.embed_prompt = nn.Embedding(len(self.mask_types), hidden_size) 

        # Embed session token
        self.use_session = use_session
        if self.use_session:
            self.eid_lookup = include_eids
            self.eid_to_indx = {r: i for i,r in enumerate(self.eid_lookup)}
            self.embed_session = nn.Embedding(len(self.eid_lookup), hidden_size) 

        self.embed_dropout = nn.Dropout(config.embedder.dropout)

        # use the custom version for flexibility
        norm = nn.LayerNorm(config.hidden_size)
        self.transformer = CustomTransformerEncoder(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=4 * config.hidden_size,
            activation=ACT2FN[config.activation],
            bias=config.bias,
            dropout=config.dropout,
            batch_first=True,
            num_layers=config.n_layers,
            norm=norm,  # final norm
            attention_mode_list=self.attn_mix_sequence  # Type of each layer
        )

        # transformer_layer = TransformerEncoderLayer(
        #     d_model = config.hidden_size,
        #     nhead = config.n_heads,
        #     dim_feedforward = 4*config.hidden_size,
        #     activation=ACT2FN[config.activation],
        #     # bias=config.bias,
        #     dropout=config.dropout,
        #     batch_first=True,
        # )

        # self.transformer = TransformerEncoder(
        #     encoder_layer = transformer_layer,
        #     num_layers = config.n_layers,
        #     norm = nn.LayerNorm(config.hidden_size),
        #     enable_nested_tensor = True,
        # )

    def forward(
        self, 
        spikes:              torch.LongTensor,                   # (bs, seq_len, n_channels)
        spikes_timestamps:   Optional[torch.LongTensor]  = None, # (bs, seq_len) 
        spikes_spacestamps:  Optional[torch.LongTensor]  = None, # (bs, n_channels) 
        neuron_regions:      Optional[np.ndarray]        = None, # (bs, n_channels)
        nemo_rep:            Optional[np.ndarray]        = None, 
        attention_mask=None,  # (bs, n_channels, n_channels) or a list of this
    ) -> torch.FloatTensor:   # (batch, n_channels, hidden_size)
        
    
        tokens = self.embed(spikes.transpose(1,2))  # (batch, n_channels, hidden_size)
        
        if self.embed_channel:
            if spikes_spacestamps is None:
                spikes_spacestamps = torch.arange(0, tokens.size(1), dtype=torch.int64, device=spikes.device)
            channel_embeds = self.channel_embeddings(spikes_spacestamps) # (n_channels, hidden_size)
            tokens += channel_embeds                                    # (batch, n_channels, hidden_size)

        if self.embed_region:
            neuron_regions = np.array(neuron_regions).T
            region_indx = self.regionlookup(neuron_regions).to(spikes.device)
            # region_indx = torch.stack([torch.tensor([self.region_to_indx[r] for r in row], dtype=torch.int64, device=spikes.device) for row in neuron_regions], dim=0)
            region_embeds = self.region_embeddings(region_indx)        # (batch, n_channels, hidden_size)
            tokens += region_embeds

        if self.embed_nemo:
            nemo_embeds = self.nemo_embeddings(nemo_rep)       
            tokens += nemo_embeds

        # Prepend prompt token 
        if self.use_prompt:
            mask_idx = torch.tensor(self.mask_to_indx[masking_mode], dtype=torch.int64, device=spikes.device)
            tokens = torch.cat((self.embed_prompt(mask_idx)[None,None,:].expand(B,-1,-1), tokens), dim=1) 

        if self.use_session:
            session_idx = torch.tensor(self.eid_to_indx[eid], dtype=torch.int64, device=spikes.device)
            tokens = torch.cat((self.embed_session(session_idx)[None,None,:].expand(B,-1,-1), tokens), dim=1)

        if self.attn_mode == 'mix_sequence':
            x = self.transformer(self.embed_dropout(tokens), attn_mask_list=attention_mask)
        else:
            attention_mask = [attention_mask for _ in range(self.n_layers)]
            x = self.transformer(self.embed_dropout(tokens), attn_mask_list=attention_mask)
            
        # x = self.transformer(self.embed_dropout(tokens))              # (batch, [1+]n_channels, hidden_size)

        return x


class iTransformer(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs,
    ):
        super().__init__()
        self.method = kwargs["method_name"]
        
        config = update_config(DEFAULT_CONFIG, config)

        # Load pretrained configs
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = torch.load(os.path.join(encoder_pt_path, "encoder_config.pth"))
            config["encoder"] = update_config(config.encoder, encoder_config)
            
        decoder_pt_path = config["decoder"].pop("from_pt", None)
        if decoder_pt_path is not None:
            decoder_config = torch.load(os.path.join(decoder_pt_path, "decoder_config.pth"))
            config["decoder"] = update_config(config.decoder, decoder_config)

        
        # Build masker
        self.masker = Masker(config.encoder.masker)

        self.use_prompt = config.encoder.embedder.use_prompt
        self.use_session = config.encoder.embedder.use_session
        
        # Build encoder
        self.encoder = iTransformerEncoder(config.encoder, self.use_prompt, self.use_session)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))

        # Build decoder
        if self.method == "ssl":
            # assert config.encoder.masker.force_active, "Can't pretrain with inactive masking"
            n_outputs = config.encoder.embedder.max_n_bins
        elif self.method == "ctc":
            n_outputs = kwargs["vocab_size"] * config.encoder.embedder.max_n_bins
            self.output_shape = (config.encoder.embedder.max_n_bins, kwargs["vocab_size"])
        elif self.method == "dyn_behaviour":
            n_outputs = config.encoder.embedder.max_n_bins
        elif self.method == "stat_behaviour":
            if kwargs["loss"] == "xent":
                n_outputs = kwargs["n_labels"]
            elif kwargs["loss"] == "mse":
                n_outputs = 1
        else:
            raise Exception(f"Method {self.method} not implemented")

        decoder_layers = []

        if self.method in ["ctc","dyn_behaviour","stat_behaviour"]:
            decoder_layers.append(AverageTokens(dim=1)) # Get rid of the channel dimension

        if config.decoder.mlp_decoder:
            decoder_layers.append(nn.Linear(config.encoder.hidden_size, config.encoder.hidden_size))
            decoder_layers.append(ACT2FN[config.decoder.activation])
        decoder_layers.append(nn.Linear(config.encoder.hidden_size, n_outputs))

        if self.method == "ssl" and not kwargs["use_lograte"]:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        if self.method == "ctc":
            decoder_layers.append(nn.LogSoftmax(dim=-1))  # CTC loss receives log-softmax-normalized logits
        self.decoder = nn.Sequential(*decoder_layers)

        # Load decoder weights
        if decoder_pt_path is not None:
            self.decoder.load_state_dict(torch.load(os.path.join(decoder_pt_path,"decoder.bin")))


        # Build loss function
        if self.method == "ssl":
            self.loss_name = kwargs["loss"]
            self.use_lograte = kwargs["use_lograte"]
            if self.loss_name == "poisson_nll":
                self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=self.use_lograte)
            elif self.loss_name == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            else:   
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for ssl")
        elif self.method == "ctc":
            self.loss_fn = nn.CTCLoss(reduction="sum", blank=kwargs["blank_id"], zero_infinity=kwargs["zero_infinity"])
        elif self.method == "dyn_behaviour":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif self.method == "stat_behaviour":
            self.loss_name = kwargs["loss"]
            if self.loss_name == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            elif self.loss_name == "cross_entropy":
                self.loss_fn = nn.CrossEntropyLoss(reduction="none")
            else:   
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for stat_behaviour")

        # Save config
        self.config = config

    def forward(
        self,
        spikes:             torch.FloatTensor,  # (bs, seq_len, n_channels)
        time_attn_mask:     torch.LongTensor,   # (bs, seq_len)
        space_attn_mask:    torch.LongTensor,   # (bs, n_channels)
        spikes_timestamps:   torch.LongTensor,   # (bs, seq_len) 
        spikes_spacestamps:  Optional[torch.LongTensor]  = None, # (bs, n_channels) 
        spikes_lengths:     Optional[torch.LongTensor]  = None, # (bs)
        targets:            Optional[torch.FloatTensor] = None, # (bs, tar_len) 
        targets_lengths:    Optional[torch.LongTensor]  = None, # (bs)
        neuron_regions:     Optional[np.ndarray]        = None, # (bs, n_channels)
        nemo_rep:         Optional[np.ndarray]        = None,
        masking_mode:     Optional[str] = None,
        spike_augmentation: Optional[bool] = False,
        eval_mask:        Optional[torch.LongTensor] = None,
        num_neuron:       Optional[torch.LongTensor] = None,
        eid:              Optional[str] = None,
    ) -> iTransformerOutput:

        # Can hard-set the masking mode during inference. Be careful here.
        if masking_mode is not None:
            self.masker.mode = masking_mode

        if self.method == "ssl":
            # assert targets is None, "No targets needed for ssl"
            targets = spikes.clone()
        
        # change the format of neuron_regions
        if isinstance(neuron_regions, list):
            neuron_regions_np = np.asarray(neuron_regions).T

        # Encode neural data. x is the masked embedded spikes. targets_mask is True for masked bins
        spikes, targets_mask = self.masker(spikes, neuron_regions_np)

        # create the attn mask
        attn_mode = self.encoder.attn_mode
        n_heads = self.encoder.n_heads

        if self.use_prompt:
            space_attn_mask = torch.cat((torch.ones((space_attn_mask.size(0), 1), dtype=torch.bool, device=space_attn_mask.device), space_attn_mask), dim=1)
            targets_mask = torch.cat((torch.zeros((targets_mask.size(0), targets_mask.size(1), 1), dtype=torch.bool, device=targets_mask.device), targets_mask), dim=2)

        if self.use_session:
            space_attn_mask = torch.cat((torch.ones((space_attn_mask.size(0), 1), dtype=torch.bool, device=space_attn_mask.device), space_attn_mask), dim=1)
            targets_mask = torch.cat((torch.zeros((targets_mask.size(0), targets_mask.size(1), 1), dtype=torch.bool, device=targets_mask.device), targets_mask), dim=2)
        

        # experiment on 1.the order of heads and batch, 2.(solved!)the orientation of the attn mask. ***********
        pad_mask = space_attn_mask.unsqueeze(1).repeat_interleave(n_heads, dim=0).to(torch.bool).to(spikes.device)  # (bs*n_heads, 1, n_channels)

        attn_mask_inter = create_attn_mask(neuron_regions_np, "inter-region", use_prompt=self.use_prompt, use_session=self.use_session).to(spikes.device)
        attn_mask_intra = create_attn_mask(neuron_regions_np, "intra-region", use_prompt=self.use_prompt, use_session=self.use_session).to(spikes.device)
        attn_mask_none = create_attn_mask(neuron_regions_np, "all", use_prompt=self.use_prompt, use_session=self.use_session).to(spikes.device)

        if attn_mode == "mix_sample":
            p = self.encoder.attn_mix_ratio
            mode_list = ['inter-region', 'intra-region', 'all']
            attn_mode = np.random.choice(mode_list, p=p)

        if attn_mode == "mix_multihead":  # TODO: what is the order of head and batch? This mode is not fixed yet.
            h_n = self.encoder.attn_mix_n
            attn_mask = torch.cat((attn_mask_inter.repeat(h_n[0], 1, 1), attn_mask_intra.repeat(h_n[1], 1, 1),
                                   attn_mask_none.repeat(h_n[2], 1, 1)), dim=0)
            attn_mask = attn_mask.repeat(spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)  # TODO: this is probably wrong
            attn_mask = attn_mask & ~(torch.eye(attn_mask.shape[1]).to(torch.bool)).to(device=attn_mask.device).unsqueeze(0)

        elif attn_mode == "mix_sequence":
            m_list = self.encoder.attn_mix_sequence
            mask_list = []
            for i, mask_mode in enumerate(m_list):
                if mask_mode == "inter-region":
                    _tmp = attn_mask_inter.repeat(n_heads * spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
                    mask_list.append(_tmp & ~(torch.eye(_tmp.shape[1]).to(torch.bool)).to(device=_tmp.device).unsqueeze(0))
                elif mask_mode == "intra-region":
                    _tmp = attn_mask_intra.repeat(n_heads * spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
                    mask_list.append(_tmp & ~(torch.eye(_tmp.shape[1]).to(torch.bool)).to(device=_tmp.device).unsqueeze(0))
                elif mask_mode == "all":
                    _tmp = attn_mask_none.repeat(n_heads * spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
                    mask_list.append(_tmp & ~(torch.eye(_tmp.shape[1]).to(torch.bool)).to(device=_tmp.device).unsqueeze(0))
            attn_mask = mask_list

        elif attn_mode == "inter-region":
            attn_mask = attn_mask_inter.repeat(n_heads*spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
            attn_mask = attn_mask & ~(torch.eye(attn_mask.shape[1]).to(torch.bool)).to(device=attn_mask.device).unsqueeze(0)
        elif attn_mode == "intra-region":
            attn_mask = attn_mask_intra.repeat(n_heads*spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
            attn_mask = attn_mask & ~(torch.eye(attn_mask.shape[1]).to(torch.bool)).to(
                device=attn_mask.device).unsqueeze(0)
        elif attn_mode == "all":
            attn_mask = attn_mask_none.repeat(n_heads*spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
            attn_mask = attn_mask & ~(torch.eye(attn_mask.shape[1]).to(torch.bool)).to(
                device=attn_mask.device).unsqueeze(0)
        elif attn_mode == "inter-region-switch":
            # the targets_mask is already what we want
            attn_mask = attn_mask_none
            attn_mask = attn_mask | targets_mask[0, 0, :].repeat(attn_mask.shape[0], 1).transpose(0, 1)  # might be some problem with the orientation here
            attn_mask = attn_mask.repeat(n_heads*spikes.size(0), 1, 1) | ~pad_mask | (~pad_mask).transpose(1, 2)
            attn_mask = attn_mask & ~(torch.eye(attn_mask.shape[1]).to(torch.bool)).to(device=attn_mask.device).unsqueeze(0)
        
        x  = self.encoder(
            spikes, spikes_timestamps, spikes_spacestamps, 
            neuron_regions=neuron_regions, nemo_rep=nemo_rep, attention_mask=attn_mask
        )    # (batch, n_channels, hidden_size)

        _, _T, _ = spikes.size()
        _, T, _ = x.size()

        if self.use_prompt or self.use_session:
            x = x[:,T-_T:]
        
        # Predict rates/ctc-logits from embeddedings
        preds = self.decoder(x)    # (bs, n_channels, seq_len) / (bs, seq_len*vocab_size) / (bs, seq_len) / (bs, n_labels)

        if self.method == "ssl":
            preds = preds.transpose(1,2)    # (bs, seq_len, n_channels)
            targets_mask = targets_mask & time_attn_mask.unsqueeze(2) & space_attn_mask.unsqueeze(1)
            # Compute the loss only over masked timesteps that are not padded 
            loss = (self.loss_fn(preds, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()
            return iTransformerOutput(
                loss=loss,
                n_examples=n_examples,
                preds=preds,
                targets=targets,
                mask=targets_mask,
            )

        # ASK: This is weird? What is this?
        # TODO: This need to be fixed for space padding.
        elif self.method == "dyn_behaviour":
            # Include padding in mask
            targets_mask = time_attn_mask
            # Compute the loss only over timesteps that are not padded 
            loss = (self.loss_fn(preds, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()
            return iTransformerOutput(
                loss=loss,
                n_examples=n_examples,
                preds=preds,
                targets=targets,
                mask=targets_mask,
            )

        elif self.method == "stat_behaviour":
            targets_mask = targets_mask & time_attn_mask.unsqueeze(2) & space_attn_mask.unsqueeze(1)
            if self.loss_name == "cross_entropy":
                loss = self.loss_fn(preds, targets.long().squeeze(1)).sum()
            elif self.loss_name == "mse":
                loss = self.loss_fn(preds.squeeze(1), targets.squeeze(1)).sum()
            n_examples = torch.tensor(len(targets), device=loss.device, dtype=torch.long)
            return iTransformerOutput(
                loss=loss,
                n_examples=n_examples,
                preds=preds,
                targets=targets,
                mask=targets_mask,
            )
            
        elif self.method == "ctc":
            preds = preds.view(preds.shape[:1] + self.output_shape)
            loss = self.loss_fn(log_probs=preds.transpose(0,1), targets=targets.long(), input_lengths=spikes_lengths, target_lengths=targets_lengths).sum()
            n_examples = targets_lengths.sum()
            return iTransformerOutput(
                loss=loss,
                n_examples=n_examples,
                preds=preds,
                targets=targets,
            )

    def save_checkpoint(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,"encoder.bin"))
        torch.save(dict(self.config.encoder), os.path.join(save_dir,"encoder_config.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,"decoder.bin"))
        torch.save(dict(self.config.decoder), os.path.join(save_dir,"decoder_config.pth"))

    def load_checkpoint(self, load_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir,"encoder.bin")))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir,"decoder.bin")))


def create_attn_mask(
        neuron_regions: np.ndarray,  # (bs, n_channels)
        mode: str,
        use_prompt=False,
        use_session=False,
):
    regions = neuron_regions[0, :]  # Assume all batches are the same. TODO: fix this for multi-session.

    attn_mask_np = (regions[:, np.newaxis] == regions)
    attn_mask = torch.tensor(attn_mask_np, dtype=torch.bool)

    if use_prompt:
        attn_mask = torch.cat((torch.zeros((1, attn_mask.shape[1]), dtype=torch.bool, device=attn_mask.device), attn_mask), dim=0)
        attn_mask = torch.cat((torch.zeros((attn_mask.shape[0], 1), dtype=torch.bool, device=attn_mask.device), attn_mask), dim=1)

    if use_session:
        attn_mask = torch.cat((torch.zeros((1, attn_mask.shape[1]), dtype=torch.bool, device=attn_mask.device), attn_mask), dim=0)
        attn_mask = torch.cat((torch.zeros((attn_mask.shape[0], 1), dtype=torch.bool, device=attn_mask.device), attn_mask), dim=1)

    # make sure each token can at least attend to itself
    if mode == "inter-region":
        return attn_mask & ~(torch.eye(attn_mask.shape[0]).to(torch.bool)).to(device=attn_mask.device)
    elif mode == "intra-region":
        return ~attn_mask
    else:
        return torch.zeros(attn_mask.shape, dtype=torch.bool)

