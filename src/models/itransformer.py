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



class iTransformerEncoder(nn.Module):

    def __init__(
        self,
        config: DictConfig,
        use_cls: bool,
    ):
        super().__init__()


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
            # self.neuron_regions = config.neuron_regions
            # self.region_to_indx = {r: i for i,r in enumerate(self.neuron_regions)}
            # self.indx_to_region = {v: k for k,v in self.region_to_indx.items()}
            self.regionlookup = RegionLookup(config)
            self.region_embeddings = nn.Sequential(
                # nn.Embedding(len(self.region_to_indx), config.hidden_size),
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
    
        self.use_cls = use_cls
        if self.use_cls:
            self.cls_embed = nn.Embedding(1, config.hidden_size)

        self.embed_dropout = nn.Dropout(config.embedder.dropout)

        transformer_layer = TransformerEncoderLayer(
            d_model = config.hidden_size,
            nhead = config.n_heads,
            dim_feedforward = 4*config.hidden_size,
            activation=ACT2FN[config.activation],
            # bias=config.bias,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer = TransformerEncoder(
            encoder_layer = transformer_layer,
            num_layers = config.n_layers,
            norm = nn.LayerNorm(config.hidden_size),
            enable_nested_tensor = True,
        )

    def forward(
        self, 
        spikes:             torch.LongTensor,                   # (bs, seq_len, n_channels)
        spikes_timestamps:   Optional[torch.LongTensor]  = None, # (bs, seq_len) 
        spikes_spacestamps:  Optional[torch.LongTensor]  = None, # (bs, n_channels) 
        neuron_regions:     Optional[np.ndarray]        = None, # (bs, n_channels)
        nemo_rep:           Optional[np.ndarray]        = None, 
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

        # Append cls token at the beginning
        if self.use_cls:
            tokens = torch.cat((self.cls_embed(torch.zeros_like(tokens[:,:1,0]).to(torch.int64)),tokens), dim=1)    # (batch, 1+n_channels, hidden_size)

        x = self.transformer(self.embed_dropout(tokens))              # (batch, [1+]n_channels, hidden_size)

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
        
        # Build encoder
        self.encoder = iTransformerEncoder(config.encoder, config.decoder.use_cls)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))


        # Build decoder
        if self.method == "ssl":
            assert config.encoder.masker.force_active, "Can't pretrain with inactive masking"
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

        self.use_cls = config.decoder.use_cls
        if self.method in ["ctc","dyn_behaviour","stat_behaviour"] and not self.use_cls:
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
            elif self.loss_name == "xent":
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

        if self.method == "ssl":
            # assert targets is None, "No targets needed for ssl"
            targets = spikes.clone()
        
        # Encode neural data. x is the masked embedded spikes. targets_mask is True for masked bins
        spikes, targets_mask = self.masker(spikes, neuron_regions)
        

        x = self.encoder(spikes, spikes_timestamps, spikes_spacestamps, neuron_regions=neuron_regions, nemo_rep=nemo_rep)    # (batch, n_channels, hidden_size)

        # Select cls token, assumed to be the first one
        if self.use_cls:
            if self.method == "ssl":
                x = x[:,1:,:]   # remove cls token to match n_channels with input
            else:
                x = x[:,0,:]    # keep only cls token for decoding 
        
        # Predict rates/ctc-logits from embeddedings
        preds = self.decoder(x)    # (bs, n_channels, seq_len) / (bs, seq_len*vocab_size) / (bs, seq_len) / (bs, n_labels)

        if self.method == "ssl":
            preds = preds.transpose(1,2)    # (bs, seq_len, n_channels)
            targets_mask = targets_mask.to(torch.int64) & time_attn_mask.unsqueeze(2).to(torch.int64)
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
            targets_mask = targets_mask.to(torch.int64) & time_attn_mask.unsqueeze(2).to(torch.int64)
            if self.loss_name == "xent":
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
