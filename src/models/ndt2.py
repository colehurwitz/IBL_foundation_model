import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadinMatrix(nn.Module):
    '''
    Linear projection to embed input spiking data. 
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class ContextualMLP(nn.Module):
    '''
    Linear projection to embed context data. 
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class ReadinCrossAttention(nn.Module):
    '''
    Cross-attention between the spiking data embeddings and context embeddings.
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class PositionalEncoding(nn.Module):
    '''
    Create space and time positional embeddings. 
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class FlippedDecoderLayer(nn.TransformerDecoderLayer):
    '''
    Perform cross-attn then self-attn rather than self-attn then cross-attn.
    Intuition is to add session context before self-attn. 
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class Masker(nn.Module):
    '''
    Mask spikes: (1) neuron patch (2) time step (3) random.
    '''
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class SpaceTimeTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def fixup_initialization(self):
        pass

    def forward(self):
        pass


class NDT2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, save_dir):
        pass


    
    