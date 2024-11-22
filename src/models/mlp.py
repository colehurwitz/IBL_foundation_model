import torch
import torch.nn as nn
# MLP
class NeuralMLP(nn.Module):

    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()

        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = nn.ReLU() if act == 'relu' else nn.Tanh()
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, timestamp):
        
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))