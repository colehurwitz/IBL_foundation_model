import torch
import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import R2Score
from torch.nn import functional as F

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


class BaselineDecoder(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.n_units = config['n_units']
        self.n_t_steps = config['n_t_steps']
        self.learning_rate = config['optimizer']['lr']
        self.r2_score = R2Score(num_outputs=self.n_t_steps, multioutput='uniform_average')

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.r2_score(pred, y)

        self.log(f"{print_str}_loss", loss, prog_bar=True)
        self.log(f"{print_str}_r2", self.r2_score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, print_str='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    
class ReducedRankDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.temporal_rank = config['reduced_rank']['temporal_rank']
        self.weight_decay = config['reduced_rank']['weight_decay']

        self.U = torch.nn.Parameter(torch.randn(self.n_units, self.temporal_rank))
        self.V = torch.nn.Parameter(torch.randn(self.temporal_rank, self.n_t_steps, self.n_t_steps))
        self.b = torch.nn.Parameter(torch.randn(self.n_t_steps,))
        self.double()

    def forward(self, x):
        self.B = torch.einsum('nr,rtd->ntd', self.U, self.V)
        pred = torch.einsum('ntd,ktn->kd', self.B, x)
        pred += self.b
        return pred


class MLPDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.hidden_size = config['mlp']['mlp_hidden_size']
        self.weight_decay = config['mlp']['weight_decay']

        self.input_layer = torch.nn.Linear(self.n_units, self.hidden_size[0])

        self.hidden_lower = torch.nn.ModuleList()
        for l in range(len(self.hidden_size)-1):
            self.hidden_lower.append(torch.nn.Linear(self.hidden_size[l], self.hidden_size[l+1]))

        self.flat_layer = torch.nn.Linear(self.hidden_size[-1]*self.n_t_steps, self.hidden_size[0])

        self.hidden_upper = torch.nn.ModuleList()
        for l in range(len(self.hidden_size)-1):
            self.hidden_upper.append(torch.nn.Linear(self.hidden_size[l], self.hidden_size[l+1]))

        self.output_layer = torch.nn.Linear(self.hidden_size[-1], self.n_t_steps)
        
        self.double()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_lower:
            x = F.relu(layer(x))
        x = F.relu(self.flat_layer(x.flatten(start_dim=1)))
        for layer in self.hidden_upper:
            x = F.relu(layer(x))
        pred = self.output_layer(x)
        return pred
    
    
class LSTMDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.lstm_hidden_size = config['lstm']['lstm_hidden_size']
        self.n_layers = config['lstm']['lstm_n_layers']
        self.hidden_size = config['lstm']['mlp_hidden_size']
        self.drop_out = config['lstm']['drop_out']
        self.weight_decay = config['lstm']['weight_decay']

        self.lstm = torch.nn.LSTM(
            input_size=self.n_units,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.n_layers,
            dropout=self.drop_out,
            batch_first=True,
        )

        self.input_layer = torch.nn.Linear(self.lstm_hidden_size, self.hidden_size[0])
        
        self.hidden = torch.nn.ModuleList()
        for l in range(len(self.hidden_size)-1):
            self.hidden.append(torch.nn.Linear(self.hidden_size[l], self.hidden_size[l+1]))

        self.output_layer = torch.nn.Linear(self.hidden_size[-1], self.n_t_steps)
        
        self.double()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        x = F.relu(self.input_layer(lstm_out[:,-1]))
        for layer in self.hidden:
            x = F.relu(layer(x))
        pred = self.output_layer(x)
        return pred


