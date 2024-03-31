import torch
import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import R2Score, Accuracy
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
        self.weight_decay = config['optimizer']['weight_decay']
        self.target = config['model']['target']
        self.output_size = config['model']['output_size']
        self.r2_score = R2Score(num_outputs=self.n_t_steps, multioutput='uniform_average')
        self.accuracy = Accuracy(task="multiclass", num_classes=self.output_size)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if self.target == 'reg':
            loss = F.mse_loss(pred, y)
        elif self.target == 'clf':
            loss = torch.nn.CrossEntropyLoss()(pred, y)
        else:
            raise NotImplementedError
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        pred = self(x)
        if self.target == 'reg':
            loss = F.mse_loss(pred, y)
            self.r2_score(pred.flatten(), y.flatten())
            self.log(f"{print_str}_metric", self.r2_score, prog_bar=True, logger=True, sync_dist=True)
        elif self.target == 'clf':
            loss = torch.nn.CrossEntropyLoss()(pred, y)
            self.accuracy(F.softmax(pred, dim=1).argmax(1), y.argmax(1))
            self.log(f"{print_str}_metric", self.accuracy, prog_bar=True, logger=True, sync_dist=True)
        else:
            raise NotImplementedError

        self.log(f"{print_str}_loss", loss, prog_bar=True)
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

        self.U = torch.nn.Parameter(torch.randn(self.n_units, self.temporal_rank))
        self.V = torch.nn.Parameter(torch.randn(self.temporal_rank, self.n_t_steps, self.output_size))
            
        self.b = torch.nn.Parameter(torch.randn(self.output_size,))
        self.double()

    def forward(self, x):
        self.B = torch.einsum('nr,rtd->ntd', self.U, self.V)
        pred = torch.einsum('ntd,ktn->kd', self.B, x)
        pred += self.b
        return pred


class MLPDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.hidden_size = tuple_type(config['mlp']['mlp_hidden_size'])
        self.drop_out = config['mlp']['drop_out']

        print(self.hidden_size[0])

        self.input_layer = torch.nn.Linear(self.n_units, self.hidden_size[0])

        self.hidden_lower = torch.nn.ModuleList()
        for l in range(len(self.hidden_size)-1):
            self.hidden_lower.append(torch.nn.Linear(self.hidden_size[l], self.hidden_size[l+1]))
            self.hidden_lower.append(torch.nn.ReLU())
            self.hidden_lower.append(torch.nn.Dropout(self.drop_out))

        self.flat_layer = torch.nn.Linear(self.hidden_size[-1]*self.n_t_steps, self.hidden_size[0])

        self.hidden_upper = torch.nn.ModuleList()
        for l in range(len(self.hidden_size)-1):
            self.hidden_upper.append(torch.nn.Linear(self.hidden_size[l], self.hidden_size[l+1]))
            self.hidden_upper.append(torch.nn.ReLU())
            self.hidden_upper.append(torch.nn.Dropout(self.drop_out))

        self.output_layer = torch.nn.Linear(self.hidden_size[-1], self.output_size)
        
        self.double()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_lower:
            x = layer(x)
        x = F.relu(self.flat_layer(x.flatten(start_dim=1)))
        for layer in self.hidden_upper:
            x = layer(x)
        pred = self.output_layer(x)
        return pred
    
    
class LSTMDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.lstm_hidden_size = config['lstm']['lstm_hidden_size']
        self.n_layers = config['lstm']['lstm_n_layers']
        self.hidden_size = tuple_type(config['lstm']['mlp_hidden_size'])
        self.drop_out = config['lstm']['drop_out']

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
            self.hidden.append(torch.nn.ReLU())
            self.hidden.append(torch.nn.Dropout(self.drop_out))

        self.output_layer = torch.nn.Linear(self.hidden_size[-1], self.output_size)
        
        self.double()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        x = F.relu(self.input_layer(lstm_out[:,-1]))
        for layer in self.hidden:
            x = layer(x)
        pred = self.output_layer(x)
        return pred


class BaselineMultiSessionDecoder(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.n_sess = len(config['n_units'])
        self.n_units = config['n_units']
        self.n_t_steps = config['n_t_steps']
        self.target = config['model']['target']
        self.output_size = config['model']['output_size']
        self.learning_rate = config['optimizer']['lr']
        self.weight_decay = config['optimizer']['weight_decay']

        self.r2_score = R2Score(num_outputs=self.n_t_steps, multioutput='uniform_average')
        self.accuracy = Accuracy(task="multiclass", num_classes=self.output_size)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        loss = torch.zeros(len(batch))
        for idx, session in enumerate(batch):
            x, y = session
            pred = self(x, idx)
            loss[idx] = torch.nn.MSELoss()(pred, y)
            if self.target == 'reg':
                loss[idx] = torch.nn.MSELoss()(pred, y)
            elif self.target == 'clf':
                loss[idx] = torch.nn.CrossEntropyLoss()(pred, y)
            else:
                raise NotImplementedError
        loss = torch.mean(loss)
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        loss, metric = torch.zeros(len(batch)), torch.zeros(len(batch))
        for idx, session in enumerate(batch):
            x, y = session
            pred = self(x, idx)
            if self.target == 'reg':
                loss[idx] = torch.nn.MSELoss()(pred, y)
                metric[idx] = self.r2_score(pred.flatten(), y.flatten())
            elif self.target == 'clf':
                loss = torch.nn.CrossEntropyLoss()(pred, y)
                metric[idx] = self.accuracy(F.softmax(pred, dim=1).argmax(1), y.argmax(1))
            else:
                raise NotImplementedError
        loss, metric = torch.mean(loss), torch.mean(metric)

        self.log(f"{print_str}_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{print_str}_metric", metric, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, print_str='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


class MultiSessionReducedRankDecoder(BaselineMultiSessionDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.temporal_rank = config['temporal_rank']

        self.Us = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(n_units, self.temporal_rank)) for n_units in self.n_units]
        )
        self.V = torch.nn.Parameter(torch.randn(self.temporal_rank, self.n_t_steps, self.output_size))
        self.bs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(self.output_size,)) for _ in range(self.n_sess)]
        )
        self.double()

    def forward(self, x, idx):
        B = torch.einsum('nr,rtd->ntd', self.Us[idx], self.V)
        pred = torch.einsum('ntd,ktn->kd', B, x)
        pred += self.bs[idx]
        return pred
        