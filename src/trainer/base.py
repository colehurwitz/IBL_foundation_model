import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random
from pytorch_memlab import profile, MemReporter
from torch.cuda.amp import autocast, GradScaler
from math import ceil
import torch.nn.functional as F
from models.masker import Masker
from models.masker import New_Masker




class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            **kwargs
    ):
        # get all the arguments
        self.model = model
        # self.reporter = MemReporter(self.model) #NEW
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        # get arguments from kwargs if they exist
        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.stitching = kwargs.get("stitching", None)
        self.num_neurons = kwargs.get("num_neurons", None)

        self.model_class = self.config.model.model_class

        if self.config.method.model_kwargs.clf:
            self.metric = 'acc'
        elif self.config.method.model_kwargs.reg:
            self.metric = 'rsquared'
        else:
            self.metric = 'r2'
                
        self.session_active_neurons = []

        if self.model_class == 'NeuroToken':
            self.masker = New_Masker(self.config.model.encoder.masker)
            self.masking_ratio = self.masker.ratio
            self.masking_mode = self.masker.mode
        else: 
            self.masking_ratio = model.encoder.masker.ratio
            self.masking_mode = model.encoder.masker.mode

        print(f"MASKING MODE: {self.masking_mode}")

        self.masking_schemes = ['neuron', 'causal']
        if self.masking_mode == "all":
            self.masking_schemes += ['intra-region', 'inter-region']

        if self.masking_mode in ["combined", "all"]:
            print("(train) switch between masking modes: ", self.masking_schemes)



        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = GradScaler()


    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        # train loop
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                # if eval_epoch_results[f'eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss}")
                    print(f"epoch: {epoch} best eval trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    # save model
                    self.save_model(name="best", epoch=epoch)
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0], 
                            preds=eval_epoch_results['eval_preds'][0], epoch=epoch,
                            active_neurons=self.session_active_neurons[0][:5]
                        )

                        if self.config.wandb.use:
                            wandb.log({"best_epoch": epoch,
                                    "best_gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                                    "best_r2_fig": wandb.Image(gt_pred_fig['plot_r2'])})

                        else:
                            gt_pred_fig['plot_gt_pred'].savefig(
                                os.path.join(self.log_dir, f"best_gt_pred_fig_{epoch}.png")
                            )
                            gt_pred_fig['plot_r2'].savefig(
                                os.path.join(self.log_dir, f"best_r2_fig_{epoch}.png")
                            )

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

            # save model by epoch
            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                if self.config.method.model_kwargs.method_name == 'ssl':

                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0], 
                        preds=eval_epoch_results['eval_preds'][0], 
                        epoch=epoch,
                        active_neurons=self.session_active_neurons[0][:5]
                    )
                    if self.config.wandb.use:
                        wandb.log({
                            "gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                            "r2_fig": wandb.Image(gt_pred_fig['plot_r2'])
                        })
                    else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"gt_pred_fig_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"r2_fig_{epoch}.png")
                        )

            # wandb log
            if self.config.wandb.use:
                wandb.log({
                    "train_loss": train_epoch_results['train_loss'],
                    "eval_loss": eval_epoch_results['eval_loss'],
                    f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
                })
                
        # save last model
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            wandb.log({"best_eval_loss": best_eval_loss,
                       f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric})

    # @profile                #NEW
    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if self.masking_mode in ["combined", "all"]:
                masking_mode = random.sample(self.masking_schemes, 1)[0]
                if masking_mode == 'temporal':
                    if self.model_class == 'NeuroToken':
                        self.masker.ratio = 0.3
                    else: 
                        self.model.encoder.masker.ratio = 0.3
                elif masking_mode == 'causal':
                    if self.model_class == 'NeuroToken':
                        self.masker.ratio = 0.6
                    else: 
                        self.model.encoder.masker.ratio = 0.6
                else:
                    if self.model_class == 'NeuroToken':
                        self.masker.ratio = self.masking_ratio
                    else: 
                        self.model.encoder.masker.ratio = self.masking_ratio
            else:
                masking_mode = self.masking_mode
            with autocast(dtype=torch.bfloat16):
                outputs = self._forward_model_outputs(batch, masking_mode)
                loss = outputs.loss
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # self.lr_scheduler.step()
            # self.optimizer.zero_grad()
            train_loss += loss.item()
            train_examples += outputs.n_examples
        return{
            "train_loss": train_loss/train_examples
        }
    
    def _forward_model_outputs(self, batch, masking_mode):
        if self.model_class == "NeuroToken":
            self.masker.mode = masking_mode
            batch['num_neurons'] = batch['spikes_data'].shape[2]
            batch['target_spikes'] = batch['spikes_data'].clone()
            batch = self.patch_spikes(batch)
            batch['spikes_data'], batch['token_masks'], batch['target_masks'] = self.masker(batch['spikes_data'], self.config.model.encoder.embedder.max_time_F, self.config.model.encoder.transformer.hidden_size, batch['neuron_regions'])
            batch['spikes_data'] = batch['spikes_data'].reshape(self.batch_size, self.n_time_patches * batch['num_neurons'], self.config.model.encoder.embedder.max_time_F)  # Shape: (B, n_time_patches * N, n_channels)
            batch = move_batch_to_device(batch, self.accelerator.device)
            return self.model(
                batch['spikes_data'], 
                time_attn_mask=batch['time_attn_mask'],
                space_attn_mask=batch['space_attn_mask'],
                timestamps=batch['spikes_timestamps'], 
                spacestamps=batch['spikes_spacestamps'], 
                token_masks = batch['token_masks'],
                targets_mask = batch['target_masks'],
                # targets = batch['target'],
                targets = batch['target_spikes'],
                neuron_regions=batch['neuron_regions'],
                masking_mode=masking_mode, 
                spike_augmentation=self.config.data.spike_augmentation,
                num_neuron=batch['num_neurons'],
                eid=batch['eid'][0]  # each batch consists of data from the same eid
            ) 
        else: 
            batch = move_batch_to_device(batch, self.accelerator.device)
            return self.model(
                batch['spikes_data'], 
                time_attn_mask=batch['time_attn_mask'],
                space_attn_mask=batch['space_attn_mask'],
                spikes_timestamps=batch['spikes_timestamps'], 
                spikes_spacestamps=batch['spikes_spacestamps'], 
                targets = batch['target'],
                neuron_regions=batch['neuron_regions'],
                masking_mode=masking_mode, 
                spike_augmentation=self.config.data.spike_augmentation,
                num_neuron=batch['spikes_data'].shape[2],
                eid=batch['eid'][0]  # each batch consists of data from the same eid
            ) 
    
    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0.
        eval_examples = 0
        session_results = {}
        for num_neuron in self.num_neurons:
            session_results[num_neuron] = {
                "gt": [],
                "preds": []
            }
        if self.eval_dataloader:
            gt, preds = [], []
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    if self.masking_mode in ["combined", "all"]:
                        masking_mode = random.sample(self.masking_schemes, 1)[0]
                        if masking_mode == 'temporal':
                            if self.model_class == 'NeuroToken':
                                self.masker.ratio = 0.3
                            else: 
                                self.model.encoder.masker.ratio = 0.3
                        elif masking_mode == 'causal':
                            if self.model_class == 'NeuroToken':
                                self.masker.ratio = 0.6
                            else: 
                                self.model.encoder.masker.ratio = 0.6
                        else:
                            if self.model_class == 'NeuroToken':
                                self.masker.ratio = self.masking_ratio
                            else: 
                                self.model.encoder.masker.ratio = self.masking_ratio
                    else:
                        masking_mode = self.masking_mode
                    with autocast(dtype=torch.bfloat16):
                        outputs = self._forward_model_outputs(batch, masking_mode)
                        loss = outputs.loss
                    eval_loss += loss.item()
                    eval_examples += outputs.n_examples
                    if self.model_class in ['NDT1', 'iTransformer']:
                        num_neuron = batch['spikes_data'].shape[2]
                    elif self.model_class in ['NDT2', 'STPatch']:
                        num_neuron = outputs.num_neuron
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        session_results[num_neuron]["gt"].append(outputs.targets.clone()[:,:,:num_neuron])
                        session_results[num_neuron]["preds"].append(outputs.preds.clone()[:,:,:num_neuron])
                    else:
                        session_results[num_neuron]["gt"].append(outputs.targets.clone())
                        session_results[num_neuron]["preds"].append(outputs.preds.clone())
                    
            results_list = []
            for idx, num_neuron in enumerate(self.num_neurons):
                _gt = torch.cat(session_results[num_neuron]["gt"], dim=0)
                _preds = torch.cat(session_results[num_neuron]["preds"], dim=0)

                if self.config.method.model_kwargs.loss == "poisson_nll":
                    _preds = torch.exp(_preds)
                elif self.config.method.model_kwargs.loss == "cross_entropy" :
                    _preds = torch.nn.functional.softmax(_preds, dim=1)
                gt.append(_gt)
                preds.append(_preds)

                if len(self.session_active_neurons) < len(self.num_neurons):
                    active_neurons = np.argsort(gt[idx].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    self.session_active_neurons.append(active_neurons)
                if self.config.method.model_kwargs.method_name == 'ssl':
                    results = metrics_list(gt = gt[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0),
                                        pred = preds[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0), 
                                        metrics=["r2"], 
                                        device=self.accelerator.device)
                    
                elif self.config.method.model_kwargs.method_name == 'sl':
                    if self.config.method.model_kwargs.clf:
                        results = metrics_list(gt = gt[idx].argmax(1),
                                            pred = preds[idx].argmax(1), 
                                            metrics=[self.metric], 
                                            device=self.accelerator.device)
                    elif self.config.method.model_kwargs.reg:
                        results = metrics_list(gt = gt[idx],
                                            pred = preds[idx],
                                            metrics=[self.metric],
                                            device=self.accelerator.device)
                results_list.append(results[self.metric])

            return {
                "eval_loss": eval_loss/(eval_examples),
                f"eval_trial_avg_{self.metric}": np.mean(results_list),
                "eval_gt": gt,
                "eval_preds": preds,
            }
    
    def plot_epoch(self, gt, preds, epoch, active_neurons):
        gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().float().numpy(),
                    pred = preds.mean(0).T.detach().cpu().float().numpy(),
                    epoch = epoch)
        
        r2_fig = plot_neurons_r2(gt = gt.mean(0).float(),
                pred = preds.mean(0).float(),
                neuron_idx=active_neurons,
                epoch = epoch)
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }
        

    def save_model(self, name="last", epoch=0):
        # save model
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))

    def patch_spikes(self, batch):
        """
        Efficiently patches neural data by chunking in time, optimized for cases where 
        n_space_patches equals the number of neurons (max_space_F = 1).

        Args:
            spikes (torch.FloatTensor): Input spikes tensor of shape (B, T, N).
            pad_time_len (int): Padding length for the time dimension.
            time_attn_mask (torch.LongTensor): Time attention mask of shape (B, T).
            space_attn_mask (torch.LongTensor): Space attention mask of shape (B, N).
            max_time_F (int): Maximum number of time frames per patch.
            pad_value (float, optional): Value to use for padding. Defaults to -1.0.

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
                - patches (torch.FloatTensor): Patched spikes of shape (B, new_seq_len, n_channels).
                - patch_mask (torch.LongTensor): Combined attention mask for patches.
                - spacestamps (torch.LongTensor): Spatial indices for patches.
                - timestamps (torch.LongTensor): Temporal indices for patches.
        """

        spikes = batch['spikes_data']
        time_attn_mask = batch['time_attn_mask']
        max_time_F = self.config.model.encoder.embedder.max_time_F
        pad_value = -1.

        B, T, N = spikes.size()
        device = spikes.device

        self.batch_size = B
        self.n_time_patches = T // max_time_F

        # Track padded values to prevent them from being used
        if (spikes[0,:,0] == pad_value).sum() == 0:
            pad_time_len = T
        else:
            pad_time_len = (spikes[0,:,0] == pad_value).nonzero().min().item() 

        # Calculate the number of time patches
        n_time_patches = ceil(T / max_time_F)
        total_padded_length = n_time_patches * max_time_F
        pad_time_len = total_padded_length - T  # Total padding required in time dimension

        # Pad spikes along the time dimension
        spikes = F.pad(spikes, (0, 0, 0, pad_time_len), value=pad_value)  # Shape: (B, total_padded_length, N)
        # Pad time attention mask
        time_attn_mask = F.pad(time_attn_mask, (0, pad_time_len), value=0)  # Shape: (B, total_padded_length)

        # Reshape and permute spikes to shape (B, n_time_patches, N, max_time_F)
        spikes = spikes.view(B, n_time_patches, max_time_F, N).permute(0, 1, 3, 2)

        # Flatten the time dimension to create patches
        patches = spikes.reshape(B, n_time_patches, N, max_time_F)

        # # Flatten n_time_patches and N dimensions to create the sequence dimension
        # patches = patches.reshape(B, n_time_patches * N, max_time_F)  # Shape: (B, n_time_patches * N, n_channels)

        # Prepare timestamps and spacestamps
        timestamps = torch.arange(n_time_patches, device=device).unsqueeze(1).expand(n_time_patches, N).flatten()
        spacestamps = torch.arange(N, device=device).unsqueeze(0).expand(n_time_patches, N).flatten()
        # Expand to match batch size
        timestamps = timestamps.unsqueeze(0).expand(B, -1)  # Shape: (B, n_time_patches * N)
        spacestamps = spacestamps.unsqueeze(0).expand(B, -1)  # Shape: (B, n_time_patches * N)

        batch['spikes_data'] = patches
        batch['spikes_spacestamps'] = spacestamps
        batch['spikes_timestamps'] = timestamps
        batch['time_attn_mask'] = time_attn_mask

        return batch
            